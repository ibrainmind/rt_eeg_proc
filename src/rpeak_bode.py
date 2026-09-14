#!/usr/bin/env python3
"""
rpeak_bode.py -- strict-causal PI prediction error against a NON-CAUSAL R-peak
reference, and the beat-domain loop analysis that says how to improve it.

Reference.  The streaming estimator's fiducial is itself causal, so it is not a
fair yardstick.  We build a non-causal reference by zero-phase filtering the
whole ECG (filtfilt, 8-20 Hz), picking peaks on the envelope, refining each to
the local |ECG| extremum and then to sub-sample resolution by parabolic fit.
Every error below is measured against that.

Model.  In the beat domain the loop is linear in the timing error.  Let T_k be
the true interval and That_k the loop's estimate.  The period estimator is an
EWMA with gain alpha, so its one-step error is

    (That - T)(z) / T(z) = (z^-1 - 1) / (1 - (1-alpha) z^-1),

a high-pass: constant heart rate is tracked exactly, variation is not.  The
phase corrector removes only Kp of the residual timing error per beat, so the
error also passes through a leaky integrator

    1 / (1 - (1-Kp) z^-1),      DC gain 1/Kp.

The strict-causal anchor therefore sees

    S(z) = [1 / (1 - (1-Kp) z^-1)] * [(z^-1 - 1) / (1 - (1-alpha) z^-1)]

with Kp = alpha = 0.30 as shipped.  Re-anchoring on the detected fiducial each
beat is Kp = 1, which removes the 1/Kp amplification; alpha = 1 is the plain
previous-RR predictor.  We validate S by filtering the MEASURED T_k sequence
through it and comparing the resulting SD with the SD actually observed.

  python rpeak_bode.py --out ../outputs/rpeak_bode.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

import dataset_metric as dm
import ekg_ilc_rt as C
import phase_estimator_eval as PE

RESP_BAND_HZ = (0.15, 0.40)      # respiratory sinus arrhythmia
RHO = 0.96                       # resonator pole radius for the IMP design
ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ---------------------------------------------------------------------------
# non-causal R-peak reference
# ---------------------------------------------------------------------------
def noncausal_rpeaks(r, fs, min_rr_s=0.30):
    """Offline R-peak instants (in samples, sub-sample refined)."""
    sos = signal.butter(2, [8, 20], btype="band", fs=fs, output="sos")
    env = np.abs(signal.sosfiltfilt(sos, r))
    env = signal.savgol_filter(env, max(3, int(0.05 * fs) | 1), 2)
    thr = np.median(env) + 0.5 * (np.percentile(env, 98) - np.median(env))
    pk, _ = signal.find_peaks(env, height=thr, distance=int(min_rr_s * fs))
    w = int(round(0.06 * fs))
    out = []
    for p in pk:
        a, b = max(0, p - w), min(len(r), p + w + 1)
        seg = r[a:b] - np.mean(r[a:b])
        i = a + int(np.argmax(np.abs(seg)))
        out.append(i + PE.parabolic_offset(np.abs(r - np.mean(r)), i))
    return np.asarray(out, dtype=float)


def match(pred_samples, ref_samples, tol_samples):
    """Nearest reference peak for each predicted instant, within tol."""
    ref = np.asarray(ref_samples)
    idx = np.searchsorted(ref, pred_samples)
    lo = np.clip(idx - 1, 0, len(ref) - 1)
    hi = np.clip(idx, 0, len(ref) - 1)
    dl = np.abs(pred_samples - ref[lo]); dh = np.abs(pred_samples - ref[hi])
    j = np.where(dl <= dh, lo, hi)
    e = pred_samples - ref[j]
    ok = np.abs(e) <= tol_samples
    return e, ok, j


# ---------------------------------------------------------------------------
# beat-domain error model
# ---------------------------------------------------------------------------
def ewma_sensitivity(alpha: float):
    """One-step period-prediction error of an EWMA tracker: (z^-1 - 1)/(1-(1-a)z^-1)."""
    return np.array([-1.0, 1.0]), np.array([1.0, -(1.0 - alpha)])


def alphabeta_sensitivity(a1: float, a2: float):
    """Same for an alpha-beta tracker that also carries a period RATE.

    S(z) = (1-z^-1)^2 / [1 + (a1-2) z^-1 + (1-a1+a2) z^-2]: a DOUBLE zero at DC,
    so a linearly drifting heart rate is predicted with zero steady-state error,
    where the EWMA leaves a standing lag.
    """
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([1.0, a1 - 2.0, 1.0 - a1 + a2])
    return b, a


def phase_leak(kp: float):
    """The phase corrector removes only kp of the residual error per beat."""
    return np.array([1.0]), np.array([1.0, -(1.0 - kp)])


def error_filter(kp: float, alpha=None, ab=None, extra=None):
    """(b, a) of S(z): RR sequence -> predicted-R timing error."""
    if ab is not None:
        b, a = alphabeta_sensitivity(*ab)
    else:
        b, a = ewma_sensitivity(alpha)
    lb, la = phase_leak(kp)
    b, a = np.convolve(b, lb), np.convolve(a, la)
    if extra is not None:
        b, a = np.convolve(b, extra[0]), np.convolve(a, extra[1])
    return b, a


def resonator(f_cycles_per_beat: float, rho: float = RHO):
    """Notch at the respiratory line: zeros on the unit circle, poles just inside."""
    w = 2 * np.pi * f_cycles_per_beat
    b = np.array([1.0, -2 * np.cos(w), 1.0])
    a = np.array([1.0, -2 * rho * np.cos(w), rho ** 2])
    return b, a


def predicted_sd(T_ms, burn=20, **kw):
    """SD of the timing error obtained by driving S(z) with the MEASURED RR."""
    b, a = error_filter(**kw)
    x = np.asarray(T_ms, float) - float(np.mean(T_ms))
    e = signal.lfilter(b, a, x)
    return float(np.std(e[burn:])), e


DESIGNS = {
    "as_built":     dict(kp=0.30, alpha=0.30),   # shipped
    "reanchored":   dict(kp=1.00, alpha=0.30),   # fix the phase gain only
    "previous_rr":  dict(kp=1.00, alpha=1.00),   # fix both: deadbeat predictor
    "alphabeta":    dict(kp=1.00, ab=(0.80, 0.05)),
}

# Note on the alpha-beta family: at a2 = 0 its denominator factors as
# (z-1)(z-(1-a1)), i.e. a pole ON the unit circle for any a1, so the apparent
# optimum of a sweep that includes a2 = 0 is a marginally stable predictor and
# must be discarded.  a2 = 0.05 keeps the poles inside.


# ---------------------------------------------------------------------------
def analyse_run(r, fs, label, duration_sec=600.0) -> dict | None:
    r = r[: int(round(duration_sec * fs))]
    est = C.phase_estimator(r, fs, use_pll=True, use_override=True, debug=True)
    fids = est["fids"]
    det = np.flatnonzero(est["detect_events"])
    phi, rrh = est["phi"], est["rrhat"]
    if len(fids) < 60 or len(det) != len(fids):
        return None

    F = np.array([f[0] for f in fids], dtype=int)
    LG = np.array([f[2] for f in fids], dtype=bool)
    ref = noncausal_rpeaks(r, fs)
    if len(ref) < 60:
        return None
    # Timing error is inherently modulo the beat, so the matching tolerance has
    # to scale with RR.  A fixed window silently censors the tail and makes the
    # measured spread look smaller than it is.
    tol = int(round(0.45 * float(np.median(np.diff(ref)))))

    # streaming fiducial (what the buffered mode anchors on) vs the offline peak
    e_fid, ok_fid, jref = match(F.astype(float), ref, tol)

    # strict-causal predicted R-peak: the loop believes the R-peak sits
    # -wrap(phi)*RRhat away from now, so its predicted instant is that displaced
    # point.  Measured against the SAME offline reference.
    pred = np.array([F[k] - PE._wrap(float(phi[F[k]])) * float(rrh[F[k]]) * fs
                     for k in range(len(F))], dtype=float)
    e_pll, ok_pll, _ = match(pred, ref, tol)

    # re-anchored alternative: previous fiducial + RR estimate
    ol = np.array([np.nan] + [F[k - 1] + rrh[min(int(det[k - 1]) + 1, len(r) - 1)] * fs
                              for k in range(1, len(F))], dtype=float)
    ok_ol = np.isfinite(ol)
    e_ol = np.full(len(F), np.nan)
    e_ol[ok_ol], m_ol, _ = match(ol[ok_ol], ref, tol)
    ok_ol[ok_ol] = m_ol

    sel = LG & ok_pll & ok_fid
    if sel.sum() < 40:
        return None
    to_ms = 1e3 / fs

    # RR series from the NON-CAUSAL reference (the true intervals).  Intervals
    # that the arrhythmia supervisor would reject are replaced by the running
    # median, so the linear model is driven by the same beats the measured error
    # is scored on -- otherwise the model sees excursions the loop never tracks.
    Tref_ms = np.diff(ref) * to_ms
    Tref_ms = Tref_ms[(Tref_ms > 300) & (Tref_ms < 2000)]
    med_T = float(np.median(Tref_ms))
    out_frac = float(np.mean(np.abs(Tref_ms - med_T) > 0.30 * med_T))
    Tref_ms = np.where(np.abs(Tref_ms - med_T) > 0.30 * med_T, med_T, Tref_ms)

    def stats(v_ms, prefix, d):
        """Bias and, separately, spread about it.

        The fiducial can sit a constant offset from the offline peak when the
        armed window closes before the broadest deflection.  That offset is
        common to every anchoring scheme -- the template simply absorbs it -- so
        the quantity that matters is the spread, reported de-biased.
        """
        v = np.asarray(v_ms, float)
        m = float(np.median(v))
        d[f"{prefix}_bias_ms"] = m
        d[f"{prefix}_sd_ms"] = float(np.std(v - m))
        d[f"{prefix}_mad_ms"] = 1.4826 * float(np.median(np.abs(v - m)))
        d[f"{prefix}_abs_p90_ms"] = float(np.percentile(np.abs(v - m), 90))

    out = {
        "label": label, "fs": fs, "n_beats": int(len(F)),
        "n_ref": int(len(ref)),
        "match_rate": float(np.mean(ok_fid)),
        "rr_mean_ms": float(np.mean(Tref_ms)), "rr_sd_ms": float(np.std(Tref_ms)),
        "rr_outlier_frac": out_frac,
    }
    stats(e_fid[sel] * to_ms, "fid", out)
    stats(e_pll[sel] * to_ms, "pll", out)
    s2 = sel & ok_ol
    if s2.sum() > 40:
        stats(e_ol[s2] * to_ms, "ol", out)

    # --- model validation and design sweep -------------------------------
    for name, kw in DESIGNS.items():
        sd, _ = predicted_sd(Tref_ms, **kw)
        out[f"model_{name}_sd_ms"] = sd

    # where the RR disturbance actually is
    f, P = signal.welch(Tref_ms - np.mean(Tref_ms), fs=1.0,
                        nperseg=min(256, len(Tref_ms) // 4 * 2))
    out["rr_spec_f"] = f.tolist()
    out["rr_spec_P"] = P.tolist()
    band = (f > 0.05) & (f < 0.5)
    f_peak = float(f[band][int(np.argmax(P[band]))])
    out["rr_peak_cycles_per_beat"] = f_peak
    out["rr_peak_hz"] = f_peak / (out["rr_mean_ms"] / 1e3)
    sd_res, _ = predicted_sd(Tref_ms, kp=1.0, alpha=1.0, extra=resonator(f_peak))
    out["model_resonator_sd_ms"] = sd_res
    # design curve: error SD against the period-estimator gain, both phase gains
    for kp in (0.30, 1.00):
        key = "kp03" if kp < 0.5 else "kp10"
        out[f"sweep_{key}"] = [predicted_sd(Tref_ms, kp=kp, alpha=al)[0]
                               for al in ALPHA_GRID]
    out["_Tref_ms"] = Tref_ms.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--subject", default="sub-070")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--runs", default="")
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--out", default="../outputs/rpeak_bode.json")
    a = ap.parse_args()

    root = dm._resolve_dataset_root(a.dataset_root)
    runs = [x for x in a.runs.split(",") if x] or dm.discover_runs(root, a.subject, a.session)
    res = []
    for run in runs:
        try:
            r, fs, ch = PE.load_ecg(root, a.subject, a.session, run)
        except Exception as e:
            print(f"  [skip] run-{run}: {type(e).__name__}: {e}", flush=True)
            continue
        o = analyse_run(r, fs, f"{a.subject}/{a.session}/run-{run}", a.duration_sec)
        if o is None:
            print(f"  [skip] run-{run}: insufficient / unmatched", flush=True)
            continue
        o["run"] = run
        res.append(o)
        print(f"  run-{run}: match {100*o['match_rate']:.1f}%  "
              f"fid {o['fid_bias_ms']:+.1f}+-{o['fid_mad_ms']:.1f}  "
              f"PLL sd {o['pll_sd_ms']:.1f}  ol sd {o.get('ol_sd_ms',float('nan')):.1f}  "
              f"| model as-built {o['model_as_built_sd_ms']:.1f} "
              f"reanch {o['model_reanchored_sd_ms']:.1f} "
              f"prevRR {o['model_previous_rr_sd_ms']:.1f} "
              f"ab {o['model_alphabeta_sd_ms']:.1f}  "
              f"RSA {o['rr_peak_hz']:.2f} Hz", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {a.out} ({len(res)} runs)")


if __name__ == "__main__":
    main()
