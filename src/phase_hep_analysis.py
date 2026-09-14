#!/usr/bin/env python3
"""
phase_hep_analysis.py -- two measurements the CFA paper needs
=============================================================

(A) PHASE REPRESENTATION.  Is the template better indexed by time-since-R
    (fixed lag, what AAS does) or by normalized cardiac phase phi=(t-t_k)/T_k?
    Both are special cases of a one-parameter warp

        offset p on beat k  <->  phase coordinate  c = p / w_k,
        w_k = (T_k / T_bar)**g,

    with g = 0 fixed lag, g = 1 fully normalized phase, g = 1/2 a Bazett-like
    square-root warp (QT scales ~ sqrt(RR), so systole is neither time-locked
    nor proportional).  We sweep g and score by SPLIT-HALF HELD-OUT variance
    reduction in the time domain -- template built on even beats, residual
    measured on odd beats and vice versa -- so no ground truth is needed and
    every g is scored on the identical time-domain samples.

(B) HEP EXPOSURE.  How much R-locked energy does the [-50, +550] ms template
    remove from the 200-600 ms window where the heartbeat-evoked potential
    lives, and what does a QRS-limited [-50, +200] ms template give up?
    Amplitudes are bias-corrected against the circular-shift scramble null of
    dataset_metric.py, and reported in microvolts so they can be compared to
    the ~1 uV scale of the HEP literature.

CLI:
  python phase_hep_analysis.py --subject sub-070 --run 20 --eeg-index 0 \
      --duration-sec 600 --out ../outputs/phase_hep_sub-070_run-20.json
  python phase_hep_analysis.py --scan --subjects sub-001,sub-002,... \
      --out ../outputs/phase_hep_scan.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dataset_metric as dm
import ekg_ilc_rt as rt

# Analysis window relative to R (seconds).  It is chosen so that the QRS and HEP
# sub-windows PARTITION it exactly -- otherwise the "share of removable energy in
# the HEP interval" is a ratio of two different supports and can exceed one.
PRE_S = 0.05
POST_S = 0.60
QRS_LO_S, QRS_HI_S = -0.05, 0.20      # what a QRS-limited template keeps
HEP_LO_S, HEP_HI_S = 0.20, 0.60       # where the heartbeat-evoked potential lives
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------------
# (A) warp-family template: build / predict
# ---------------------------------------------------------------------------
def _beat_periods(r_samples: np.ndarray, fs: float, n: int) -> np.ndarray:
    """Per-beat period T_k in seconds; last beat inherits the previous period."""
    d = np.diff(r_samples).astype(float) / fs
    if len(d) == 0:
        return np.full(len(r_samples), np.nan)
    return np.append(d, d[-1])


def build_warp_template(
    y: np.ndarray,
    r_samples: np.ndarray,
    T: np.ndarray,
    fs: float,
    gamma: float,
    T_bar: float,
    c_grid: np.ndarray,
) -> np.ndarray:
    """Average y over beats on the warped coordinate grid c_grid (seconds at T_bar).

    Beat k contributes y(t_k + c * w_k) for each c in c_grid, w_k = (T_k/T_bar)**gamma.
    gamma=0 collapses to plain fixed-lag averaging at offsets c.
    """
    N = len(y)
    tn = np.arange(N, dtype=float) / fs
    acc = np.zeros(len(c_grid))
    cnt = np.zeros(len(c_grid))
    for fid, Tk in zip(r_samples, T):
        if not np.isfinite(Tk) or Tk <= 0:
            continue
        w = (Tk / T_bar) ** gamma
        ts = fid / fs + c_grid * w
        ok = (ts >= 0.0) & (ts <= tn[-1])
        if not np.any(ok):
            continue
        acc[ok] += np.interp(ts[ok], tn, y)
        cnt[ok] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


def predict_beat(
    template: np.ndarray,
    c_grid: np.ndarray,
    offsets_s: np.ndarray,
    Tk: float,
    gamma: float,
    T_bar: float,
) -> np.ndarray:
    """Warp the template back onto real time offsets (seconds) of one beat."""
    w = (Tk / T_bar) ** gamma
    return np.interp(offsets_s / w, c_grid, template)


def eval_gamma(
    y: np.ndarray,
    r_samples: np.ndarray,
    T: np.ndarray,
    fs: float,
    gamma: float,
    T_bar: float,
    c_grid: np.ndarray,
) -> dict:
    """Split-half held-out residual variance reduction, overall / QRS / HEP window.

    Template from even beats scores odd beats and vice versa.  The evaluation
    offsets are identical for every gamma, and each beat's window is clipped at
    the next R-peak so no sample is scored twice.
    """
    idx = np.arange(len(r_samples))
    folds = [(idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)]
    N = len(y)
    tn = np.arange(N, dtype=float) / fs

    bands = {
        "all": (-PRE_S, POST_S),
        "qrs": (QRS_LO_S, QRS_HI_S),
        "hep": (HEP_LO_S, HEP_HI_S),
    }
    num = {b: 0.0 for b in bands}   # residual energy
    den = {b: 0.0 for b in bands}   # original energy

    for train_m, test_m in folds:
        tmpl = build_warp_template(
            y, r_samples[train_m], T[train_m], fs, gamma, T_bar, c_grid
        )
        for fid, Tk in zip(r_samples[test_m], T[test_m]):
            if not np.isfinite(Tk) or Tk <= 0:
                continue
            hi = min(POST_S, Tk)      # clip at next R so windows do not overlap
            off = np.arange(-int(round(PRE_S * fs)), int(round(hi * fs))) / fs
            ts = fid / fs + off
            ok = (ts >= 0.0) & (ts <= tn[-1])
            if not np.any(ok):
                continue
            off, ts = off[ok], ts[ok]
            seg = np.interp(ts, tn, y)
            pred = predict_beat(tmpl, c_grid, off, Tk, gamma, T_bar)
            res = seg - pred
            for b, (lo, hi_b) in bands.items():
                m = (off >= lo) & (off < hi_b)
                if not np.any(m):
                    continue
                num[b] += float(np.sum(res[m] ** 2))
                den[b] += float(np.sum(seg[m] ** 2))

    out = {"gamma": gamma}
    for b in bands:
        # Variance-reduction in dB: how much beat-window energy the held-out
        # template actually removes.  Positive = energy removed.
        out[f"vr_db_{b}"] = (
            10.0 * np.log10(den[b] / num[b]) if num[b] > 0 and den[b] > 0 else np.nan
        )
    return out


# ---------------------------------------------------------------------------
# (B) HEP-window exposure against the scramble null
# ---------------------------------------------------------------------------
def window_energy(template: np.ndarray, c_grid: np.ndarray, lo: float, hi: float) -> tuple[float, int]:
    m = (c_grid >= lo) & (c_grid < hi)
    return float(np.sum(template[m] ** 2)), int(np.sum(m))


def hep_exposure(
    y: np.ndarray,
    r_samples: np.ndarray,
    T: np.ndarray,
    fs: float,
    gamma: float,
    T_bar: float,
    c_grid: np.ndarray,
    n_scramble: int = 100,
    seed: int = 0,
    min_shift_s: float = 2.0,
) -> dict:
    """Bias-corrected R-locked RMS in the QRS and HEP windows, vs a scramble null."""
    N = len(y)
    tmpl = build_warp_template(y, r_samples, T, fs, gamma, T_bar, c_grid)

    bands = {
        "qrs": (QRS_LO_S, QRS_HI_S),
        "hep": (HEP_LO_S, HEP_HI_S),
        "all": (-PRE_S, POST_S),
    }
    e_true = {}
    L = {}
    for b, (lo, hi) in bands.items():
        e_true[b], L[b] = window_energy(tmpl, c_grid, lo, hi)

    rng = np.random.default_rng(seed)
    min_shift = int(round(min_shift_s * fs))
    e_scr = {b: np.empty(n_scramble) for b in bands}
    for i in range(n_scramble):
        scr = dm.scramble_samples(r_samples, N, rng, min_shift)
        tscr = build_warp_template(y, scr, T, fs, gamma, T_bar, c_grid)
        for b, (lo, hi) in bands.items():
            e_scr[b][i], _ = window_energy(tscr, c_grid, lo, hi)

    sd = float(np.std(y))
    out = {"sd_uV": sd * 1e6}
    for b in bands:
        m = float(np.mean(e_scr[b]))
        out[f"snr_db_{b}"] = 10.0 * np.log10(e_true[b] / (m + 1e-30))
        out[f"p_{b}"] = float(np.mean(e_scr[b] >= e_true[b]))
        rms = np.sqrt(max(e_true[b] - m, 0.0) / max(L[b], 1))
        out[f"rms_uV_{b}"] = rms * 1e6
        out[f"rms_over_sd_{b}"] = rms / (sd + 1e-30)
    # Fraction of removable R-locked energy that a QRS-limited template forgoes.
    num = max(e_true["hep"] - float(np.mean(e_scr["hep"])), 0.0)
    den = max(e_true["all"] - float(np.mean(e_scr["all"])), 0.0)
    out["hep_energy_fraction"] = num / den if den > 0 else np.nan
    out["template"] = (tmpl * 1e6).tolist()
    out["c_grid"] = c_grid.tolist()
    return out


# ---------------------------------------------------------------------------
# per-run driver
# ---------------------------------------------------------------------------
def _prepare(y, r, fs, n_scramble):
    """Shared per-run work: learnable beats, periods, warped grid."""
    est = rt.phase_estimator(r, fs, use_pll=True, use_override=True)
    r_samples = np.asarray([f[0] for f in est["fids"] if f[2]], dtype=int)
    if len(r_samples) < 40:
        return None
    T = _beat_periods(r_samples, fs, len(y))
    good = np.isfinite(T) & (T > 0.3) & (T < 2.0)
    r_samples, T = r_samples[good], T[good]
    if len(r_samples) < 40:
        return None
    T_bar = float(np.mean(T))
    # Warped-coordinate grid, at the sample spacing of the mean beat, wide enough
    # that every beat's [-PRE, POST] window maps inside it.
    w_min = float(np.min(T / T_bar))
    c_lo = -PRE_S / max(w_min, 1e-3) - 0.02
    c_hi = POST_S / max(w_min, 1e-3) + 0.02
    return r_samples, T, T_bar, np.arange(c_lo, c_hi, 1.0 / fs)


def _stats(r_samples, T, T_bar, fs, label, subject, session, run, eeg_index):
    med = float(np.median(T))
    mad = float(np.median(np.abs(T - med)))
    return {
        "label": label, "subject": subject, "session": session, "run": run,
        "eeg_index": eeg_index, "fs": fs, "n_beats": int(len(r_samples)),
        "rr_mean_s": T_bar, "rr_sd_s": float(np.std(T)),
        "rr_cv": float(np.std(T) / T_bar),
        "rr_min_s": float(np.min(T)), "rr_max_s": float(np.max(T)),
        # Robust HRV: a missed or doubled detection inflates the plain SD far more
        # than genuine RSA does, so the sweep is indexed by a MAD-based CV and the
        # outlier share is reported alongside it.
        "rr_median_s": med,
        "rr_cv_robust": 1.4826 * mad / med,
        "rr_outlier_frac": float(np.mean(np.abs(T - med) > 0.3 * med)),
    }


def analyse_channel(y, r_samples, T, T_bar, c_grid, fs, meta, n_scramble, keep_template):
    res = dict(meta)
    res["gamma_sweep"] = [
        eval_gamma(y, r_samples, T, fs, g, T_bar, c_grid) for g in GAMMAS
    ]
    res["hep"] = hep_exposure(
        y, r_samples, T, fs, 1.0, T_bar, c_grid, n_scramble=n_scramble
    )
    if not keep_template:
        res["hep"].pop("template", None)
        res["hep"].pop("c_grid", None)
    return res


def analyse_run(
    dataset_root: Path,
    subject: str,
    session: str,
    run: str,
    eeg_index: int,
    ecg_index: int,
    duration_sec: float,
    start_sec: float = 0.0,
    n_scramble: int = 100,
    pull: bool = False,
) -> dict | None:
    y, r, fs, label = dm.load_eeg_ecg(
        dataset_root, subject, session, run, eeg_index, ecg_index, pull,
        highpass_hz=0.5, notch_hz=50.0, start_sec=start_sec, duration_sec=duration_sec,
    )
    prep = _prepare(y, r, fs, n_scramble)
    if prep is None:
        return None
    r_samples, T, T_bar, c_grid = prep
    meta = _stats(r_samples, T, T_bar, fs, label, subject, session, run, eeg_index)
    return analyse_channel(y, r_samples, T, T_bar, c_grid, fs, meta, n_scramble, True)


def analyse_run_all_channels(
    dataset_root: Path, subject: str, session: str, run: str, ecg_index: int,
    duration_sec: float, start_sec: float = 0.0, n_scramble: int = 100,
    pull: bool = False,
) -> list[dict]:
    """Load one run once and analyse every EEG channel against the same beat train."""
    import process_data as pd
    rd = pd.load_run_data(dataset_root, subject=subject, session=session, run=run, pull=pull)
    ecg = rd.ecg[ecg_index]
    fs = float(ecg.fs)
    max_dur = min(min(len(e.values) / e.fs for e in rd.eeg), len(ecg.values) / ecg.fs)
    n_out = max(2, int(np.floor(max_dur * fs)))
    r = np.asarray(ecg.values[:n_out], dtype=float)
    t = np.arange(n_out) / fs
    a = max(0.0, start_sec)
    b = t[-1] if duration_sec is None else min(t[-1], a + duration_sec)
    m = (t >= a) & (t <= b)
    r = r[m]

    prep = None
    out = []
    for ei, e in enumerate(rd.eeg):
        if e.fs == fs:
            y = np.asarray(e.values[:n_out], dtype=float)
        else:
            y = pd._resample_linear(np.asarray(e.values, dtype=float), e.fs, fs, n_out)
        y = pd._mne_preprocess_trace(y, fs, highpass_hz=0.5, notch_hz=50.0)[m]
        if prep is None:
            prep = _prepare(y, r, fs, n_scramble)
            if prep is None:
                return []
        r_samples, T, T_bar, c_grid = prep
        label = f"{subject}/{session}/run-{run} EEG={e.name} ECG={ecg.name}"
        meta = _stats(r_samples, T, T_bar, fs, label, subject, session, run, ei)
        out.append(analyse_channel(y, r_samples, T, T_bar, c_grid, fs, meta,
                                   n_scramble, False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--subject", default="sub-070")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--run", default="20")
    ap.add_argument("--eeg-index", type=int, default=0)
    ap.add_argument("--ecg-index", type=int, default=0)
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--n-scramble", type=int, default=100)
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--scan", action="store_true", help="loop subjects x runs")
    ap.add_argument("--subjects", default="", help="comma list for --scan")
    ap.add_argument("--max-runs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = dm._resolve_dataset_root(a.dataset_root)
    results = []
    if a.scan:
        subs = [s for s in a.subjects.split(",") if s] or dm.discover_subjects(root, a.session)
        for sub in subs:
            try:
                runs = dm.discover_runs(root, sub, a.session)[: a.max_runs]
            except Exception as e:
                print(f"[skip] {sub}: {e}")
                continue
            for run in runs:
                try:
                    chans = analyse_run_all_channels(
                        root, sub, a.session, run, a.ecg_index, a.duration_sec,
                        a.start_sec, a.n_scramble, a.pull)
                except Exception as e:
                    print(f"[skip] {sub}/{run}: {type(e).__name__}: {e}", flush=True)
                    continue
                for r in chans:
                    results.append(r)
                    g = {d["gamma"]: d["vr_db_all"] for d in r["gamma_sweep"]}
                    print(f"{sub}/run-{run}/eeg{r['eeg_index']}  N={r['n_beats']:4d} "
                          f"rrCV={r['rr_cv_robust']:.3f}  vr(g=0)={g[0.0]:+.2f}dB "
                          f"vr(g=1)={g[1.0]:+.2f}dB  "
                          f"HEPsnr={r['hep']['snr_db_hep']:+.1f}dB "
                          f"HEPrms={r['hep']['rms_uV_hep']:.2f}uV", flush=True)
    else:
        r = analyse_run(root, a.subject, a.session, a.run, a.eeg_index, a.ecg_index,
                        a.duration_sec, a.start_sec, a.n_scramble, a.pull)
        if r is None:
            print("not enough learnable beats")
            return
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "hep"}, indent=2, default=float))
        print(json.dumps({k: v for k, v in r["hep"].items()
                          if k not in ("template", "c_grid")}, indent=2, default=float))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1, default=float))
        print(f"wrote {a.out}  ({len(results)} entries)")


if __name__ == "__main__":
    main()
