#!/usr/bin/env python3
"""
phase_swap_compare.py -- what does phase warping buy the Figure-1 canceller?

Figure 1 of the paper is produced by ekg_ilc_rt.ilc_clean(mode="noncausal"),
which indexes its template by fixed lag from the R-peak.  ilc_clean now takes a
warp exponent gamma (0 = fixed lag, exactly the original code path; 1 = the
template is stretched to each beat, i.e. normalized cardiac phase).  This script
runs the same canceller four ways -- {non-causal, one-beat-delay} x {gamma=0,
gamma=1} -- and reports what changes.

Two settings, because the answer depends entirely on how the artifact is
generated:

  SYNTHETIC, with ground truth.  gen_eeg places a FIXED-SHAPE PQRST at each beat,
    so its artifact is strictly time-locked and gamma=1 can only lose.  We
    therefore also run a stretched variant, in which each beat's waveform is
    scaled in time by T_k/Tbar, where gamma=1 is correct by construction.  The
    two bracket the real answer.

  REAL, on the record Figure 1 is drawn from.  No ground truth, so we measure
    the R-locked energy the canceller removes, main lobe and late window, against
    a circular-shift null.  Both arms have the same number of template bins, so
    the comparison is not confounded by degrees of freedom -- but the non-causal
    template must be scored HELD OUT (built on even beats, measured on odd, and
    vice versa).  Subtracting a template from the beats that built it drives the
    R-triggered average of the residual to zero identically at gamma=0, which
    measures nothing.  The buffered arm needs no such care: it subtracts before
    it updates, so each bin is already scored against beats it has not seen.

  python phase_swap_compare.py --out ../outputs/phase_swap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

import dataset_metric as dm
import ekg_ilc_rt as C

FS = 250.0
DUR = 180.0
GAMMAS = (0.0, 0.5, 1.0)
MODES = ("noncausal", "buffered")


# ---------------------------------------------------------------------------
# synthetic: time-locked (as shipped) vs phase-stretched artifact
# ---------------------------------------------------------------------------
def gen_eeg_variant(fs, dur, seed, stretch: bool, rsa: float = 0.08,
                    drift: float = 0.20, artifact_gain: float = 3.0):
    """gen_eeg, but optionally stretching each beat's waveform by T_k/Tbar."""
    d, beat_t, btype, rng = C.gen_sources(fs, dur, seed, anomalies=False, rsa=rsa)
    N = int(fs * dur)
    t = np.arange(N) / fs

    if stretch:
        T = np.diff(beat_t)
        T = np.append(T, T[-1])
        Tbar = float(np.mean(T))
        d = np.zeros(N)
        r0 = int(0.30 * fs)
        base = C._pqrst(fs)                      # fixed-shape prototype
        tb = (np.arange(len(base)) - r0) / fs    # its time axis, s from R
        for bt, Tk in zip(beat_t, T):
            w = Tk / Tbar
            idx = int(round(bt * fs))
            lo, hi = idx - int(round(r0 * w)), idx + int(round((len(base) - r0) * w))
            n = np.arange(max(lo, 0), min(hi, N))
            if len(n) == 0:
                continue
            d[n] += np.interp((n / fs - bt) / w, tb, base)

    f_nl = lambda x: np.tanh(1.6 * x) + 0.18 * x ** 2
    h = np.array([0.05, 0.20, 0.35, 0.25, 0.10, 0.05]); h /= h.sum()
    dr = 1.0 + drift * np.sin(2 * np.pi * 0.02 * t)
    cfa = dr * signal.lfilter(h, 1.0, f_nl(d)) * artifact_gain
    white = rng.normal(0, 1, N)
    pink = signal.lfilter([0.049922, -0.095993, 0.050612, -0.004408],
                          [1, -2.494956, 2.017265, -0.522189], white)
    pink /= np.std(pink)
    alpha = (0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t)) * np.sin(2 * np.pi * 10 * t)
    s = 0.7 * pink + 0.6 * alpha
    y = s + cfa + rng.normal(0, 0.10, N)
    r = d + rng.normal(0, 0.03, N)
    return t, y, s, cfa, r, beat_t


def snr_db(s, est, mask):
    return 10.0 * np.log10(np.sum(s[mask] ** 2) / (np.sum((s - est)[mask] ** 2) + 1e-30))


def synthetic_block(seeds=(11, 13, 17), rsas=(0.04, 0.08, 0.20)) -> list[dict]:
    rows = []
    for stretch in (False, True):
        for rsa in rsas:
            acc = {(m, g): [] for m in MODES for g in GAMMAS}
            cvs, raws = [], []
            for seed in seeds:
                t, y, s, cfa, r, beat_t = gen_eeg_variant(FS, DUR, seed, stretch, rsa=rsa)
                est = C.phase_estimator(r, FS, True, True)
                ss = t > (t[-1] * 0.5)
                raw = snr_db(s, y, ss)
                raws.append(raw)
                T = np.diff(beat_t)
                cvs.append(float(np.std(T) / np.mean(T)))
                for m in MODES:
                    for g in GAMMAS:
                        e, _ = C.ilc_clean(y, FS, est, m, True, gamma=g)
                        acc[(m, g)].append(snr_db(s, e, ss) - raw)
            row = {
                "artifact": "phase-stretched" if stretch else "time-locked",
                "rsa": rsa, "cv_rr": float(np.mean(cvs)),
                "input_snr_db": float(np.mean(raws)),
            }
            for (m, g), v in acc.items():
                row[f"{m}_g{g}"] = float(np.mean(v))
            for m in MODES:
                row[f"{m}_gain"] = row[f"{m}_g1.0"] - row[f"{m}_g0.0"]
            rows.append(row)
            print(f"  {row['artifact']:>16s}  rsa={rsa:.2f} CV={row['cv_rr']*100:4.1f}%  "
                  + "  ".join(f"{m}: g0={row[f'{m}_g0.0']:+.2f} g1={row[f'{m}_g1.0']:+.2f} "
                              f"(D={row[f'{m}_gain']:+.2f})" for m in MODES), flush=True)
    return rows


# ---------------------------------------------------------------------------
# real: R-locked energy removed, against a circular-shift null
# ---------------------------------------------------------------------------
def rlocked_energy(z, r_samples, ra, Lw, fs, lo_s, hi_s):
    """Energy of the R-triggered average of z over [lo_s, hi_s] relative to R."""
    tmpl = dm.build_template(z, r_samples, ra, Lw)
    lo = max(0, ra + int(round(lo_s * fs)))
    hi = min(Lw, ra + int(round(hi_s * fs)))
    return float(np.sum(tmpl[lo:hi] ** 2)), hi - lo


def build_warped_template(y, r_samples, W, ra, Lw):
    """Non-causal average on the warped grid; mirrors ilc_clean(mode='noncausal')."""
    N = len(y)
    acc = np.zeros(Lw); cnt = np.zeros(Lw)
    for fid, wk in zip(r_samples, W):
        for o in range(Lw):
            k = int(fid) + int(round((o - ra) * wk))
            if 0 <= k < N:
                acc[o] += y[k]; cnt[o] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


def apply_warped_template(y, Rs, W, template, ra, Lw):
    """Subtract a template anchored at the nearest R; mirrors ilc_clean's indexing."""
    N = len(y)
    out = y.copy()
    Rs = np.asarray(Rs)
    for n in range(N):
        j = np.searchsorted(Rs, n); idx = -1; bd = 1e18
        for jj in (j - 1, j):
            if 0 <= jj < len(Rs):
                o = C._bin(n, Rs[jj], W[jj], ra)
                if 0 <= o < Lw and abs(n - Rs[jj]) < bd:
                    bd = abs(n - Rs[jj]); idx = o
        if idx >= 0:
            out[n] = y[n] - template[idx]
    return out


def _nearest_beat(N, Rs):
    """Index of the nearest R-peak for every sample."""
    n = np.arange(N)
    j = np.searchsorted(Rs, n)
    lo = np.clip(j - 1, 0, len(Rs) - 1)
    hi = np.clip(j, 0, len(Rs) - 1)
    return np.where(np.abs(n - Rs[lo]) <= np.abs(n - Rs[hi]), lo, hi)


def noncausal_heldout(y, r_samples, W, ra, Lw):
    """Split-half non-causal cancellation.

    Each sample is cleaned by the template built from the half of the beats that
    does NOT contain its own nearest beat.  Anchoring still uses every beat, so
    the indexing is identical to ilc_clean; only the template contents change.
    Without this, subtracting a template from the beats that built it drives the
    R-triggered residual to zero identically at gamma=0.
    """
    N = len(y)
    idx = np.arange(len(r_samples))
    nb = _nearest_beat(N, np.asarray(r_samples))
    out = y.copy()
    for train_m, test_m in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        tmpl = build_warped_template(y, r_samples[train_m], W[train_m], ra, Lw)
        cleaned = apply_warped_template(y, r_samples, W, tmpl, ra, Lw)
        m = test_m[nb]
        out[m] = cleaned[m]
    return out


def beat_offset_mask(N, Rs, fs, lo_s, hi_s):
    """Samples whose offset from the NEAREST R-peak falls in [lo_s, hi_s)."""
    nb = _nearest_beat(N, np.asarray(Rs))
    off = (np.arange(N) - np.asarray(Rs)[nb]) / fs
    return (off >= lo_s) & (off < hi_s)


def real_block(dataset_root, subject, session, run, eeg_index, ecg_index,
               duration_sec, n_scramble=200, seed=0) -> dict:
    y, r, fs, label = dm.load_eeg_ecg(
        dataset_root, subject, session, run, eeg_index, ecg_index, False,
        highpass_hz=0.5, notch_hz=50.0, start_sec=0.0, duration_sec=duration_sec,
    )
    est = C.phase_estimator(r, fs, True, True)
    r_samples = np.asarray([f[0] for f in est["fids"] if f[2]], dtype=int)
    ra, Lw = C._template_dims(fs)
    N = len(y)

    # the shipped template runs to +0.50 s, so the late band is clipped there
    bands = {"lobe": (-0.05, 0.15), "late": (0.20, 0.50)}
    masks = {b: beat_offset_mask(N, r_samples, fs, *w) for b, w in bands.items()}

    out = {"label": label, "fs": fs, "n_beats": int(len(r_samples)),
           "sd_uV": float(np.std(y)) * 1e6}
    T = np.diff(r_samples) / fs
    out["rr_mean_s"] = float(np.mean(T))
    med = float(np.median(T))
    out["cv_rr_robust"] = 1.4826 * float(np.median(np.abs(T - med))) / med

    # descriptive: how big is the R-locked deflection, against a circular-shift null
    rng = np.random.default_rng(seed)
    for b, (lo_s, hi_s) in bands.items():
        e = np.empty(n_scramble)
        for i in range(n_scramble):
            scr = dm.scramble_samples(r_samples, N, rng, int(round(2.0 * fs)))
            e[i], _ = rlocked_energy(y, scr, ra, Lw, fs, lo_s, hi_s)
        floor = float(np.mean(e))
        e0, L = rlocked_energy(y, r_samples, ra, Lw, fs, lo_s, hi_s)
        out[f"raw_{b}_db_over_floor"] = 10 * np.log10(e0 / (floor + 1e-30))
        out[f"raw_{b}_rms_uV"] = float(np.sqrt(max(e0 - floor, 0.0) / L)) * 1e6

    # comparison: held-out variance reduction inside the beat window.  Coordinate
    # free, so it does not privilege either gamma, and computed on the signal
    # ilc_clean actually returns.
    for g in GAMMAS:
        W_learn = C.beat_warps(r_samples, fs, g)
        variants = {
            "noncausal": noncausal_heldout(y, r_samples, W_learn, ra, Lw),
            "buffered": C.ilc_clean(y, fs, est, "buffered", True, gamma=g)[0],
        }
        for m, clean in variants.items():
            for b, msk in masks.items():
                num = float(np.sum(clean[msk] ** 2))
                den = float(np.sum(y[msk] ** 2))
                out[f"{m}_g{g}_{b}_removed_db"] = 10 * np.log10(den / (num + 1e-30))
    for m in MODES:
        print(f"  {m}: " + "  ".join(
            f"g={g}: lobe {out[f'{m}_g{g}_lobe_removed_db']:+.2f}dB "
            f"late {out[f'{m}_g{g}_late_removed_db']:+.2f}dB" for g in GAMMAS), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--subject", default="sub-070")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--run", default="20")
    ap.add_argument("--eeg-index", type=int, default=1)
    ap.add_argument("--ecg-index", type=int, default=0)
    ap.add_argument("--duration-sec", type=float, default=300.0)
    ap.add_argument("--extra", default="sub-001:01:1,sub-070:01:1,sub-118:02:1",
                    help="comma list subject:run:eeg_index for a second opinion")
    ap.add_argument("--out", default="../outputs/phase_swap.json")
    a = ap.parse_args()

    print("SYNTHETIC (ground truth, SNR improvement in dB):", flush=True)
    syn = synthetic_block()

    root = dm._resolve_dataset_root(a.dataset_root)
    print(f"\nREAL, Figure-1 record {a.subject}/run-{a.run}/eeg{a.eeg_index}:", flush=True)
    real = [real_block(root, a.subject, a.session, a.run, a.eeg_index, a.ecg_index,
                       a.duration_sec)]

    for spec in [x for x in a.extra.split(",") if x]:
        sub, run, ei = spec.split(":")
        if (sub, run, int(ei)) == (a.subject, a.run, a.eeg_index):
            continue
        print(f"\nREAL, {sub}/run-{run}/eeg{ei}:", flush=True)
        try:
            real.append(real_block(root, sub, a.session, run, int(ei), a.ecg_index,
                                   a.duration_sec))
        except Exception as e:
            print(f"  [skip] {type(e).__name__}: {e}")

    Path(a.out).write_text(json.dumps({"synthetic": syn, "real": real},
                                      indent=1, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
