#!/usr/bin/env python3
"""
phase_warp_synth.py -- when does the normalized-phase coordinate actually pay?

The real recordings answer "does it help here"; this answers "where would it".
We synthesize an ear-EEG-like trace whose cardiac artifact is deliberately mixed:

  * a QRS component locked to a fixed time offset from R (systole is closer to
    time-locked than proportional to RR), and
  * a T-wave component locked to a fixed NORMALIZED PHASE (diastole absorbs the
    beat-to-beat variation),

which is the physiologically motivated case in which neither gamma=0 nor
gamma=1 is right by construction.  Sweeping the RR standard deviation and
scoring with the same split-half held-out variance reduction used on the real
data gives the advantage of normalized phase over fixed lag as a function of
heart-rate variability, and the RR variability at which it becomes material.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import phase_hep_analysis as ph

FS = 256.0
RR_MEAN = 0.85
DUR = 600.0


def _biphasic(x: np.ndarray) -> np.ndarray:
    """Unit-scale biphasic deflection on a normalized support x in [0,1]."""
    return np.exp(-((x - 0.30) / 0.10) ** 2) - 0.55 * np.exp(-((x - 0.62) / 0.18) ** 2)


def synth(sigma_rr: float, seed: int = 0, snr_lin: float = 0.35) -> tuple:
    """Return (y, r_samples, T) for one synthetic record at RR SD sigma_rr (s)."""
    rng = np.random.default_rng(seed)
    # --- beat train -------------------------------------------------------
    T = []
    tot = 0.0
    while tot < DUR:
        Tk = float(np.clip(rng.normal(RR_MEAN, sigma_rr), 0.40, 1.60))
        T.append(Tk)
        tot += Tk
    T = np.asarray(T[:-1])
    t_k = np.concatenate([[1.0], 1.0 + np.cumsum(T)[:-1]])
    n = int(DUR * FS)
    tn = np.arange(n) / FS

    # --- neural background: 1/f-ish, not phase locked ---------------------
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    W[1:] /= np.sqrt(f[1:])
    s = np.fft.irfft(W, n)
    s /= np.std(s)

    # --- cardiac artifact: QRS time-locked + T-wave phase-locked ----------
    d = np.zeros(n)
    for tk, Tk in zip(t_k, T):
        # QRS: fixed 0-120 ms window after R, independent of Tk
        m = (tn >= tk - 0.02) & (tn < tk + 0.12)
        if np.any(m):
            d[m] += 1.0 * _biphasic((tn[m] - (tk - 0.02)) / 0.14)
        # T-wave: fixed phase band 0.25-0.65 of the cycle, stretched with Tk
        m = (tn >= tk + 0.25 * Tk) & (tn < tk + 0.65 * Tk)
        if np.any(m):
            d[m] += 0.45 * _biphasic((tn[m] - (tk + 0.25 * Tk)) / (0.40 * Tk))
    d /= np.std(d)

    y = snr_lin * s + d * 1.0 + 0.15 * rng.standard_normal(n)
    r_samples = np.round(t_k * FS).astype(int)
    keep = (r_samples > 0) & (r_samples < n)
    return y, r_samples[keep], T[keep]


def run_sweep(sigmas, seeds=(0, 1, 2)) -> list[dict]:
    out = []
    for sg in sigmas:
        per = {g: [] for g in ph.GAMMAS}
        cvs = []
        for sd in seeds:
            y, r_samples, T = synth(sg, seed=sd)
            T_bar = float(np.mean(T))
            w_min = float(np.min(T / T_bar))
            c_grid = np.arange(-ph.PRE_S / w_min - 0.02,
                               ph.POST_S / w_min + 0.02, 1.0 / FS)
            med = float(np.median(T))
            cvs.append(1.4826 * float(np.median(np.abs(T - med))) / med)
            for g in ph.GAMMAS:
                per[g].append(
                    ph.eval_gamma(y, r_samples, T, FS, g, T_bar, c_grid)["vr_db_all"]
                )
        row = {"sigma_rr_s": sg, "cv_robust": float(np.mean(cvs))}
        for g in ph.GAMMAS:
            row[f"vr_g{g}"] = float(np.mean(per[g]))
        row["delta_db"] = row[f"vr_g{1.0}"] - row[f"vr_g{0.0}"]
        row["best_gamma"] = max(ph.GAMMAS, key=lambda g: row[f"vr_g{g}"])
        out.append(row)
        print(f"sigma={sg*1e3:5.0f} ms  cv={row['cv_robust']*100:5.1f}%  "
              f"vr(0)={row['vr_g0.0']:+.3f}  vr(1)={row['vr_g1.0']:+.3f}  "
              f"D={row['delta_db']:+.3f} dB  best g={row['best_gamma']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../outputs/phase_warp_synth.json")
    ap.add_argument("--sigmas", default="0,0.02,0.04,0.06,0.08,0.10,0.15")
    a = ap.parse_args()
    sig = [float(x) for x in a.sigmas.split(",")]
    rows = run_sweep(sig)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
