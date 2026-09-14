#!/usr/bin/env python3
"""
pll_design.py -- design the strict-causal R-peak predictor as a PI/PID loop.

State model (the one a PLL actually implements): position is the beat INSTANT,
velocity is the period.

    predict   that_k^- = that_{k-1} + That_{k-1},   That_k^- = That_{k-1}
    residual  r_k      = (t_k + v_k) - that_k^-
    update    that_k   = that_k^- + a r_k                 (proportional)
              That_k   = That_{k-1} + b r_k               (integral)
              [+ c (r_k - r_{k-1}) on That for the derivative term]

a is the proportional (phase) gain, b the integral (period) gain: an alpha-beta
filter and a PI PLL are the same object.  The quantity of interest is the
one-step prediction error e_k = that_k^- - t_k, which is exactly the
strict-causal anchor error.

Everything is evaluated by running the recursion on the MEASURED beat instants,
so no linearisation is involved and the numbers are directly comparable with the
measured error of the shipped loop.

  python pll_design.py --out ../outputs/pll_design.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BURN = 30


def run_pi(T_ms, a, b, c=0.0, sigma_v_ms=0.0, seed=0):
    """One-step prediction error of the PI(D) loop on a measured RR sequence."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(T_ms, float))])
    v = rng.normal(0.0, sigma_v_ms, len(t)) if sigma_v_ms > 0 else np.zeros(len(t))
    that = t[0]
    That = float(np.mean(T_ms))
    r_prev = 0.0
    e = np.empty(len(t) - 1)
    for k in range(1, len(t)):
        pred = that + That                       # t_k^- , the anchor we publish
        e[k - 1] = pred - t[k]
        r = (t[k] + v[k]) - pred
        that = pred + a * r
        That = That + b * r + c * (r - r_prev)
        r_prev = r
    return float(np.std(e[BURN:])), e


def run_as_built(T_ms, kp=0.30, alpha=0.30, sigma_v_ms=0.0, seed=0):
    """The shipped structure: proportional phase pull + separate EWMA on measured RR."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(T_ms, float))])
    v = rng.normal(0.0, sigma_v_ms, len(t)) if sigma_v_ms > 0 else np.zeros(len(t))
    that = t[0]
    That = float(np.mean(T_ms))
    e = np.empty(len(t) - 1)
    for k in range(1, len(t)):
        pred = that + That
        e[k - 1] = pred - t[k]
        meas = t[k] + v[k]
        that = pred + kp * (meas - pred)          # only kp of the error removed
        if k >= 2:
            That = That + alpha * ((meas - (t[k - 1] + v[k - 1])) - That)
        r_prev = 0.0
    return float(np.std(e[BURN:])), e


def stable(a, b):
    """alpha-beta stability: 0 < a < 2, 0 < b, b < 4 - 2a."""
    return 0.0 < a < 2.0 and 0.0 < b < 4.0 - 2.0 * a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bode", default="../outputs/rpeak_bode.json")
    ap.add_argument("--out", default="../outputs/pll_design.json")
    a_ = ap.parse_args()

    d = [r for r in json.loads(Path(a_.bode).read_text())
         if r["match_rate"] >= 0.95 and r["rr_mean_ms"] >= 500]
    Ts = [np.asarray(r["_Tref_ms"]) for r in d]
    sv = float(np.median([r["fid_mad_ms"] for r in d]))
    print(f"{len(Ts)} runs, fiducial noise sigma_v = {sv:.1f} ms")

    def score(fn, **kw):
        return float(np.median([fn(T, sigma_v_ms=sv, **kw)[0] for T in Ts]))

    built = score(run_as_built, kp=0.30, alpha=0.30)
    print(f"\nas built (Kp=0.3, alpha=0.3)            {built:6.1f} ms")
    print(f"P only, deadbeat (a=1, b=0+)            "
          f"{score(run_pi, a=1.0, b=1e-6):6.1f} ms")

    # --- PI grid ---------------------------------------------------------
    A = np.round(np.arange(0.2, 1.81, 0.1), 2)
    B = np.round(np.arange(0.0, 1.61, 0.05), 3)
    grid = np.full((len(A), len(B)), np.nan)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if b > 0 and stable(a, b):
                grid[i, j] = score(run_pi, a=a, b=b)
    k = np.unravel_index(np.nanargmin(grid), grid.shape)
    a_pi, b_pi, sd_pi = float(A[k[0]]), float(B[k[1]]), float(grid[k])
    print(f"best PI  a={a_pi:.2f} b={b_pi:.2f}          {sd_pi:6.1f} ms")

    # --- add a derivative term ------------------------------------------
    best_d = (sd_pi, a_pi, b_pi, 0.0)
    for c in np.round(np.arange(-0.4, 0.81, 0.05), 3):
        for a in np.round(np.arange(max(0.2, a_pi - 0.4), a_pi + 0.45, 0.1), 2):
            for b in np.round(np.arange(max(0.0, b_pi - 0.4), b_pi + 0.45, 0.05), 3):
                if b <= 0 or not stable(a, b):
                    continue
                s = score(run_pi, a=a, b=b, c=c)
                if s < best_d[0]:
                    best_d = (s, float(a), float(b), float(c))
    print(f"best PID a={best_d[1]:.2f} b={best_d[2]:.2f} c={best_d[3]:+.2f}  "
          f"{best_d[0]:6.1f} ms")

    out = {
        "n_runs": len(Ts), "sigma_v_ms": sv,
        "sd_as_built_ms": built,
        "sd_deadbeat_ms": score(run_pi, a=1.0, b=1e-6),
        "pi": {"a": a_pi, "b": b_pi, "sd_ms": sd_pi},
        "pid": {"a": best_d[1], "b": best_d[2], "c": best_d[3], "sd_ms": best_d[0]},
        "grid_a": A.tolist(), "grid_b": B.tolist(), "grid_sd": grid.tolist(),
    }
    # per-run check with the chosen PI gains
    out["per_run"] = [
        {"run": r["run"],
         "measured_pll_sd_ms": r["pll_sd_ms"],
         "sim_as_built_ms": run_as_built(T, 0.30, 0.30, sv)[0],
         "sim_pi_ms": run_pi(T, a_pi, b_pi, sigma_v_ms=sv)[0]}
        for r, T in zip(d, Ts)]
    Path(a_.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {a_.out}")


if __name__ == "__main__":
    main()
