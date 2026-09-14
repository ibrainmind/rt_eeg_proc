#!/usr/bin/env python3
"""
rr_process_id.py -- identify the RR process, then design the PLL from it.

The gain sweep in the previous section found that both loop gains want to be as
large as possible.  That is a symptom, not an explanation.  This script does the
identification that explains it, and turns it into a design procedure.

Step 1  Is RR stationary about a mean, or integrated?  Compare the sample ACF of
        T_k with that of its first difference.
Step 2  If the difference is MA(1) with negative lag-1 correlation, the series is
        a local-level (random walk plus noise) process,
            T_k = mu_k + e_k,   mu_k = mu_{k-1} + w_k,
        for which EXPONENTIAL SMOOTHING IS THE OPTIMAL PREDICTOR and the optimal
        gain follows in closed form from rho_1:  rho_1 = -theta/(1+theta^2) and
        alpha* = 1 - theta.  No sweep required.
Step 3  The phase loop gain follows from the measurement noise.  With the beat
        instant measured to sigma_v and the period wandering by sigma_w per beat,
        the steady-state alpha-beta (Kalata) gains are set by the tracking index
        Lambda = sigma_w / sigma_v.
Step 4  Check the prediction against the measured error.

  python rr_process_id.py --out ../outputs/rr_process_id.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import rpeak_bode as RB

NLAG = 8


def acf(x, nlag=NLAG):
    x = np.asarray(x, float) - np.mean(x)
    d = float(np.dot(x, x))
    return np.array([float(np.dot(x[:len(x) - k], x[k:])) / d for k in range(nlag + 1)])


def local_level_from_rho1(rho1: float):
    """Invert rho_1 = -theta/(1+theta^2) for the invertible MA(1) root.

    Returns (theta, alpha_star, q) where q = sigma_w^2/sigma_e^2 is the
    signal-to-noise ratio of the local-level model.  rho1 >= 0 means there is no
    white component to smooth away: the optimal filter is deadbeat, alpha = 1.
    """
    if rho1 >= 0:
        return 0.0, 1.0, float("inf")
    rho1 = max(rho1, -0.5)                       # invertibility bound
    disc = 1.0 - 4.0 * rho1 ** 2
    theta = (-1.0 + np.sqrt(disc)) / (2.0 * rho1)
    if abs(theta) > 1:
        theta = 1.0 / theta
    alpha = 1.0 - theta
    q = alpha ** 2 / max(1.0 - alpha, 1e-12)     # sigma_w^2 / sigma_e^2
    return float(theta), float(alpha), float(q)


def kalata(lam: float):
    """Steady-state alpha-beta gains for tracking index Lambda (unit sample time)."""
    if not np.isfinite(lam):
        return 1.0, 2.0
    r = (4.0 + lam - np.sqrt(8.0 * lam + lam ** 2)) / 4.0
    a = 1.0 - r ** 2
    b = 2.0 * (2.0 - a) - 4.0 * np.sqrt(max(1.0 - a, 0.0))
    return float(a), float(b)


def analyse(T_ms, sigma_v_ms) -> dict:
    T = np.asarray(T_ms, float)
    dT = np.diff(T)
    rT, rD = acf(T), acf(dT)
    rho1 = float(rD[1])
    var_d = float(np.var(dT))

    theta, alpha_star, q = local_level_from_rho1(rho1)
    # variance split implied by the MA(1) fit: var(dT) = sw^2 + 2 se^2
    se2 = max(-rho1, 0.0) * var_d
    sw2 = max(var_d - 2.0 * se2, 0.0)
    sw, se = float(np.sqrt(sw2)), float(np.sqrt(se2))
    lam = sw / max(sigma_v_ms, 1e-9)
    a_kal, b_kal = kalata(lam)

    return {
        "n": int(len(T)),
        "rr_sd_ms": float(np.std(T)),
        "dT_sd_ms": float(np.sqrt(var_d)),
        "acf_T": rT.tolist(), "acf_dT": rD.tolist(),
        "rho1_dT": rho1,
        "theta": theta, "alpha_star": alpha_star, "q_snr": q,
        "sigma_w_ms": sw, "sigma_e_ms": se, "sigma_v_ms": float(sigma_v_ms),
        "tracking_index": float(lam),
        "kalata_alpha": a_kal, "kalata_beta": b_kal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bode", default="../outputs/rpeak_bode.json")
    ap.add_argument("--out", default="../outputs/rr_process_id.json")
    a = ap.parse_args()

    d = [r for r in json.loads(Path(a.bode).read_text())
         if r["match_rate"] >= 0.95 and r["rr_mean_ms"] >= 500]
    res = []
    for r in d:
        o = analyse(r["_Tref_ms"], r["fid_mad_ms"])
        o["run"] = r["run"]
        o["measured_pll_sd_ms"] = r["pll_sd_ms"]
        # what the identified design would deliver, on this run's own RR
        T = np.asarray(r["_Tref_ms"])
        o["sd_as_built_ms"] = RB.predicted_sd(T, kp=0.30, alpha=0.30)[0]
        o["sd_alpha_star_ms"] = RB.predicted_sd(T, kp=1.0, alpha=o["alpha_star"])[0]
        o["sd_kalata_ms"] = RB.predicted_sd(
            T, kp=1.0, ab=(o["kalata_alpha"], o["kalata_beta"]))[0]
        res.append(o)
        print(f"  run-{o['run']}: rho1(dT)={o['rho1_dT']:+.3f}  "
              f"sw={o['sigma_w_ms']:5.1f} se={o['sigma_e_ms']:5.1f} "
              f"sv={o['sigma_v_ms']:4.1f} ms  Lambda={o['tracking_index']:6.1f}  "
              f"alpha*={o['alpha_star']:.2f}  "
              f"SD: built {o['sd_as_built_ms']:.0f} -> a* {o['sd_alpha_star_ms']:.0f} "
              f"-> kalata {o['sd_kalata_ms']:.0f} ms", flush=True)

    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {a.out} ({len(res)} runs)")
    m = lambda k: float(np.median([x[k] for x in res]))
    print(f"\nmedians: rho1 {m('rho1_dT'):+.3f}  sigma_w {m('sigma_w_ms'):.1f} ms  "
          f"sigma_e {m('sigma_e_ms'):.1f} ms  sigma_v {m('sigma_v_ms'):.1f} ms")
    print(f"         Lambda {m('tracking_index'):.1f}  alpha* {m('alpha_star'):.2f}  "
          f"kalata a {m('kalata_alpha'):.2f} b {m('kalata_beta'):.2f}")
    print(f"         SD as-built {m('sd_as_built_ms'):.0f} -> alpha* "
          f"{m('sd_alpha_star_ms'):.0f} -> kalata {m('sd_kalata_ms'):.0f} ms")


if __name__ == "__main__":
    main()
