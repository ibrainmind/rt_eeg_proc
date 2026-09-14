#!/usr/bin/env python3
"""
pll_design_figure.py -- identification, design surface, and the resulting gain.

  python pll_design_figure.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pll_design as PD
import rr_process_id as RI

plt.rcParams.update({"font.family": "serif", "font.size": 9})
NLAG = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bode", default="../outputs/rpeak_bode.json")
    ap.add_argument("--design", default="../outputs/pll_design.json")
    ap.add_argument("--fig", default="../paper/fig_real/fig_pll_design.pdf")
    ap.add_argument("--macros", default="../paper/pll_design_numbers.tex")
    a_ = ap.parse_args()

    d = [r for r in json.loads(Path(a_.bode).read_text())
         if r["match_rate"] >= 0.95 and r["rr_mean_ms"] >= 500]
    D = json.loads(Path(a_.design).read_text())
    Ts = [np.asarray(r["_Tref_ms"]) for r in d]
    sv = D["sigma_v_ms"]
    sw = float(np.median([np.std(np.diff(T)) for T in Ts]))
    lam = sw / sv
    ka, kb = RI.kalata(lam)

    fig, ax = plt.subplots(1, 3, figsize=(9.8, 2.9))

    # ---- (a) identification -------------------------------------------
    rT = np.median([RI.acf(T, NLAG) for T in Ts], axis=0)
    rD = np.median([RI.acf(np.diff(T), NLAG) for T in Ts], axis=0)
    lags = np.arange(NLAG + 1)
    ax[0].axhline(0, color="0.5", lw=0.8)
    ax[0].plot(lags, rT, "-o", ms=3.5, color="tab:blue", label=r"$T_k$")
    ax[0].plot(lags, rD, "-s", ms=3.5, color="tab:red", label=r"$\Delta T_k$")
    ci = 1.96 / np.sqrt(np.median([len(T) for T in Ts]))
    ax[0].axhspan(-ci, ci, color="0.85", zorder=0)
    ax[0].set_xlabel("lag (beats)")
    ax[0].set_ylabel("autocorrelation")
    ax[0].set_title("(a) identify the RR process", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(alpha=0.3)
    ax[0].text(0.97, 0.06, rf"$\rho_1(\Delta T)={rD[1]:+.2f}>0$" "\n"
               "smoother than a random walk",
               transform=ax[0].transAxes, fontsize=6.5, ha="right", color="tab:red")

    # ---- (b) design surface -------------------------------------------
    A = np.asarray(D["grid_a"]); B = np.asarray(D["grid_b"])
    G = np.asarray(D["grid_sd"], dtype=float)
    G = np.where(np.isfinite(G), G, np.nan)
    lv = [17, 18, 20, 23, 27, 32, 40, 52, 70]
    cs = ax[1].contourf(B, A, np.clip(G, 0, 70), levels=lv, cmap="viridis_r")
    ax[1].contour(B, A, np.clip(G, 0, 70), levels=lv, colors="w", linewidths=0.4)
    bb = np.linspace(0, B.max(), 50)
    ax[1].plot(bb, (4 - bb) / 2, "r--", lw=1.1, label=r"stability $b=4-2a$")
    ax[1].scatter([D["pi"]["b"]], [D["pi"]["a"]], marker="*", s=110, c="w",
                  edgecolors="k", zorder=5, label="grid optimum")
    ax[1].scatter([kb], [ka], marker="o", s=42, c="tab:orange", edgecolors="k",
                  zorder=5, label="Kalata, from $\\Lambda$")
    ax[1].scatter([0.30], [0.30], marker="X", s=52, c="tab:red", edgecolors="k",
                  zorder=5, label="as built")
    ax[1].set_xlim(0, B.max()); ax[1].set_ylim(A.min(), A.max())
    ax[1].set_xlabel(r"integral gain $b$")
    ax[1].set_ylabel(r"proportional gain $a$")
    ax[1].set_title("(b) design surface (ms)", fontsize=9)
    ax[1].legend(fontsize=6, frameon=False, loc="upper right")
    fig.colorbar(cs, ax=ax[1], pad=0.02, aspect=26)

    # ---- (c) integral action is the whole story ------------------------
    bs = np.round(np.arange(0.0, 1.85, 0.05), 3)
    for a, col in ((0.30, "tab:red"), (1.00, "tab:blue")):
        y = []
        for b in bs:
            if b <= 0 or not PD.stable(a, b):
                y.append(np.nan); continue
            v = float(np.median([PD.run_pi(T, a, b, sigma_v_ms=sv)[0] for T in Ts]))
            y.append(v if np.isfinite(v) and v < 200 else np.nan)
        ax[2].plot(bs, y, color=col, lw=1.4, label=rf"$a={a:g}$")
    ax[2].axhline(D["sd_as_built_ms"], color="tab:red", ls=":", lw=1.1,
                  label="as built (shipped structure)")
    ax[2].axhline(D["sd_deadbeat_ms"], color="0.45", ls="--", lw=1.0,
                  label=r"P only ($b\to 0$)")
    ax[2].scatter([D["pi"]["b"]], [D["pi"]["sd_ms"]], marker="*", s=110, c="w",
                  edgecolors="k", zorder=5)
    ax[2].set_ylim(0, 60)
    ax[2].set_xlabel(r"integral gain $b$")
    ax[2].set_ylabel("predicted R-peak error SD (ms)")
    ax[2].set_title("(c) it is the integral term that matters", fontsize=9)
    ax[2].legend(fontsize=6.5, frameon=False)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    Path(a_.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a_.fig, bbox_inches="tight")
    print(f"wrote {a_.fig}")

    sc = lambda a, b: float(np.median([PD.run_pi(T, a, b, sigma_v_ms=sv)[0] for T in Ts]))
    m = {
        "pdNruns": f"{len(Ts)}",
        "pdRho": f"{rD[1]:+.2f}",
        "pdSigmaW": f"{sw:.1f}",
        "pdSigmaV": f"{sv:.1f}",
        "pdLambda": f"{lam:.0f}",
        "pdKalA": f"{ka:.2f}",
        "pdKalB": f"{kb:.2f}",
        "pdKalSd": f"{sc(ka, kb):.1f}",
        "pdOptA": f"{D['pi']['a']:.2f}",
        "pdOptB": f"{D['pi']['b']:.2f}",
        "pdOptSd": f"{D['pi']['sd_ms']:.1f}",
        "pdBuilt": f"{D['sd_as_built_ms']:.0f}",
        "pdPonly": f"{D['sd_deadbeat_ms']:.0f}",
        "pdPidC": f"{D['pid']['c']:+.2f}",
        "pdPidSd": f"{D['pid']['sd_ms']:.1f}",
    }
    Path(a_.macros).write_text("\n".join(
        rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()) + "\n")
    print(f"wrote {a_.macros}")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
