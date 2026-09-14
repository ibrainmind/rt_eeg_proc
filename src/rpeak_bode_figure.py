#!/usr/bin/env python3
"""
rpeak_bode_figure.py -- Bode of the beat-domain error dynamics + model validation.

  python rpeak_bode_figure.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import rpeak_bode as RB

plt.rcParams.update({"font.family": "serif", "font.size": 9})

MATCH_MIN = 0.95
LABEL = {
    "as_built":    r"as built  ($K_p{=}0.3,\ \alpha{=}0.3$)",
    "reanchored":  r"re-anchored  ($K_p{=}1,\ \alpha{=}0.3$)",
    "previous_rr": r"previous RR  ($K_p{=}1,\ \alpha{=}1$)",
    "alphabeta":   r"$\alpha$-$\beta$  ($K_p{=}1$)",
}
COLOR = {"as_built": "tab:red", "reanchored": "tab:blue",
         "previous_rr": "tab:green", "alphabeta": "tab:purple"}


def ok_runs(d):
    return [r for r in d if r["match_rate"] >= MATCH_MIN and r["rr_mean_ms"] >= 500]


def med(rows, k):
    v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../outputs/rpeak_bode.json")
    ap.add_argument("--fig", default="../paper/fig_real/fig_rpeak_bode.pdf")
    ap.add_argument("--macros", default="../paper/rpeak_bode_numbers.tex")
    ap.add_argument("--highlight-run", default="20")
    a = ap.parse_args()

    d = json.loads(Path(a.json).read_text())
    ok = ok_runs(d)
    hl = next((r for r in d if r.get("run") == a.highlight_run), None)
    print(f"{len(ok)} of {len(d)} runs kept (match >= {MATCH_MIN:.0%})")

    fig, ax = plt.subplots(1, 3, figsize=(9.8, 2.9))

    # ---- (a) Bode magnitude of the error sensitivity -------------------
    w = np.logspace(np.log10(0.004), np.log10(0.5), 600)      # cycles per beat
    for name, kw in RB.DESIGNS.items():
        b, aa = RB.error_filter(**kw)
        _, h = signal.freqz(b, aa, worN=2 * np.pi * w)
        ax[0].semilogx(w, 20 * np.log10(np.abs(h) + 1e-12),
                       color=COLOR[name], lw=1.4, label=LABEL[name])
    # measured RR disturbance spectrum, pooled and normalised for shape only
    f = np.asarray(ok[0]["rr_spec_f"])
    P = np.mean([np.interp(f, r["rr_spec_f"], r["rr_spec_P"]) for r in ok], axis=0)
    P = P / P.max()
    ax2 = ax[0].twinx()
    ax2.fill_between(f[1:], 0, P[1:], color="0.75", alpha=0.45, zorder=0)
    ax2.set_ylim(0, 3.2); ax2.set_yticks([])
    ax2.set_ylabel("RR disturbance (norm.)", fontsize=7, color="0.45")
    ax[0].axhline(0, color="0.6", lw=0.7)
    ax[0].set_xlabel("frequency (cycles per beat)")
    ax[0].set_ylabel(r"$|S|$: RR variation $\to$ timing error (dB)")
    ax[0].set_title("(a) beat-domain error sensitivity", fontsize=9)
    ax[0].set_ylim(-45, 22)
    ax[0].legend(fontsize=6, frameon=False, loc="upper left")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].set_zorder(ax2.get_zorder() + 1); ax[0].patch.set_visible(False)

    # ---- (b) model vs measurement --------------------------------------
    mx = [r["model_as_built_sd_ms"] for r in ok]
    my = [r["pll_sd_ms"] for r in ok]
    rx = [r["model_reanchored_sd_ms"] for r in ok if "ol_sd_ms" in r]
    ry = [r["ol_sd_ms"] for r in ok if "ol_sd_ms" in r]
    ax[1].scatter(mx, my, s=16, color="tab:red", label="as built")
    ax[1].scatter(rx, ry, s=16, color="tab:blue", label="re-anchored")
    lim = [0, max(mx + my + rx + ry) * 1.1]
    ax[1].plot(lim, lim, "k--", lw=0.9, label="model = measured")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("model-predicted error SD (ms)")
    ax[1].set_ylabel("measured error SD (ms)")
    ax[1].set_title("(b) model validation, 1 point per run", fontsize=9)
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    ax[1].grid(alpha=0.3)

    # ---- (c) the design curve: error against the two loop gains ---------
    al = np.asarray(RB.ALPHA_GRID)
    for key, kp, col in (("kp03", 0.30, "tab:red"), ("kp10", 1.00, "tab:blue")):
        Y = np.array([r[f"sweep_{key}"] for r in ok])
        ax[2].plot(al, np.median(Y, axis=0), "-o", color=col, ms=3, lw=1.3,
                   label=rf"$K_p={kp:g}$")
    ax[2].scatter([0.3], [med(ok, "model_as_built_sd_ms")], s=60, marker="*",
                  color="k", zorder=4)
    ax[2].annotate("as built", (0.3, med(ok, "model_as_built_sd_ms")),
                   textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax[2].scatter([1.0], [med(ok, "model_previous_rr_sd_ms")], s=60, marker="*",
                  color="k", zorder=4)
    ax[2].annotate("recommended", (1.0, med(ok, "model_previous_rr_sd_ms")),
                   textcoords="offset points", xytext=(-12, 8), fontsize=7, ha="right")
    ax[2].set_xlabel(r"period-estimator gain $\alpha$")
    ax[2].set_ylabel("predicted R-peak error SD (ms)")
    ax[2].set_title("(c) both gains want to be larger", fontsize=9)
    ax[2].set_ylim(0, None)
    ax[2].legend(fontsize=7, frameon=False)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, bbox_inches="tight")
    print(f"wrote {a.fig}")

    m = {
        "rbNruns": f"{len(ok)}",
        "rbNrunsAll": f"{len(d)}",
        "rbFidJitter": f"{med(ok,'fid_mad_ms'):.1f}",
        "rbMatch": f"{100*med(ok,'match_rate'):.0f}",
        "rbPllSd": f"{med(ok,'pll_sd_ms'):.0f}",
        "rbPllPninety": f"{med(ok,'pll_abs_p90_ms'):.0f}",
        "rbOlSd": f"{med(ok,'ol_sd_ms'):.0f}",
        "rbModelAsBuilt": f"{med(ok,'model_as_built_sd_ms'):.0f}",
        "rbModelReanch": f"{med(ok,'model_reanchored_sd_ms'):.0f}",
        "rbModelLastRR": f"{med(ok,'model_previous_rr_sd_ms'):.0f}",
        "rbModelAB": f"{med(ok,'model_alphabeta_sd_ms'):.0f}",
        "rbModelRes": f"{med(ok,'model_resonator_sd_ms'):.0f}",
        "rbRRpeakHz": f"{med(ok,'rr_peak_hz'):.2f}",
        "rbRRpeakCpb": f"{med(ok,'rr_peak_cycles_per_beat'):.3f}",
        "rbRRsd": f"{med(ok,'rr_sd_ms'):.0f}",
    }
    err = [abs(r["pll_sd_ms"] - r["model_as_built_sd_ms"]) / r["pll_sd_ms"] for r in ok]
    m["rbModelErrPct"] = f"{100*np.median(err):.0f}"
    if hl is not None:
        m.update({"rbHlRun": hl["run"], "rbHlPll": f"{hl['pll_sd_ms']:.0f}",
                  "rbHlModel": f"{hl['model_as_built_sd_ms']:.0f}",
                  "rbHlOl": f"{hl['ol_sd_ms']:.0f}",
                  "rbHlAB": f"{hl['model_alphabeta_sd_ms']:.0f}",
                  "rbHlPrev": f"{hl['model_previous_rr_sd_ms']:.0f}",
                  "rbHlFidJit": f"{hl['fid_mad_ms']:.1f}"})
    Path(a.macros).write_text("\n".join(
        rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()) + "\n")
    print(f"wrote {a.macros}")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
