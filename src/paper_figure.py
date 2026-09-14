#!/usr/bin/env python3
"""
paper_figure.py -- the paper's two-panel real-data figure.

(a) Coordinate choice: held-out advantage of normalized phase over fixed lag
    against robust RR variability, one point per recording with a resolvable
    template, with the synthetic expectation overlaid.
(b) Streaming cost: held-out suppression of the buffered mode against injected
    anchor jitter, median over the same recordings, with the non-causal,
    buffered and strict-causal operating points and the measured R-peak
    prediction error marked.

  python paper_figure.py --out ../paper/fig_real/fig_phase_hep.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import causal_modes as cm
import phase_hep_figure as pf

# Records whose R-peak prediction error is dominated by detector failures rather
# than by heart-rate variability are excluded from the streaming comparison; the
# paper's Note on detector requirements is exactly this precondition.
PRED_ERR_MAX_MS = 60.0
# The modes are compared only where the batch estimator removes a measurable
# amount; below that there is no suppression to retain and the ratio is noise.
NC_MIN_DB = 0.25


def summarize_modes(recs: list[dict], gamma: str = "1.0") -> dict:
    rows = []
    for r in recs:
        m = r["modes"][gamma]
        nc = r["vr_noncausal"][gamma]["all"]
        if not np.isfinite(m["pred_err_mad_ms"]) or m["pred_err_mad_ms"] > PRED_ERR_MAX_MS:
            continue
        if not np.isfinite(nc) or nc < NC_MIN_DB:
            continue
        jit = {j: m[f"vr_jit{j:g}"] for j in cm.JITTERS_MS}
        rows.append({
            "label": r["label"], "nc": nc,
            "buf": m["vr_buffered"], "strict": m["vr_strict"],
            "ret_buf": m["vr_buffered"] / nc,
            "ret_strict": m["vr_strict"] / nc,
            "pred_mad": m["pred_err_mad_ms"], "pred_sd": m["pred_err_sd_ms"],
            "jit": jit,
            # normalised by this record's own zero-jitter buffered suppression
            "jit_ret": {j: (jit[j] / jit[0.0]) if jit[0.0] > 0 else np.nan
                        for j in cm.JITTERS_MS},
        })
    if not rows:
        return {"n": 0, "_rows": rows}

    def med(k):
        return float(np.median([r[k] for r in rows]))

    return {
        "n": len(rows),
        "nc_med": med("nc"), "buf_med": med("buf"), "strict_med": med("strict"),
        "buf_cost_med": float(np.median([r["nc"] - r["buf"] for r in rows])),
        "strict_cost_med": float(np.median([r["buf"] - r["strict"] for r in rows])),
        "ret_buf_med": med("ret_buf"), "ret_strict_med": med("ret_strict"),
        "pred_mad_med": med("pred_mad"),
        "pred_mad_p90": float(np.percentile([r["pred_mad"] for r in rows], 90)),
        "jit_ret_med": {j: float(np.nanmedian([r["jit_ret"][j] for r in rows]))
                        for j in cm.JITTERS_MS},
        "_rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", default="../outputs/phase_hep_gated.json")
    ap.add_argument("--modes", default="../outputs/causal_modes.json")
    ap.add_argument("--synth", default="../outputs/phase_warp_synth.json")
    ap.add_argument("--out", default="../paper/fig_real/fig_phase_hep.pdf")
    ap.add_argument("--stats-out", default="../outputs/paper_stats.json")
    a = ap.parse_args()

    s = pf.summarize(json.loads(Path(a.gated).read_text()), gate_db=-1e9)
    modes = summarize_modes(json.loads(Path(a.modes).read_text()))
    synth = json.loads(Path(a.synth).read_text()) if Path(a.synth).exists() else []
    print(json.dumps({k: v for k, v in modes.items() if not k.startswith("_")},
                     indent=2, default=float))

    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.4))

    # --- (a) coordinate ---------------------------------------------------
    keep = s["_keep"]
    cv = np.array([r["cv"] for r in keep]) * 100.0
    d = np.array([r["d"] for r in keep])
    sz = 5.0 + 1.4 * np.clip([r["snr_all"] for r in keep], 0, 40)
    if synth:
        ax[0].plot([r["cv_robust"] * 100 for r in synth],
                   [r["delta_db"] for r in synth], "-o", color="tab:orange",
                   ms=2.6, lw=1.1, label="synthetic, phase-locked T-wave")
    ax[0].axhline(0.0, color="0.6", lw=0.8)
    ax[0].scatter(cv, d, s=sz, c="tab:blue", alpha=0.7, edgecolors="none", zorder=3,
                  label="SeizeIT2 recordings")
    ax[0].set_xlabel(r"robust RR variability $\mathrm{CV}_{RR}$ (%)", fontsize=8)
    ax[0].set_ylabel(r"$\Delta$ held-out VR (dB)", fontsize=8)
    ax[0].set_title("(a) normalized phase minus fixed lag", fontsize=8)
    ax[0].tick_params(labelsize=7)
    ax[0].legend(fontsize=6, loc="upper left", frameon=False)

    # --- (b) streaming cost ----------------------------------------------
    j = np.array(cm.JITTERS_MS)
    v = 100.0 * np.array([modes["jit_ret_med"][x] for x in cm.JITTERS_MS])
    ax[1].plot(j, v, "-o", color="tab:blue", ms=3, lw=1.2,
               label=r"buffered vs.\ injected jitter $\sigma_j$")
    ax[1].axhline(100.0, color="0.8", lw=0.6)
    ax[1].axhline(100.0 * modes["ret_strict_med"] / modes["ret_buf_med"],
                  color="tab:red", ls=":", lw=1.2,
                  label="strict causal (predicted anchor)")
    ax[1].axvline(modes["pred_mad_med"], color="tab:red", lw=0.8, alpha=0.5)
    ax[1].text(modes["pred_mad_med"] + 1.0, 92.0,
               f"measured prediction\nerror {modes['pred_mad_med']:.0f} ms",
               fontsize=6, va="top", color="tab:red")
    ax[1].set_xlabel(r"anchor jitter $\sigma_j$ (ms)", fontsize=8)
    ax[1].set_ylabel("suppression retained (%)", fontsize=8)
    ax[1].set_title("(b) what one beat of latency buys", fontsize=8)
    ax[1].tick_params(labelsize=7)
    ax[1].legend(fontsize=6, loc="lower left", frameon=False)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")

    stats = {k: v for k, v in modes.items() if not k.startswith("_")}
    stats.update({k: v for k, v in s.items() if not k.startswith("_")})
    Path(a.stats_out).write_text(json.dumps(stats, indent=2, default=float))
    print(f"wrote {a.stats_out}")


if __name__ == "__main__":
    main()
