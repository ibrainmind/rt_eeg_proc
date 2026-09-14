#!/usr/bin/env python3
"""
hep_figure.py -- the HEP-exposure figure for the companion note.

Plots the non-causal template of one showcase record in microvolts against time
since the R-peak, with the QRS-limited support and the 200-600 ms HEP interval
marked and annotated with their bias-corrected phase-locked RMS.

  python hep_figure.py --out ../paper/fig_real/fig_hep_template.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dataset_metric as dm
import phase_hep_analysis as ph

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--subject", default="sub-070")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--run", default="20")
    ap.add_argument("--eeg-index", type=int, default=1)
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--n-scramble", type=int, default=200)
    ap.add_argument("--out", default="../paper/fig_real/fig_hep_template.pdf")
    a = ap.parse_args()

    root = dm._resolve_dataset_root(a.dataset_root)
    rec = ph.analyse_run(root, a.subject, a.session, a.run, a.eeg_index, 0,
                         a.duration_sec, 0.0, a.n_scramble, False)
    h = rec["hep"]
    c = np.asarray(h["c_grid"])
    m = np.asarray(h["template"])
    print(json.dumps({k: v for k, v in h.items()
                      if k not in ("template", "c_grid")}, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.axvspan(ph.QRS_LO_S * 1e3, ph.QRS_HI_S * 1e3, color="tab:green", alpha=0.12)
    ax.axvspan(ph.HEP_LO_S * 1e3, ph.HEP_HI_S * 1e3, color="tab:red", alpha=0.12)
    w = (c >= -ph.PRE_S) & (c <= ph.POST_S)
    ax.plot(c[w] * 1e3, m[w], color="k", lw=1.1)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel("time since R (ms)")
    ax.set_ylabel(r"template ($\mu$V)")
    ax.text(0.02, 0.95,
            f"QRS-limited support\n{h['rms_uV_qrs']:.1f} " + r"$\mu$V rms",
            transform=ax.transAxes, fontsize=7.5, va="top", color="tab:green")
    ax.text(0.58, 0.95,
            f"HEP interval\n{h['rms_uV_hep']:.2f} " + r"$\mu$V rms" +
            f"\n{100 * h['hep_energy_fraction']:.0f}% of removable energy",
            transform=ax.transAxes, fontsize=7.5, va="top", color="tab:red")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
