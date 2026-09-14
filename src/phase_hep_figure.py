#!/usr/bin/env python3
"""
phase_hep_figure.py -- paper figure + summary numbers for the phase/HEP sections.

Panel (a): held-out advantage of normalized phase over fixed lag,
           D = vr(gamma=1) - vr(gamma=0) in dB, against robust RR variability,
           over every scanned record with a resolvable R-locked template.
Panel (b): the non-causal template of the showcase record in microvolts, with the
           QRS-limited support and the 200-600 ms HEP window marked and their
           bias-corrected R-locked RMS annotated.

  python phase_hep_figure.py --scan ../outputs/phase_hep_scan.json \
      --subject sub-070 --run 20 --eeg-index 1 \
      --out ../paper/fig_real/fig_phase_hep.pdf
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

SNR_GATE_DB = 10.0     # record has a template worth comparing coordinates on


def summarize(scan: list[dict], gate_db: float = SNR_GATE_DB) -> dict:
    rows = []
    for r in scan:
        g = {d["gamma"]: d for d in r["gamma_sweep"]}
        rows.append({
            "label": r["label"],
            "cv": r.get("rr_cv_robust", r["rr_cv"]),
            "cv_raw": r["rr_cv"],
            "snr_all": r["hep"]["snr_db_all"],
            "snr_hep": r["hep"]["snr_db_hep"],
            "rms_hep": r["hep"]["rms_uV_hep"],
            "rms_qrs": r["hep"]["rms_uV_qrs"],
            "hep_frac": r["hep"]["hep_energy_fraction"],
            "vr0": g[0.0]["vr_db_all"],
            "vr1": g[1.0]["vr_db_all"],
            "vrh0": g[0.0]["vr_db_hep"],
            "vrh1": g[1.0]["vr_db_hep"],
            "d": g[1.0]["vr_db_all"] - g[0.0]["vr_db_all"],
        })
    keep = [r for r in rows if np.isfinite(r["snr_all"]) and r["snr_all"] >= gate_db]
    out = {"n_records": len(rows), "n_kept": len(keep), "gate_db": gate_db}
    if keep:
        d = np.array([r["d"] for r in keep])
        cv = np.array([r["cv"] for r in keep])
        out.update({
            "delta_median_db": float(np.median(d)),
            "delta_min_db": float(np.min(d)),
            "delta_max_db": float(np.max(d)),
            "delta_abs_max_db": float(np.max(np.abs(d))),
            "cv_median": float(np.median(cv)),
            "cv_p90": float(np.percentile(cv, 90)),
            "cv_max": float(np.max(cv)),
            "hep_snr_median_db": float(np.median([r["snr_hep"] for r in keep])),
            "hep_rms_median_uV": float(np.median([r["rms_hep"] for r in keep])),
            "hep_rms_max_uV": float(np.max([r["rms_hep"] for r in keep])),
            "hep_frac_median": float(np.median([r["hep_frac"] for r in keep])),
            "hep_frac_max": float(np.max([r["hep_frac"] for r in keep])),
            "n_hep_significant": int(np.sum([r["snr_hep"] >= 3.0 for r in keep])),
        })
    out["_rows"] = rows
    out["_keep"] = keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="../outputs/phase_hep_scan.json")
    ap.add_argument("--synth", default="../outputs/phase_warp_synth.json")
    ap.add_argument("--gate-db", type=float, default=SNR_GATE_DB,
                    help="pass a large negative value when --scan is already gated")
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--subject", default="sub-070")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--run", default="20")
    ap.add_argument("--eeg-index", type=int, default=1)
    ap.add_argument("--ecg-index", type=int, default=0)
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--n-scramble", type=int, default=200)
    ap.add_argument("--out", default="../paper/fig_real/fig_phase_hep.pdf")
    a = ap.parse_args()

    scan = json.loads(Path(a.scan).read_text())
    s = summarize(scan)
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("_")},
                     indent=2, default=float))

    # showcase record
    root = dm._resolve_dataset_root(a.dataset_root)
    rec = ph.analyse_run(root, a.subject, a.session, a.run, a.eeg_index,
                         a.ecg_index, a.duration_sec, 0.0, a.n_scramble, False)
    hep = rec["hep"]
    c = np.asarray(hep["c_grid"])
    m = np.asarray(hep["template"])
    print("\nshowcase:", rec["label"])
    print(json.dumps({k: v for k, v in hep.items()
                      if k not in ("template", "c_grid")}, indent=2, default=float))
    print(json.dumps({d["gamma"]: d for d in rec["gamma_sweep"]}, indent=2, default=float))
    print(json.dumps({k: rec[k] for k in
                      ("n_beats", "rr_mean_s", "rr_sd_s", "rr_cv", "rr_cv_robust",
                       "rr_outlier_frac", "rr_min_s", "rr_max_s")}, indent=2, default=float))

    synth = json.loads(Path(a.synth).read_text()) if Path(a.synth).exists() else []

    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.3))

    # (a) coordinate comparison: synthetic expectation vs measured records
    keep = s["_keep"]
    cv = np.array([r["cv"] for r in keep]) * 100.0
    d = np.array([r["d"] for r in keep])
    sz = 6.0 + 1.6 * np.clip([r["snr_all"] for r in keep], 0, 40)
    if synth:
        scv = np.array([r["cv_robust"] for r in synth]) * 100.0
        sd_ = np.array([r["delta_db"] for r in synth])
        ax[0].plot(scv, sd_, "-o", color="tab:orange", ms=3, lw=1.2,
                   label="synthetic, phase-locked T-wave")
    ax[0].axhline(0.0, color="0.6", lw=0.8, zorder=1)
    ax[0].scatter(cv, d, s=sz, c="tab:blue", alpha=0.7, edgecolors="none", zorder=3,
                  label="SeizeIT2 recordings")
    ax[0].set_xlabel("robust RR variability $\\mathrm{CV}_{RR}$ (%)", fontsize=8)
    ax[0].set_ylabel("$\\Delta$ held-out VR (dB)", fontsize=8)
    ax[0].set_title("(a) normalized phase minus fixed lag", fontsize=8)
    ax[0].tick_params(labelsize=7)
    ax[0].legend(fontsize=6, loc="upper left", frameon=False)

    # (b) template with QRS / HEP support
    ax[1].axvspan(ph.QRS_LO_S * 1e3, ph.QRS_HI_S * 1e3, color="tab:green", alpha=0.12)
    ax[1].axvspan(ph.HEP_LO_S * 1e3, ph.HEP_HI_S * 1e3, color="tab:red", alpha=0.12)
    w = (c >= -ph.PRE_S) & (c <= ph.POST_S)
    ax[1].plot(c[w] * 1e3, m[w], color="k", lw=1.0)
    ax[1].axhline(0.0, color="0.6", lw=0.6)
    ax[1].set_xlabel("time since R (ms)", fontsize=8)
    ax[1].set_ylabel("template ($\\mu$V)", fontsize=8)
    ax[1].set_title("(b) template support and the HEP interval", fontsize=8)
    ax[1].tick_params(labelsize=7)
    ax[1].text(0.02, 0.95, f"QRS-limited\n{hep['rms_uV_qrs']:.1f} $\\mu$V rms",
               transform=ax[1].transAxes, fontsize=6.5, va="top", color="tab:green")
    ax[1].text(0.56, 0.95,
               f"HEP interval\n{hep['rms_uV_hep']:.2f} $\\mu$V rms\n"
               f"{100*hep['hep_energy_fraction']:.0f}% of energy",
               transform=ax[1].transAxes, fontsize=6.5, va="top", color="tab:red")

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
