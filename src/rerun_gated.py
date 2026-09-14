#!/usr/bin/env python3
"""Re-measure only the recordings with a resolvable template, after a window change.

Reads a prior scan, selects the records passing the SNR gate, and re-runs them so
the QRS/HEP sub-windows partition the analysis window exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dataset_metric as dm
import phase_hep_analysis as ph
import phase_hep_figure as pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="../outputs/phase_hep_scan_full.json")
    ap.add_argument("--out", default="../outputs/phase_hep_gated.json")
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--n-scramble", type=int, default=100)
    ap.add_argument("--gate-db", type=float, default=pf.SNR_GATE_DB)
    a = ap.parse_args()

    scan = json.loads(Path(a.scan).read_text())
    want = {(r["subject"], r["session"], r["run"]): None
            for r in scan if r["hep"]["snr_db_all"] >= a.gate_db}
    keep_ei = {}
    for r in scan:
        if r["hep"]["snr_db_all"] >= a.gate_db:
            keep_ei.setdefault((r["subject"], r["session"], r["run"]), set()).add(r["eeg_index"])
    print(f"{len(scan)} records analysed, {sum(len(v) for v in keep_ei.values())} gated "
          f"across {len(keep_ei)} runs")

    root = dm._resolve_dataset_root(a.dataset_root)
    out = []
    for (sub, ses, run), eis in sorted(keep_ei.items()):
        try:
            chans = ph.analyse_run_all_channels(root, sub, ses, run, 0, a.duration_sec,
                                                0.0, a.n_scramble, False)
        except Exception as e:
            print(f"[skip] {sub}/{run}: {type(e).__name__}: {e}", flush=True)
            continue
        for r in chans:
            if r["eeg_index"] not in eis:
                continue
            out.append(r)
            g = {d["gamma"]: d["vr_db_all"] for d in r["gamma_sweep"]}
            print(f"{sub}/run-{run}/eeg{r['eeg_index']}  cv={r['rr_cv_robust']:.3f} "
                  f"D={g[1.0]-g[0.0]:+.3f}dB  HEPsnr={r['hep']['snr_db_hep']:+.1f} "
                  f"HEPrms={r['hep']['rms_uV_hep']:.2f}uV "
                  f"frac={r['hep']['hep_energy_fraction']:.3f}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out} ({len(out)} records)")


if __name__ == "__main__":
    main()
