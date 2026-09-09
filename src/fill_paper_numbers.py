#!/usr/bin/env python3
"""
fill_paper_numbers.py -- substitute measured numbers into paper.tex and hep_note.tex.

Every @@TOKEN@@ in the .tokens template is filled from the analysis outputs, so
the prose, the tables and the figures cannot drift apart.  Sources:

  outputs/phase_hep_scan_full.json  every recording analysed (denominator only)
  outputs/phase_hep_gated.json      the recordings with a resolvable template
  outputs/causal_modes.json         non-causal / buffered / strict comparison
  the showcase record               re-analysed directly for the HEP note

  python fill_paper_numbers.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import causal_modes as cm
import dataset_metric as dm
import paper_figure as pfig
import phase_hep_analysis as ph
import phase_hep_figure as pf


def build_values(a) -> dict:
    full = json.loads(Path(a.scan).read_text())
    gated = json.loads(Path(a.gated).read_text())
    s = pf.summarize(gated, gate_db=-1e9)      # already gated
    keep = s["_keep"]
    d_hep = sorted(r["vrh1"] - r["vrh0"] for r in keep)
    med_hep = d_hep[len(d_hep) // 2]

    m = pfig.summarize_modes(json.loads(Path(a.modes).read_text()))
    jr = m["jit_ret_med"]

    root = dm._resolve_dataset_root(a.dataset_root)
    rec = ph.analyse_run(root, a.subject, a.session, a.run, a.eeg_index, 0,
                         600.0, 0.0, a.n_scramble, False)
    h = rec["hep"]

    return {
        # corpus
        "NREC": f"{len(full)}",
        "NKEPT": f"{len(gated)}",
        "NHEPSIG": f"{s['n_hep_significant']}",
        # coordinate comparison
        "DMED": f"{s['delta_median_db']:+.3f}",
        "DMIN": f"{s['delta_min_db']:+.2f}",
        "DMAX": f"{s['delta_max_db']:+.2f}",
        "DABS": f"{s['delta_abs_max_db']:.2f}",
        "DHEPMED": f"{med_hep:+.3f}",
        "CVMED": f"{100 * s['cv_median']:.1f}",
        "CVMAX": f"{100 * s['cv_max']:.1f}",
        # streaming modes
        "NMODE": f"{m['n']}",
        "NCMIN": f"{pfig.NC_MIN_DB:.2f}",
        "NCMED": f"{m['nc_med']:+.2f}",
        "BUFMED": f"{m['buf_med']:+.2f}",
        "STRICTMED": f"{m['strict_med']:+.2f}",
        "RETBUF": f"{100 * m['ret_buf_med']:.0f}",
        "PREDMAD": f"{m['pred_mad_med']:.0f}",
        "PREDP90": f"{m['pred_mad_p90']:.0f}",
        "AGAIN": f"{cm.A_GAIN:g}",
        "WARMUP": f"{cm.WARMUP_BEATS}",
        "JIT2": f"{100 * jr[2.0]:.0f}",
        "JIT5": f"{100 * jr[5.0]:.0f}",
        "JIT10": f"{100 * jr[10.0]:.0f}",
        "JIT20": f"{100 * jr[20.0]:.0f}",
        # HEP note
        "HEPSNRMED": f"{s['hep_snr_median_db']:.1f}",
        "HEPRMSMED": f"{s['hep_rms_median_uV']:.2f}",
        "HEPRMSMAX": f"{s['hep_rms_max_uV']:.2f}",
        "HEPFRACMED": f"{100 * s['hep_frac_median']:.0f}",
        "HEPFRACMAX": f"{100 * s['hep_frac_max']:.0f}",
        "SHOWQRS": f"{h['rms_uV_qrs']:.1f}",
        "SHOWHEPSNR": f"{h['snr_db_hep']:.1f}",
        "HEPRMS": f"{h['rms_uV_hep']:.2f}",
        "HEPFRAC": f"{100 * h['hep_energy_fraction']:.0f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="../outputs/phase_hep_scan_full.json")
    ap.add_argument("--gated", default="../outputs/phase_hep_gated.json")
    ap.add_argument("--modes", default="../outputs/causal_modes.json")
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--subject", default="sub-070")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--run", default="20")
    ap.add_argument("--eeg-index", type=int, default=1)
    ap.add_argument("--n-scramble", type=int, default=200)
    ap.add_argument("--docs", default="../paper/paper.tex,../paper/hep_note.tex")
    a = ap.parse_args()

    vals = build_values(a)
    print(json.dumps(vals, indent=2))

    for doc in a.docs.split(","):
        out = Path(doc)
        tpl = out.with_suffix(out.suffix + ".tokens")
        src = tpl if tpl.exists() else out
        txt = src.read_text()
        for k, v in vals.items():
            txt = txt.replace(f"@@{k}@@", v)
        out.write_text(txt)
        left = [ln.strip()[:90] for ln in txt.splitlines() if "@@" in ln]
        print(f"{out}: {len(left)} unfilled")
        for ln in left:
            print("   ", ln)


if __name__ == "__main__":
    main()
