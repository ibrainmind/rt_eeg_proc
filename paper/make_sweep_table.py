#!/usr/bin/env python3
"""
make_sweep_table.py -- the reference-predictability sweep as a compact table.

Replaces fig3_synthetic_coherence_sweep.pdf with tab_sweep_generated.tex so the
same baseline comparison fits the ICASSP page budget.  Reuses the generator and
the cancellers of generate_figures.py / ekg_ilc_rt.py unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal

import generate_figures as G
import ekg_ilc_rt as C


def sweep_rows(fs=G.FS, dur=75.0, seeds=(11, 13, 17), fracs=(1.0, 0.5, 0.0)):
    rows = []
    for pf in fracs:
        coh, rls, non, buf = [], [], [], []
        for seed in seeds:
            t, y, s, cfa, r = G.synth_eeg_predictability_sweep(fs, dur, seed, pf)
            est = C.phase_estimator(r, fs, True, True)
            ra, Lw = C._template_dims(fs)
            valid = [fid for fid, ty, lg in est["fids"]
                     if lg and fid - ra >= 0 and fid - ra + Lw <= len(r)]
            if len(valid) < 8:
                continue
            ecg_avg = np.mean([r[f - ra:f - ra + Lw] for f in valid], axis=0)
            cfa_avg = np.mean([cfa[f - ra:f - ra + Lw] for f in valid], axis=0)
            f_coh, cxy = signal.coherence(ecg_avg, cfa_avg, fs=fs,
                                          nperseg=min(64, len(ecg_avg)))
            band = (f_coh >= 1.0) & (f_coh <= 20.0)
            e_rls = C.rls_clean(y, r, fs, "noncausal", True, False)
            e_non, _ = C.ilc_clean(y, fs, est, "noncausal", True)
            e_buf, _ = C.ilc_clean(y, fs, est, "buffered", True)
            ss = t > (t[-1] * 0.5)
            raw = G.snr_db_truth(s, y, ss)
            coh.append(float(np.mean(cxy[band])))
            rls.append(G.snr_db_truth(s, e_rls, ss) - raw)
            non.append(G.snr_db_truth(s, e_non, ss) - raw)
            buf.append(G.snr_db_truth(s, e_buf, ss) - raw)
        rows.append(tuple(float(np.mean(v)) for v in (coh, rls, non, buf)))
        print(f"pf={pf:.2f}  coh={rows[-1][0]:.2f}  rls={rows[-1][1]:+.2f}  "
              f"buf={rows[-1][3]:+.2f}  non={rows[-1][2]:+.2f}", flush=True)
    return rows


def write_tex(rows, out: Path):
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Semi-synthetic reference-predictability sweep: SNR improvement over the",
        r"contaminated input (dB), measured against the true neural signal. Coherence is the mean",
        r"$1$--$20$~Hz coherence between the averaged ECG beat and the averaged \emph{true} CFA beat.",
        r"RLS-ANC is granted a sample-aligned ECG waveform throughout, i.e.\ its best case.}",
        r"\label{tab:sweep}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"ECG-to-CFA coherence & RLS-ANC & Buffered & Non-causal \\",
        r"\midrule",
    ]
    for coh, rls, non, buf in rows:
        lines.append(f"${coh:.2f}$ & ${rls:+.1f}$ & ${buf:+.1f}$ & ${non:+.1f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tab_sweep_generated.tex")
    a = ap.parse_args()
    write_tex(sweep_rows(), Path(a.out))


if __name__ == "__main__":
    main()
