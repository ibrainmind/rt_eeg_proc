#!/usr/bin/env python3
"""
generate_figures.py -- regenerate paper figures from code.

Figure mapping (paper labels):
- Figure 1 (`fig:real_aligned3beats`): real aligned ECG/EEG segment from ds005873.
- Figure 4 (`fig:conv`): semi-synthetic convergence and spectrum.
- Figure 5 (`fig:time`): semi-synthetic time-domain cleanup.
- Table 1 (`tab:results`): generated semi-synthetic SNR summary.

Requires src/ekg_ilc_rt.py (the reference implementation of the synthesis:
signal generation, phase estimator, ILC/repetitive canceller, NRLS residual)
to be importable from this repository.

Usage:
    # Regenerate all manuscript assets (Figure 1 + synthetic figures + table)
    python paper/generate_figures.py

    # Generate only Figure 1 from real dataset segment
    python paper/generate_figures.py --only-fig1-real \
      --dataset-root datasets/ds005873 --subject sub-001 --session ses-01 --run 01 \
      --fig1-start-sec 64 --fig1-duration-sec 2.5 --fig1-highpass-hz 0.5 --fig1-notch-hz 50

    # Synthetic figures/table only (skip real-data Figure 1)
    python paper/generate_figures.py --skip-fig1-real
Produces:
    fig_real/fig_real_aligned_3beats.png
    fig1_convergence_spectrum.pdf
    fig2_timedomain.pdf
    tab_results_generated.tex
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import ekg_ilc_rt as C
import process_data as P

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

FS = 250.0
DUR = 90.0
SEED = 11


def run_pipeline():
    t, y, s, d, r, beat_t, beat_type = C.gen_eeg(FS, DUR, SEED, True, 0.20, 3.0)
    est = C.phase_estimator(r, FS, True, True)

    results = {}
    for mode in ["noncausal", "buffered", "causal"]:
        e_ilc, templ = C.ilc_clean(y, FS, est, mode, True)
        e_final = C.rls_clean(e_ilc, r, FS, mode, True, True)
        results[mode] = (e_ilc, e_final, templ)
    return t, y, s, d, r, est, results


def snr_db_truth(s, est, mask):
    return 10 * np.log10(np.sum(s[mask] ** 2) / np.sum((s[mask] - est[mask]) ** 2))


def compute_method_summary(t, y, s, r, results):
    ss = t > t[-1] * 0.5
    summary = []
    summary.append(("Input (contaminated)", "--", snr_db_truth(s, y, ss)))

    # Linear ANC baseline: no ILC, linear reference only (T1), causal RLS.
    e_linear = C.rls_clean(y, r, FS, "causal", True, False)
    summary.append(("Linear RLS ANC", "causal", snr_db_truth(s, e_linear, ss)))

    summary.append(("Proposed (causal)", "0", snr_db_truth(s, results["causal"][1], ss)))
    summary.append(("Proposed (buffered)", "~1 beat", snr_db_truth(s, results["buffered"][1], ss)))
    summary.append(("Proposed (non-causal)", "offline", snr_db_truth(s, results["noncausal"][1], ss)))
    return summary


def write_summary_table_tex(summary_rows, out_path="tab_results_generated.tex"):
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Steady-state artifact SNR (semi-synthetic; generated).}",
        r"\label{tab:results}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Method & Latency & SNR (dB) \\",
        r"\midrule",
    ]
    for method, latency, snr in summary_rows:
        lines.append("{} & {} & {:.2f} \\\\".format(method, latency, snr))
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def fig1_convergence_spectrum(t, y, s, results, fs):
    N = len(t)
    ss = t > t[-1] * 0.5

    def windowed_snr(est, win=int(8 * fs)):
        centers, vals = [], []
        for a in range(0, N - win, win // 2):
            b = a + win
            vals.append(10 * np.log10(np.sum(s[a:b] ** 2) / np.sum((s[a:b] - est[a:b]) ** 2)))
            centers.append((a + b) / 2 / fs)
        return np.array(centers), np.array(vals)

    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.9))

    tc, snr_in = windowed_snr(y)
    ax[0].plot(tc, snr_in, label="input", lw=1.1, color="0.5")
    for mode, color in zip(["causal", "buffered", "noncausal"], ["C3", "C0", "C2"]):
        _, e_final, _ = results[mode]
        _, snr_out = windowed_snr(e_final)
        ax[0].plot(tc, snr_out, label=mode, lw=1.3, color=color)
    ax[0].set_title("(a) Online SNR convergence")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel(r"artifact SNR (dB)")
    ax[0].legend(ncol=2)
    ax[0].grid(alpha=0.3)

    _, e_final_buf, _ = results["buffered"]
    f_ax, P_y = signal.welch(y[ss], fs, nperseg=1024)
    _, P_clean = signal.welch(e_final_buf[ss], fs, nperseg=1024)
    _, P_s = signal.welch(s[ss], fs, nperseg=1024)
    ax[1].semilogy(f_ax, P_y, lw=1.0, alpha=0.8, label="contaminated", color="0.5")
    ax[1].semilogy(f_ax, P_clean, lw=1.4, label="cleaned", color="C0")
    ax[1].semilogy(f_ax, P_s, ":", lw=1.4, label="true", color="C2")
    ax[1].axvspan(9, 11, color="green", alpha=0.12)
    ax[1].set_xlim(0, 40)
    ax[1].set_title("(b) Spectrum (buffered mode)")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("PSD")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("fig1_convergence_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_timedomain(t, y, s, d, r, est, results, fs):
    e_ilc_buf, e_final_buf, _ = results["buffered"]
    R = np.array([f[0] for f in est["fids"]])

    z0, z1 = int(40 * fs), int(43.2 * fs)
    tt = t[z0:z1]

    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.0), sharex=True)

    ax[0].plot(tt, r[z0:z1], lw=0.8, color="0.2")
    for Rk in R:
        if z0 <= Rk < z1:
            ax[0].axvline(Rk / fs, color="C2", lw=0.7, alpha=0.6)
    ax[0].set_title("(a) Synthetic ECG reference (PQRST)")
    ax[0].set_ylabel(r"$r(t)$")
    ax[0].grid(alpha=0.3)

    ax[1].plot(tt, y[z0:z1], lw=0.7, color="0.55", label="contaminated")
    ax[1].plot(tt, e_final_buf[z0:z1], lw=1.0, color="C0", label="cleaned")
    ax[1].plot(tt, s[z0:z1], lw=1.0, color="C2", ls=":", label="true neural")
    ax[1].set_title("(b) EEG: contaminated vs. cleaned vs. truth")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel(r"$y(t)$")
    ax[1].legend(loc="upper right")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig2_timedomain.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_real_aligned(
    dataset_root: str,
    subject: str,
    session: str,
    run: str,
    start_sec: float,
    duration_sec: float,
    eeg1_index: int,
    eeg2_index: int,
    ecg_index: int,
    highpass_hz: float,
    notch_hz: float,
    out_path: str,
) -> None:
    """Generate Figure 1: short aligned real ECG/EEG segment from ds005873."""
    ds_path = dataset_root
    if not os.path.isabs(ds_path):
        ds_path = os.path.join(REPO_ROOT, ds_path)

    t, eeg1, eeg2, ecg, fs, fig_label, eeg1_name, eeg2_name, ecg_name = P.load_aligned_real_segment(
        dataset_root=Path(os.path.abspath(ds_path)),
        subject=subject,
        session=session,
        run=run,
        pull=True,
        eeg_indices=(eeg1_index, eeg2_index),
        ecg_index=ecg_index,
        start_sec=start_sec,
        duration_sec=duration_sec,
        highpass_hz=highpass_hz,
        notch_hz=notch_hz,
    )

    # Run the phase tracker on a padded interval to avoid edge effects in short windows.
    pad_sec = 6.0
    t_pad, _, _, ecg_pad, _, _, _, _, _ = P.load_aligned_real_segment(
        dataset_root=Path(os.path.abspath(ds_path)),
        subject=subject,
        session=session,
        run=run,
        pull=True,
        eeg_indices=(eeg1_index, eeg2_index),
        ecg_index=ecg_index,
        start_sec=max(0.0, start_sec - pad_sec),
        duration_sec=duration_sec + 2.0 * pad_sec,
        highpass_hz=highpass_hz,
        notch_hz=notch_hz,
    )
    est = C.phase_estimator(ecg_pad, fs, use_pll=True, use_override=True)
    fid_idx_pad = np.array([int(fid) for fid, _, _ in est.get("fids", [])], dtype=int)
    fid_idx_pad = fid_idx_pad[(fid_idx_pad >= 0) & (fid_idx_pad < len(t_pad))]
    r_times_all = t_pad[fid_idx_pad] if fid_idx_pad.size > 0 else np.array([], dtype=float)
    r_times = r_times_all[(r_times_all >= t[0]) & (r_times_all <= t[-1])]

    shade_half_width = 0.06
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.8), sharex=True)

    # Top subplot: ECG/EKG with detected R-peaks.
    axes[0].plot(t, ecg, lw=0.9, color="C3", label=ecg_name)
    if r_times.size > 0:
        r_vals = np.interp(r_times, t, ecg)
        axes[0].scatter(r_times, r_vals, s=14, color="k", zorder=4, label="R peaks")
    axes[0].set_ylabel("V")
    axes[0].set_title("ECG/EKG")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    # Middle and bottom subplots: EEG channels.
    axes[1].plot(t, eeg1, lw=0.8, color="C0")
    axes[1].set_ylabel("V")
    axes[1].set_title(f"EEG 1: {eeg1_name}")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, eeg2, lw=0.8, color="C1")
    axes[2].set_ylabel("V")
    axes[2].set_title(f"EEG 2: {eeg2_name}")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)

    # Highlight R-peak neighborhoods on all three subplots.
    for ax in axes:
        for rt in r_times:
            ax.axvspan(rt - shade_half_width, rt + shade_half_width, color="C1", alpha=0.10, lw=0)
        for rt in r_times:
            ax.axvline(rt, color="0.45", lw=0.7, ls="--", alpha=0.55)

    fig.suptitle(
        "EAR-EEG cardiac arfifacts from SeizeIT2 openneuro dataset",
        fontsize=12,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])

    out_abs = out_path
    if not os.path.isabs(out_abs):
        out_abs = os.path.join(THIS_DIR, out_abs)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    plt.savefig(out_abs, dpi=150)
    plt.close(fig)
    print(f"wrote Figure 1: {out_abs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures.")
    parser.add_argument("--dataset-root", default="datasets/ds005873", help="Dataset root path")
    parser.add_argument("--subject", default="sub-001", help="Subject ID")
    parser.add_argument("--session", default="ses-01", help="Session ID")
    parser.add_argument("--run", default="01", help="Run ID")
    parser.add_argument("--fig1-start-sec", type=float, default=64.0, help="Figure 1 start time (s)")
    parser.add_argument("--fig1-duration-sec", type=float, default=2.5, help="Figure 1 duration (s)")
    parser.add_argument("--fig1-eeg1-index", type=int, default=0, help="Figure 1 EEG index #1")
    parser.add_argument("--fig1-eeg2-index", type=int, default=1, help="Figure 1 EEG index #2")
    parser.add_argument("--fig1-ecg-index", type=int, default=0, help="Figure 1 ECG index")
    parser.add_argument("--fig1-highpass-hz", type=float, default=0.5, help="Figure 1 EEG high-pass (Hz)")
    parser.add_argument("--fig1-notch-hz", type=float, default=50.0, help="Figure 1 EEG notch (Hz)")
    parser.add_argument(
        "--fig1-out",
        default="fig_real/fig_real_aligned_3beats.png",
        help="Figure 1 output path (relative to paper/ or absolute)",
    )
    parser.add_argument("--skip-fig1-real", action="store_true", help="Skip Figure 1 real-data generation")
    parser.add_argument("--only-fig1-real", action="store_true", help="Generate only Figure 1")
    return parser.parse_args()


def print_summary_table(summary_rows):
    print("Method summary (steady-state SNR):")
    for method, latency, snr in summary_rows:
        print(f"{method:24s} | {latency:8s} | {snr:5.2f} dB")


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_fig1_real:
        fig1_real_aligned(
            dataset_root=args.dataset_root,
            subject=args.subject,
            session=args.session,
            run=args.run,
            start_sec=args.fig1_start_sec,
            duration_sec=args.fig1_duration_sec,
            eeg1_index=args.fig1_eeg1_index,
            eeg2_index=args.fig1_eeg2_index,
            ecg_index=args.fig1_ecg_index,
            highpass_hz=args.fig1_highpass_hz,
            notch_hz=args.fig1_notch_hz,
            out_path=args.fig1_out,
        )

    if args.only_fig1_real:
        sys.exit(0)

    t, y, s, d, r, est, results = run_pipeline()
    fig1_convergence_spectrum(t, y, s, results, FS)
    fig2_timedomain(t, y, s, d, r, est, results, FS)
    summary_rows = compute_method_summary(t, y, s, r, results)
    write_summary_table_tex(summary_rows)
    print_summary_table(summary_rows)
    print("wrote fig1_convergence_spectrum.pdf, fig2_timedomain.pdf, and tab_results_generated.tex")
