#!/usr/bin/env python3
"""
generate_figures.py -- regenerate paper figures from code.

Figure mapping (paper labels):
- Figure 1 (`fig:real_aligned3beats`): real aligned ECG/EEG segment from ds005873.
- Figure 3 (`fig:synthetic_coh_sweep`): synthetic coherence-versus-cancellation sweep.
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
    fig3_synthetic_coherence_sweep.pdf
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
from matplotlib.ticker import FormatStrFormatter
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
    _, e_final_causal, _ = results["causal"]
    f_ax, P_y = signal.welch(y[ss], fs, nperseg=1024)
    _, P_buffered = signal.welch(e_final_buf[ss], fs, nperseg=1024)
    _, P_causal = signal.welch(e_final_causal[ss], fs, nperseg=1024)
    ax[1].semilogy(f_ax, P_y, lw=1.0, alpha=0.8, label="original", color="0.5")
    ax[1].semilogy(f_ax, P_buffered, lw=1.4, label="buffered", color="C0")
    ax[1].semilogy(f_ax, P_causal, lw=1.4, label="strict real-time", color="C3")
    ax[1].axvspan(9, 11, color="green", alpha=0.12)
    ax[1].set_xlim(0, 40)
    ax[1].set_title("(b) Spectrum: original, buffered, and strict real-time")
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


def _alt_cfa_waveform(fs: float) -> np.ndarray:
    tt = np.arange(-0.30, 0.50, 1.0 / fs)
    g = lambda c, a, w: a * np.exp(-0.5 * ((tt - c) / w) ** 2)
    return (
        g(-0.12, 0.10, 0.045)
        + g(0.03, -0.30, 0.030)
        + g(0.13, 0.22, 0.040)
        + g(0.26, -0.10, 0.055)
    )


def _place_waveform_on_beats(fs: float, dur: float, beat_t: np.ndarray, waveform: np.ndarray) -> np.ndarray:
    N = int(fs * dur)
    out = np.zeros(N)
    ra = int(0.30 * fs)
    for bt in beat_t:
        idx = int(round(bt * fs))
        lo = idx - ra
        a0 = max(0, -lo)
        a1 = min(len(waveform), N - lo)
        if a1 > a0:
            out[max(lo, 0):max(lo, 0) + (a1 - a0)] += waveform[a0:a1]
    return out


def synth_eeg_predictability_sweep(
    fs: float,
    dur: float,
    seed: int,
    predictable_frac: float,
    drift: float = 0.20,
    artifact_gain: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic EEG with fixed artifact power and swept ECG-to-artifact predictability.

    `predictable_frac=1` is largely ECG-coherent linear artifact.
    `predictable_frac=0` is phase-locked but waveform-decorrelated artifact.
    """
    d, beat_t, btype, rng = C.gen_sources(fs, dur, seed, anomalies=False)
    N = len(d)
    t = np.arange(N) / fs

    linear_h = np.array([0.05, 0.20, 0.35, 0.25, 0.10, 0.05], dtype=float)
    linear_h /= linear_h.sum()
    cfa_linear = signal.lfilter(linear_h, 1.0, d)

    alt = _place_waveform_on_beats(fs, dur, beat_t, _alt_cfa_waveform(fs))

    cfa_linear /= np.std(cfa_linear) + 1e-12
    alt /= np.std(alt) + 1e-12
    cfa_core = predictable_frac * cfa_linear + (1.0 - predictable_frac) * alt
    cfa_core /= np.std(cfa_core) + 1e-12

    dr = 1.0 + drift * np.sin(2 * np.pi * 0.02 * t)
    cfa = dr * cfa_core * artifact_gain

    white = rng.normal(0, 1, N)
    pink = signal.lfilter(
        [0.049922, -0.095993, 0.050612, -0.004408],
        [1, -2.494956, 2.017265, -0.522189],
        white,
    )
    pink /= np.std(pink) + 1e-12
    alpha = (0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t)) * np.sin(2 * np.pi * 10 * t)
    s = 0.7 * pink + 0.6 * alpha
    y = s + cfa + rng.normal(0, 0.10, N)
    r = d + rng.normal(0, 0.03, N)
    return t, y, s, cfa, r


def fig3_synthetic_coherence_sweep(
    out_path: str = "fig3_synthetic_coherence_sweep.pdf",
    fs: float = FS,
    dur: float = 75.0,
    seeds: tuple[int, ...] = (11, 13, 17),
) -> None:
    """Generate a synthetic sweep with separate template calibration and cancellation passes."""
    def apply_calibrated_template(y, fs, est, template, mode, learning_gain=0.06):
        """Apply a calibrated template with one delayed update per beat."""
        n = len(y)
        ra, width = C._template_dims(fs)
        fids = est["fids"]
        peaks = np.asarray([fid for fid, _, _ in fids], dtype=int)
        learnable = np.asarray([lg for _, _, lg in fids], dtype=bool)
        template = template.copy()
        out = y.copy()
        fid_index = 0
        locked_fid = None
        for sample in range(n):
            while fid_index < len(peaks) and sample >= peaks[fid_index]:
                current_index = fid_index
                locked_fid = peaks[fid_index]
                fid_index += 1
                previous_index = current_index - 1
                if previous_index >= 0 and learnable[previous_index]:
                    previous_fid = peaks[previous_index]
                    lo = max(0, previous_fid - ra)
                    hi = min(n, locked_fid)
                    for delayed_sample in range(lo, hi):
                        offset = delayed_sample - previous_fid + ra
                        if 0 <= offset < width:
                            if mode == "buffered":
                                out[delayed_sample] = y[delayed_sample] - template[offset]
                            template[offset] += learning_gain * (y[delayed_sample] - template[offset])
            if mode == "buffered":
                # The current beat remains buffered until its following R.
                continue
            else:
                # Before the next R arrives, use the loop's predicted anchor;
                # after lock, use the detected R anchor for the remainder.
                phase = est["phi"][sample]
                rr_hat = est["rrhat"][sample]
                time_to_next = (1.0 - phase) * rr_hat
                if time_to_next <= 0.30:
                    anchor = sample - phase * rr_hat * fs
                else:
                    anchor = locked_fid
            if anchor is None:
                continue
            offset = int(round(sample - anchor + ra))
            if 0 <= offset < width:
                out[sample] = y[sample] - template[offset]
        return out

    predictable_fracs = np.linspace(0.0, 1.0, 9)
    rows = []

    for pf in predictable_fracs:
        coh_vals = []
        rls_vals = []
        non_vals = []
        buf_vals = []
        strict_vals = []
        for seed in seeds:
            t, y, s, cfa, r = synth_eeg_predictability_sweep(fs, dur, seed, pf)
            est = C.phase_estimator(r, fs, True, True)

            ra, Lw = C._template_dims(fs)
            valid = [fid for fid, ty, lg in est["fids"] if lg and fid - ra >= 0 and fid - ra + Lw <= len(r)]
            if len(valid) < 8:
                continue

            ecg_beats = np.vstack([r[fid - ra:fid - ra + Lw] for fid in valid])
            cfa_beats = np.vstack([cfa[fid - ra:fid - ra + Lw] for fid in valid])
            ecg_avg = np.mean(ecg_beats, axis=0)
            cfa_avg = np.mean(cfa_beats, axis=0)
            f_coh, cxy = signal.coherence(ecg_avg, cfa_avg, fs=fs, nperseg=min(64, len(ecg_avg)))
            band = (f_coh >= 1.0) & (f_coh <= 20.0)
            if not np.any(band):
                continue

            e_rls = C.rls_clean(y, r, fs, "noncausal", True, False)
            # Pass 1: obtain a converged full-record calibration template. Pass
            # 2: cancel with one delayed learning update per completed beat.
            _, calibrated_template = C.ilc_clean(y, fs, est, "noncausal", True)
            e_non, _ = C.ilc_clean(y, fs, est, "noncausal", True)
            e_buf = apply_calibrated_template(y, fs, est, calibrated_template, "buffered")
            e_strict = apply_calibrated_template(y, fs, est, calibrated_template, "strict")
            # Compare all methods only after the streaming ILC has warmed up;
            # the non-causal curve otherwise gets an unfair full-record template.
            warmup_beats = min(60, len(valid) - 1)
            warmup_time = valid[warmup_beats] / fs
            # The buffered mode releases a beat only when the following R
            # arrives, so at the end of the record its last beat is still in
            # the buffer and no output exists for it. Close the scoring window
            # at the last released sample, identically for every method, so the
            # comparison is not charged for that one uncancelled beat.
            all_peaks = [int(fid) for fid, _, _ in est["fids"]]
            scored_end = len(t)
            if len(all_peaks) >= 2:
                scored_end = min(all_peaks[-2] - ra + Lw, all_peaks[-1])
            ss = (t >= warmup_time) & (np.arange(len(t)) < scored_end)
            raw_snr = snr_db_truth(s, y, ss)

            coh_vals.append(float(np.mean(cxy[band])))
            rls_vals.append(snr_db_truth(s, e_rls, ss) - raw_snr)
            non_vals.append(snr_db_truth(s, e_non, ss) - raw_snr)
            buf_vals.append(snr_db_truth(s, e_buf, ss) - raw_snr)
            strict_vals.append(snr_db_truth(s, e_strict, ss) - raw_snr)

        if coh_vals:
            rows.append((
                float(np.mean(coh_vals)),
                float(np.mean(rls_vals)),
                float(np.mean(non_vals)),
                float(np.mean(buf_vals)),
                float(np.mean(strict_vals)),
            ))

    if not rows:
        raise RuntimeError("Synthetic coherence sweep produced no valid points.")

    arr = np.asarray(rows)
    order = np.argsort(arr[:, 0])
    coh_x = arr[order, 0]
    rls_y = arr[order, 1]
    non_y = arr[order, 2]
    buf_y = arr[order, 3]
    strict_y = arr[order, 4]

    fig, ax = plt.subplots(figsize=(3.8, 1.45))
    ax.plot(coh_x, rls_y, "o-", color="0.35", lw=1.2, ms=2.4, label="RLS cancellation")
    ax.plot(coh_x, non_y, "s-", color="C0", lw=1.3, ms=2.4, label="Non-causal cancellation [3]")
    ax.plot(coh_x, buf_y, "^-", color="C2", lw=1.3, ms=2.4, label="Proposed buffered cancellation")
    ax.plot(coh_x, strict_y, "D-", color="C3", lw=1.3, ms=2.2, label="Strict real-time cancellation")
    ax.set_xlabel("mean beat-window coherence: ECG vs true CFA", fontsize=5)
    ax.set_ylabel("CFA removal (dB)", fontsize=5)
    ax.set_title("Synthetic sweep: predictability vs cancellation", fontsize=5.5)
    ax.set_xlim(0.9, 0.3)
    ax.set_xticks(np.arange(0.9, 0.2, -0.1))
    ax.set_yticks([0.0, 10.0, 20.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", labelsize=4.8)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=3.8, markerscale=0.75,
              frameon=False, borderaxespad=0.0)
    plt.tight_layout(rect=[0.0, 0.0, 0.74, 1.0])
    plt.savefig(out_path, bbox_inches="tight")
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
    axes[1].plot(t, eeg1, lw=0.8, color="C0", label=eeg1_name)
    axes[1].set_ylabel("V")
    axes[1].set_title(f"EEG 1: {eeg1_name}")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right")

    axes[2].plot(t, eeg2, lw=0.8, color="C1", label=eeg2_name)
    axes[2].set_ylabel("V")
    axes[2].set_title(f"EEG 2: {eeg2_name}")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="lower right")

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


def fig2_noncausal_cleanup(
    dataset_root: str,
    subject: str,
    session: str,
    run: str,
    start_sec: float,
    duration_sec: float,
    eeg2_index: int,
    ecg_index: int,
    highpass_hz: float,
    notch_hz: float,
    full_dur_sec: float = 300.0,
    out_path: str = "fig_real/fig_noncausal_cleanup.png",
) -> None:
    """Delegate the manuscript figure to the canonical process_data workflow."""
    ds_path = Path(dataset_root)
    if not ds_path.is_absolute():
        ds_path = Path(REPO_ROOT) / ds_path
    output = Path(out_path)
    if not output.is_absolute():
        output = Path(THIS_DIR) / output
    P.run_real_cancellation_figure(
        dataset_root=ds_path.resolve(),
        subject=subject,
        session=session,
        run=run,
        start_sec=start_sec,
        duration_sec=duration_sec,
        eeg_index=eeg2_index,
        ecg_index=ecg_index,
        full_duration_sec=full_dur_sec,
        out_path=output.resolve(),
    )


def _coherence_curve(x: np.ndarray, y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nseg = min(len(x), len(y), 64)
    nseg = max(32, nseg)
    f_coh, cxy = signal.coherence(x, y, fs=fs, nperseg=nseg, detrend="constant")
    return f_coh, cxy


def _extract_template_beat(r: np.ndarray, fids: list[tuple[int, str, bool]], ra: int, Lw: int, target_idx: int) -> tuple[int, np.ndarray]:
    valid = [int(fid) for fid, _, lg in fids if lg and fid - ra >= 0 and fid - ra + Lw <= len(r)]
    if not valid:
        raise RuntimeError("Could not find a fully contained learnable beat window for coherence plotting.")
    beat_idx = min(valid, key=lambda fid: abs(fid - target_idx))
    lo = beat_idx - ra
    hi = lo + Lw
    return beat_idx, r[lo:hi]


def fig3_real_coherence_view(
    dataset_root: str,
    subject: str,
    session: str,
    run: str,
    eeg_index: int,
    ecg_index: int,
    start_sec: float,
    duration_sec: float,
    highpass_hz: float,
    notch_hz: float,
    beat_target_sec: float = 10.0,
    out_path: str = "fig_real/fig_real_coherence_combined.png",
) -> None:
    """Generate real-data coherence/template figure for the paper.

    Left: one R-windowed ECG beat overlaid with non-causal, buffered, and causal EEG templates.
    Right: coherence of ECG beat against raw EEG beat and the three EEG templates.
    """
    ds_path = dataset_root
    if not os.path.isabs(ds_path):
        ds_path = os.path.join(REPO_ROOT, ds_path)
    ds_path_abs = Path(os.path.abspath(ds_path))

    t, y, r, fs, src_label = P.load_passloss_real_inputs(
        dataset_root=ds_path_abs,
        subject=subject,
        session=session,
        run=run,
        pull=True,
        eeg_index=eeg_index,
        ecg_index=ecg_index,
    )

    start = max(0.0, float(start_sec))
    end = min(t[-1], start + max(0.0, float(duration_sec)))
    if end <= start:
        raise ValueError(f"Invalid Figure 2 window start={start} end={end}")
    m = (t >= start) & (t <= end)
    t = t[m]
    y = y[m]
    r = r[m]
    y = P._mne_preprocess_trace(y, fs, highpass_hz=highpass_hz, notch_hz=notch_hz)

    est = C.phase_estimator(r, fs, use_pll=True, use_override=True, debug=True)
    _, template_non = C.ilc_clean(y, fs, est, mode="noncausal", use_ilc=True)
    _, template_buf = C.ilc_clean(y, fs, est, mode="buffered", use_ilc=True)
    # _, template_cau = C.ilc_clean(y, fs, est, mode="causal", use_ilc=True)

    ra, Lw = C._template_dims(fs)
    ts = (np.arange(Lw) - ra) / fs
    target_idx = int(round(max(0.0, float(beat_target_sec)) * fs))
    beat_idx, ref_beat = _extract_template_beat(r, est["fids"], ra, Lw, target_idx)
    lo = beat_idx - ra
    hi = lo + Lw
    raw_beat = y[lo:hi]

    f_raw, coh_raw = _coherence_curve(raw_beat, ref_beat, fs)
    f_non, coh_non = _coherence_curve(template_non, ref_beat, fs)
    f_buf, coh_buf = _coherence_curve(template_buf, ref_beat, fs)

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], wspace=0.32, hspace=0.18)
    ax_ecg = fig.add_subplot(gs[0, 0])
    ax_eeg = fig.add_subplot(gs[1, 0], sharex=ax_ecg)
    ax_coh = fig.add_subplot(gs[:, 1])

    template_non_uv = template_non * 1e6
    template_buf_uv = template_buf * 1e6
    ax_ecg.plot(ts, ref_beat, color="C3", lw=1.0, label="ECG beat")
    ax_ecg.axvline(0.0, color="0.45", lw=0.7, ls="--")
    ax_ecg.axvspan(-0.06, 0.06, color="C1", alpha=0.08)
    ax_ecg.set_title("(a) R-locked ECG beat and EEG templates")
    ax_ecg.set_ylabel("ECG (V)", color="C3")
    ax_ecg.tick_params(axis="y", colors="C3")
    ax_ecg.tick_params(axis="x", labelbottom=False)
    ax_ecg.grid(alpha=0.25)
    ax_ecg.legend(loc="upper right", fontsize=6)

    ax_eeg.plot(ts, template_non_uv, color="C0", lw=1.3, label="Non-causal EEG template")
    ax_eeg.plot(ts, template_buf_uv, color="C2", lw=1.1, label="Buffered EEG template")
    ax_eeg.axvline(0.0, color="0.45", lw=0.7, ls="--")
    ax_eeg.axvspan(-0.06, 0.06, color="C1", alpha=0.08)
    ax_eeg.set_xlabel("time since R (s)")
    ax_eeg.set_ylabel("EEG template (uV)")
    ax_eeg.set_ylim(-30.0, 50.0)
    ax_eeg.set_yticks([-30, -10, 10, 30, 50])
    ax_eeg.grid(alpha=0.25)
    ax_eeg.legend(loc="upper left", bbox_to_anchor=(0.48, 1.0), fontsize=6)

    ax_coh.plot(f_raw, coh_raw, color="0.65", lw=1.0, label="ECG beat vs raw EEG beat")
    ax_coh.plot(f_non, coh_non, color="C0", lw=1.3, label="ECG beat vs non-causal template")
    ax_coh.plot(f_buf, coh_buf, color="C2", lw=1.1, label="ECG beat vs buffered template")
    ax_coh.set_xlim(0.0, min(60.0, fs / 2.0))
    ax_coh.set_ylim(0.0, 1.05)
    ax_coh.set_title("(b) Coherence on the same R-locked window")
    ax_coh.set_xlabel("frequency (Hz)")
    ax_coh.set_ylabel("coherence")
    ax_coh.grid(alpha=0.25)
    ax_coh.legend(loc="upper left", bbox_to_anchor=(0.50, 1.0), fontsize=6)

    fig.suptitle(
        f"Real-data coherence view ({subject}/{session}/run-{run}, beat near {beat_idx / fs:.1f} s)",
        fontsize=9,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_abs = out_path if os.path.isabs(out_path) else os.path.join(THIS_DIR, out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    plt.savefig(out_abs, dpi=150)
    plt.close(fig)
    print(f"wrote Figure 3 coherence view: {out_abs}")
    print(f"  {src_label}")
    print("  beat center: %.2f s | learnable beats: %d" % (beat_idx / fs, len([1 for _, _, lg in est["fids"] if lg])))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures.")
    parser.add_argument("--dataset-root", default="datasets/ds005873", help="Dataset root path")
    parser.add_argument("--subject", default="sub-070", help="Subject ID")
    parser.add_argument("--session", default="ses-01", help="Session ID")
    parser.add_argument("--run", default="20", help="Run ID")
    parser.add_argument("--fig1-start-sec", type=float, default=340.45, help="Figure 1 start time (s)")
    parser.add_argument("--fig1-duration-sec", type=float, default=1.8, help="Figure 1 duration (s)")
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
    parser.add_argument(
        "--fig2-full-dur-sec", type=float, default=300.0,
        help="Figure 2: duration of full recording loaded from t=0 for non-causal template (s)",
    )
    parser.add_argument(
        "--fig2-out",
        default="fig_real/fig_noncausal_cleanup.png",
        help="Figure 2 output path (relative to paper/ or absolute)",
    )
    parser.add_argument("--skip-fig2-noncausal", action="store_true", help="Skip Figure 2 non-causal cleanup generation")
    parser.add_argument(
        "--fig3-beat-target-sec", type=float, default=10.1,
        help="Figure 3: target beat time (seconds within loaded window) for the coherence/template view",
    )
    parser.add_argument(
        "--fig3-out",
        default="fig_real/fig_real_coherence_combined.png",
        help="Figure 3 output path (relative to paper/ or absolute)",
    )
    parser.add_argument("--skip-fig3-coherence", action="store_true", help="Skip Figure 3 coherence/template generation")
    parser.add_argument(
        "--fig4-out",
        default="fig3_synthetic_coherence_sweep.pdf",
        help="Synthetic sweep figure output path (relative to paper/ or absolute)",
    )
    parser.add_argument("--skip-fig4-sweep", action="store_true", help="Skip synthetic coherence sweep generation")
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

    if not args.skip_fig2_noncausal:
        fig2_noncausal_cleanup(
            dataset_root=args.dataset_root,
            subject=args.subject,
            session=args.session,
            run=args.run,
            start_sec=args.fig1_start_sec,
            duration_sec=args.fig1_duration_sec,
            eeg2_index=args.fig1_eeg2_index,
            ecg_index=args.fig1_ecg_index,
            highpass_hz=args.fig1_highpass_hz,
            notch_hz=args.fig1_notch_hz,
            full_dur_sec=args.fig2_full_dur_sec,
            out_path=args.fig2_out,
        )

    if not args.skip_fig3_coherence:
        fig3_real_coherence_view(
            dataset_root=args.dataset_root,
            subject=args.subject,
            session=args.session,
            run=args.run,
            eeg_index=args.fig1_eeg2_index,
            ecg_index=args.fig1_ecg_index,
            start_sec=0.0,
            duration_sec=args.fig2_full_dur_sec,
            highpass_hz=args.fig1_highpass_hz,
            notch_hz=args.fig1_notch_hz,
            beat_target_sec=args.fig3_beat_target_sec,
            out_path=args.fig3_out,
        )

    if not args.skip_fig4_sweep:
        fig4_out = args.fig4_out
        if not os.path.isabs(fig4_out):
            fig4_out = os.path.join(THIS_DIR, fig4_out)
        fig3_synthetic_coherence_sweep(out_path=fig4_out)

    t, y, s, d, r, est, results = run_pipeline()
    fig1_convergence_spectrum(t, y, s, results, FS)
    fig2_timedomain(t, y, s, d, r, est, results, FS)
    summary_rows = compute_method_summary(t, y, s, r, results)
    write_summary_table_tex(summary_rows)
    print_summary_table(summary_rows)
    print("wrote fig3_synthetic_coherence_sweep.pdf, fig1_convergence_spectrum.pdf, fig2_timedomain.pdf, and tab_results_generated.tex")
