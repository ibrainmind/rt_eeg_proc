#!/usr/bin/env python3
"""
process_data.py

Modular loader + plotting utility for OpenNeuro ds005873 EDF runs.

Features:
- Reusable functions that other Python modules can import.
- CLI with argparse.
- Optional `datalad get` pull for a specific run file.
- Plot layout with shared time axis:
  - Top: EEG channel 1
  - Middle: EEG channel 2
  - Bottom: ECG/EKG channel(s)

Examples:
    python process_data.py --dataset-root ds005873 --subject sub-001 --session ses-01 --run 01 --duration-sec 30 --show
    python process_data.py --plot-tracker --dataset-root ds005873 --subject sub-001 --session ses-01 --run 01 --duration-sec 30 --out outputs/sub-001_run-01_tracker.png
    python process_data.py --plot-passloss --dataset-root ds005873 --subject sub-001 --session ses-01 --run 01 --duration-sec 120 --out-prefix outputs/passloss_real
    python process_data.py --plot-passloss --dataset-root ds005873 --subject sub-001 --session ses-01 --run 01 --eeg-index 1 --ecg-index 0 --band-lo 8 --band-hi 20 --show
    python process_data.py --plot-ilc-template --dataset-root ds005873 --subject sub-001 --session ses-01 --run 01 --eeg-index 1 --ecg-index 0 --duration-sec 120 --out outputs/sub-001_run-01_ilc_template.png
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal
from scipy.signal import coherence, welch

import ekg_ilc_rt as rt


@dataclass
class SignalTrace:
    """Single signal trace with sampling metadata."""

    name: str
    values: np.ndarray
    fs: float
    unit: str


@dataclass
class RunData:
    """Container for selected run data."""

    eeg: List[SignalTrace]
    ecg: List[SignalTrace]


@dataclass
class TrackerComparison:
    """Fiducial beat detections and tracker state for visualization."""

    t: np.ndarray
    ecg: np.ndarray
    emph: np.ndarray
    fs: float
    fids: np.ndarray
    fid_times: np.ndarray
    rrhat: np.ndarray
    rr_meas_times: np.ndarray
    rr_meas_values: np.ndarray
    peak: np.ndarray
    base: np.ndarray
    gate: np.ndarray
    armed: np.ndarray
    thresh_met: np.ndarray
    detect_events: np.ndarray
    rr_meas_detect_times: np.ndarray
    thr_hi: float
    thr_lo: float
    ab: float


def build_run_file_paths(
    dataset_root: Path,
    subject: str = "sub-001",
    session: str = "ses-01",
    run: str = "01",
) -> Tuple[Path, Path]:
    """Build expected EEG and ECG EDF paths for a dataset run."""
    eeg_name = f"{subject}_{session}_task-szMonitoring_run-{run}_eeg.edf"
    ecg_name = f"{subject}_{session}_task-szMonitoring_run-{run}_ecg.edf"
    eeg_path = dataset_root / subject / session / "eeg" / eeg_name
    ecg_path = dataset_root / subject / session / "ecg" / ecg_name
    return eeg_path, ecg_path


def maybe_pull_with_datalad(dataset_root: Path, run_files: Sequence[Path], pull: bool = True) -> None:
    """Ensure specific run files are present, optionally via `datalad get`."""
    missing = [p for p in run_files if not p.exists()]
    if not missing or not pull:
        return

    if not (dataset_root / ".git").exists():
        missing_text = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Run file(s) not found:\n{missing_text}\nDataset root is not a git/datalad repo: {dataset_root}"
        )

    rels = [str(p.relative_to(dataset_root)) for p in missing]
    cmd = ["datalad", "get", "-r", *rels]
    result = subprocess.run(cmd, cwd=dataset_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Failed to pull run file with datalad: {' '.join(cmd)}\n{msg}")

    still_missing = [p for p in run_files if not p.exists()]
    if still_missing:
        missing_text = "\n".join(str(p) for p in still_missing)
        raise FileNotFoundError(f"Run file(s) still missing after datalad get:\n{missing_text}")


def _non_annotation_channels(ch_names: Sequence[str]) -> List[str]:
    ann_re = re.compile(r"annotation", flags=re.IGNORECASE)
    return [ch for ch in ch_names if ch and not ann_re.search(ch)]


def load_raw_edf(edf_path: Path) -> mne.io.BaseRaw:
    """Load EDF using MNE."""
    return mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")


def select_two_eeg_channels(raw_eeg: mne.io.BaseRaw) -> List[str]:
    """Select two EEG channels from the EEG recording."""
    candidates = _non_annotation_channels(raw_eeg.ch_names)
    if len(candidates) < 2:
        raise ValueError("Could not find at least two EEG channels in EEG EDF file.")
    return candidates[:2]


def select_ecg_channels(raw_ecg: mne.io.BaseRaw) -> List[str]:
    """Select ECG/EKG channels from ECG recording; fallback to first signal channel."""
    ecg_re = re.compile(r"(ecg|ekg)", flags=re.IGNORECASE)
    candidates = _non_annotation_channels(raw_ecg.ch_names)
    named_ecg = [ch for ch in candidates if ecg_re.search(ch)]
    if named_ecg:
        return named_ecg
    if candidates:
        return [candidates[0]]
    raise ValueError("Could not find ECG/EKG channel in ECG EDF file.")


def traces_from_raw(raw: mne.io.BaseRaw, channel_names: Sequence[str], unit: str = "V") -> List[SignalTrace]:
    """Extract selected channels from an MNE Raw object as SignalTrace instances."""
    traces: List[SignalTrace] = []
    sfreq = float(raw.info["sfreq"])
    for ch_name in channel_names:
        values = raw.get_data(picks=[ch_name])[0]
        traces.append(SignalTrace(name=ch_name, values=values, fs=sfreq, unit=unit))
    return traces


def load_run_data(
    dataset_root: Path,
    subject: str = "sub-001",
    session: str = "ses-01",
    run: str = "01",
    pull: bool = True,
) -> RunData:
    """Load two EEG channels and ECG/EKG channel(s) from one run."""
    eeg_file, ecg_file = build_run_file_paths(
        dataset_root, subject=subject, session=session, run=run
    )
    maybe_pull_with_datalad(dataset_root, [eeg_file, ecg_file], pull=pull)

    raw_eeg = load_raw_edf(eeg_file)
    raw_ecg = load_raw_edf(ecg_file)

    eeg_channels = select_two_eeg_channels(raw_eeg)
    ecg_channels = select_ecg_channels(raw_ecg)

    eeg_traces = traces_from_raw(raw_eeg, eeg_channels)
    ecg_traces = traces_from_raw(raw_ecg, ecg_channels)

    return RunData(eeg=eeg_traces, ecg=ecg_traces)


def _time_axis(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=float) / fs


def _resample_linear(x: np.ndarray, fs_in: float, fs_out: float, n_out: int) -> np.ndarray:
    """Resample 1D signal with linear interpolation to a target sample count."""
    if n_out <= 1:
        return np.asarray(x[:n_out], dtype=float)
    t_in = np.arange(len(x), dtype=float) / float(fs_in)
    t_out = np.arange(n_out, dtype=float) / float(fs_out)
    return np.interp(t_out, t_in, x)


def _mne_preprocess_trace(
    x: np.ndarray,
    fs: float,
    highpass_hz: float | None,
    notch_hz: float | None,
) -> np.ndarray:
    """Apply optional MNE filtering to a 1D trace."""
    y = np.asarray(x, dtype=float)
    if highpass_hz is not None and highpass_hz > 0.0:
        y = mne.filter.filter_data(
            y,
            sfreq=fs,
            l_freq=highpass_hz,
            h_freq=None,
            method="iir",
            verbose="ERROR",
        )
    if notch_hz is not None and notch_hz > 0.0 and notch_hz < (fs / 2.0):
        y = mne.filter.notch_filter(
            y,
            Fs=fs,
            freqs=[notch_hz],
            method="iir",
            verbose="ERROR",
        )
    return np.asarray(y, dtype=float)


def load_passloss_real_inputs(
    dataset_root: Path,
    subject: str = "sub-001",
    session: str = "ses-01",
    run: str = "01",
    pull: bool = True,
    eeg_index: int = 0,
    ecg_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Load and align one EEG + one ECG trace for real-data pass-loss estimation.

    Returns (t, eeg_values, ecg_values, fs, label), aligned to ECG sampling rate.
    """
    run_data = load_run_data(
        dataset_root=dataset_root,
        subject=subject,
        session=session,
        run=run,
        pull=pull,
    )

    if not (0 <= eeg_index < len(run_data.eeg)):
        raise IndexError(f"eeg_index out of range: {eeg_index}, available={len(run_data.eeg)}")
    if not (0 <= ecg_index < len(run_data.ecg)):
        raise IndexError(f"ecg_index out of range: {ecg_index}, available={len(run_data.ecg)}")

    eeg = run_data.eeg[eeg_index]
    ecg = run_data.ecg[ecg_index]
    fs = float(ecg.fs)

    max_dur = min(len(eeg.values) / eeg.fs, len(ecg.values) / ecg.fs)
    n_ecg = max(2, int(np.floor(max_dur * ecg.fs)))
    n_out = max(2, int(np.floor(max_dur * fs)))

    ecg_vals = np.asarray(ecg.values[:n_ecg], dtype=float)
    if eeg.fs == fs:
        eeg_vals = np.asarray(eeg.values[:n_out], dtype=float)
    else:
        eeg_vals = _resample_linear(np.asarray(eeg.values, dtype=float), eeg.fs, fs, n_out)

    n = min(len(eeg_vals), len(ecg_vals))
    eeg_vals = eeg_vals[:n]
    ecg_vals = ecg_vals[:n]
    t = _time_axis(n, fs)
    label = f"{dataset_root.name} | {subject} | {session} | run-{run} | EEG={eeg.name} ECG={ecg.name}"
    return t, eeg_vals, ecg_vals, fs, label


def load_aligned_real_segment(
    dataset_root: Path,
    subject: str = "sub-001",
    session: str = "ses-01",
    run: str = "01",
    pull: bool = True,
    eeg_indices: Tuple[int, int] = (0, 1),
    ecg_index: int = 0,
    start_sec: float = 0.0,
    duration_sec: float | None = 2.5,
    highpass_hz: float | None = 0.5,
    notch_hz: float | None = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, str, str, str]:
    """Load a short, aligned real-data segment (2 EEG + 1 ECG) for paper figures.

    Returns:
        (t, eeg1, eeg2, ecg, fs, figure_label, eeg1_name, eeg2_name, ecg_name)
    where EEG traces are preprocessed by MNE high-pass/notch filters.
    """
    run_data = load_run_data(
        dataset_root=dataset_root,
        subject=subject,
        session=session,
        run=run,
        pull=pull,
    )

    e1_idx, e2_idx = int(eeg_indices[0]), int(eeg_indices[1])
    if not (0 <= e1_idx < len(run_data.eeg)):
        raise IndexError(f"eeg_indices[0] out of range: {e1_idx}, available={len(run_data.eeg)}")
    if not (0 <= e2_idx < len(run_data.eeg)):
        raise IndexError(f"eeg_indices[1] out of range: {e2_idx}, available={len(run_data.eeg)}")
    if not (0 <= ecg_index < len(run_data.ecg)):
        raise IndexError(f"ecg_index out of range: {ecg_index}, available={len(run_data.ecg)}")

    eeg1_trace = run_data.eeg[e1_idx]
    eeg2_trace = run_data.eeg[e2_idx]
    ecg_trace = run_data.ecg[ecg_index]

    fs = float(ecg_trace.fs)
    max_dur = min(
        len(eeg1_trace.values) / eeg1_trace.fs,
        len(eeg2_trace.values) / eeg2_trace.fs,
        len(ecg_trace.values) / ecg_trace.fs,
    )
    n_out = max(2, int(np.floor(max_dur * fs)))

    def _to_fs(vals: np.ndarray, fs_in: float) -> np.ndarray:
        if fs_in == fs:
            return np.asarray(vals[:n_out], dtype=float)
        return _resample_linear(np.asarray(vals, dtype=float), fs_in, fs, n_out)

    eeg1 = _to_fs(eeg1_trace.values, float(eeg1_trace.fs))
    eeg2 = _to_fs(eeg2_trace.values, float(eeg2_trace.fs))
    ecg = _to_fs(ecg_trace.values, float(ecg_trace.fs))

    # Apply visual preprocessing to EEG only (as used in the paper Figure 1 text).
    eeg1 = _mne_preprocess_trace(eeg1, fs, highpass_hz=highpass_hz, notch_hz=notch_hz)
    eeg2 = _mne_preprocess_trace(eeg2, fs, highpass_hz=highpass_hz, notch_hz=notch_hz)

    t = _time_axis(n_out, fs)
    start = max(0.0, float(start_sec))
    end = t[-1] if duration_sec is None else min(t[-1], start + max(0.0, float(duration_sec)))
    if end <= start:
        raise ValueError(f"Invalid time window start={start} end={end}")
    m = (t >= start) & (t <= end)

    figure_label = f"{dataset_root.name} | {subject} | {session} | run-{run}"
    return (
        t[m],
        eeg1[m],
        eeg2[m],
        ecg[m],
        fs,
        figure_label,
        eeg1_trace.name,
        eeg2_trace.name,
        ecg_trace.name,
    )


def run_real_passloss(
    dataset_root: Path,
    subject: str,
    session: str,
    run: str,
    pull: bool,
    eeg_index: int,
    ecg_index: int,
    band_lo: float,
    band_hi: float,
    win_sec: float,
    hop_sec: float,
    beats_per_est: int,
    beat_step: int,
    start_sec: float,
    duration_sec: float | None,
    out_prefix: str | None,
    show: bool,
) -> None:
    """Compute and plot real-data pass loss using algorithms from ekg_ilc_rt.py."""
    import ekg_ilc_rt as rt

    t, y, r, fs, src_label = load_passloss_real_inputs(
        dataset_root=dataset_root,
        subject=subject,
        session=session,
        run=run,
        pull=pull,
        eeg_index=eeg_index,
        ecg_index=ecg_index,
    )

    start = max(0.0, float(start_sec))
    end = t[-1] if duration_sec is None else min(t[-1], start + max(0.0, float(duration_sec)))
    if end <= start:
        raise ValueError(f"Invalid time window start={start} end={end}")
    m = (t >= start) & (t <= end)
    t = t[m]
    y = y[m]
    r = r[m]

    f, loss_db_f, coh_f, _ = rt.estimate_pass_loss_freq(y, r, fs)
    tc, loss_db_t, coh_t = rt.estimate_pass_loss_timevarying(
        y,
        r,
        fs,
        band_lo=band_lo,
        band_hi=band_hi,
        win_sec=win_sec,
        hop_sec=hop_sec,
    )
    est = rt.phase_estimator(r, fs, use_pll=True, use_override=True)
    tc_b, loss_db_b, coh_b, nbeats_used = rt.estimate_pass_loss_beatsync(
        y,
        r,
        fs,
        est,
        band_lo=band_lo,
        band_hi=band_hi,
        beats_per_est=beats_per_est,
        step_beats=beat_step,
    )

    bm = (f >= band_lo) & (f <= band_hi)
    print("Pass loss input: real data")
    print(f"  {src_label}")
    print("  window: %.2f-%.2f s  fs=%.3f Hz  n=%d" % (start, end, fs, len(t)))
    if np.any(bm):
        print("Pass loss summary:")
        print("  band %.1f-%.1f Hz mean pass loss: %.2f dB" % (band_lo, band_hi, np.mean(loss_db_f[bm])))
        print("  band %.1f-%.1f Hz mean coherence: %.3f" % (band_lo, band_hi, np.mean(coh_f[bm])))
    if len(loss_db_b) > 0:
        print("  Figure 3 beat-synchronous mean pass loss: %.2f dB (nbeats=%d)" % (np.mean(loss_db_b), nbeats_used))
    else:
        print("  Figure 3 beat-synchronous estimate unavailable (not enough valid beats)")

    if out_prefix:
        out_f = f"{out_prefix}_freq.png"
        out_t = f"{out_prefix}_timevary.png"
        out_b = f"{out_prefix}_figure3_beatsync.png"
    else:
        base = f"outputs/passloss_real_{subject}_{session}_run-{run}"
        out_f = None if show else f"{base}_freq.png"
        out_t = None if show else f"{base}_timevary.png"
        out_b = None if show else f"{base}_figure3_beatsync.png"

    # Inject pyplot object for ekg_ilc_rt plotting helpers.
    rt.plt = plt
    rt.plot_pass_loss_frequency(f, loss_db_f, coh_f, out=out_f, show=False)
    rt.plot_pass_loss_timevarying(tc, loss_db_t, coh_t, band_lo, band_hi, out=out_t, show=False)
    if len(tc_b) > 0:
        rt.plot_pass_loss_beatsync(tc_b, loss_db_b, coh_b, band_lo, band_hi, out=out_b, show=False)

    if show:
        rt.plot_pass_loss_frequency(f, loss_db_f, coh_f, out=None, show=False, close=False)
        rt.plot_pass_loss_timevarying(tc, loss_db_t, coh_t, band_lo, band_hi, out=None, show=False, close=False)
        if len(tc_b) > 0:
            rt.plot_pass_loss_beatsync(tc_b, loss_db_b, coh_b, band_lo, band_hi, out=None, show=False, close=False)
        plt.show()
        plt.close("all")

    if out_f:
        print(f"Saved figure: {out_f}")
    if out_t:
        print(f"Saved figure: {out_t}")
    if out_b and len(tc_b) > 0:
        print(f"Saved figure: {out_b}")


def run_real_ilc_template_view(
    dataset_root: Path,
    subject: str,
    session: str,
    run: str,
    pull: bool,
    eeg_index: int,
    ecg_index: int,
    start_sec: float,
    duration_sec: float | None,
    out_path: Path | None,
    show: bool,
    highpass_hz: float | None,
    notch_hz: float | None,
) -> None:
    """Lock beats on one real recording, build non-causal ILC template, and plot it against one ECG beat."""
    import ekg_ilc_rt as rt

    t, y, r, fs, src_label = load_passloss_real_inputs(
        dataset_root=dataset_root,
        subject=subject,
        session=session,
        run=run,
        pull=pull,
        eeg_index=eeg_index,
        ecg_index=ecg_index,
    )

    start = max(0.0, float(start_sec))
    end = t[-1] if duration_sec is None else min(t[-1], start + max(0.0, float(duration_sec)))
    if end <= start:
        raise ValueError(f"Invalid time window start={start} end={end}")

    m = (t >= start) & (t <= end)
    t = t[m]
    y = y[m]
    r = r[m]

    y = _mne_preprocess_trace(y, fs, highpass_hz=highpass_hz, notch_hz=notch_hz)

    est = rt.phase_estimator(r, fs, use_pll=True, use_override=True, debug=True)
    clean_ilc, template = rt.ilc_clean(y, fs, est, mode="noncausal", use_ilc=True)

    fids = np.array([int(fid) for fid, _, lg in est["fids"] if lg], dtype=int)
    if len(fids) == 0:
        raise RuntimeError("No learnable beats were found for non-causal ILC template extraction.")

    ra, Lw = rt._template_dims(fs)
    ts = (np.arange(Lw) - ra) / fs
    beat_idx = fids[len(fids) // 2]
    lo = beat_idx - ra
    hi = lo + Lw
    if lo < 0 or hi > len(r):
        valid = [fid for fid in fids if fid - ra >= 0 and fid - ra + Lw <= len(r)]
        if not valid:
            raise RuntimeError("Could not find a fully contained ECG beat segment for reference plotting.")
        beat_idx = valid[len(valid) // 2]
        lo = beat_idx - ra
        hi = lo + Lw

    ref_beat = r[lo:hi]
    tpl_rms = float(np.sqrt(np.mean(template**2)))
    tpl_p2p = float(np.max(template) - np.min(template))
    ref_rms = float(np.sqrt(np.mean(ref_beat**2)))
    ref_p2p = float(np.max(ref_beat) - np.min(ref_beat))

    beat_windows = []
    for fid, ty, lg in est["fids"]:
        if not lg:
            continue
        beat_lo = fid - ra
        beat_hi = beat_lo + Lw
        if beat_lo < 0 or beat_hi > len(r):
            continue
        beat_windows.append(r[beat_lo:beat_hi])
    if len(beat_windows) == 0:
        raise RuntimeError("Could not extract any peak-aligned ECG beat windows.")
    ecg_beat_avg = np.mean(np.vstack(beat_windows), axis=0)

    def _coherence_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nseg = min(len(x), len(y), 64)
        nseg = max(32, nseg)
        f_coh, cxy = coherence(x, y, fs=fs, nperseg=nseg, detrend="constant")
        return f_coh, cxy

    def _mean_band_coh(f_coh: np.ndarray, cxy: np.ndarray, lo_hz: float = 1.0, hi_hz: float = 20.0) -> float:
        band = (f_coh >= lo_hz) & (f_coh <= hi_hz)
        if not np.any(band):
            return float("nan")
        return float(np.mean(cxy[band]))

    spec_fs = fs
    spec_nperseg = min(len(template), max(64, int(round(spec_fs * 4.0))))
    f_tpl, pxx_tpl = welch(template, fs=spec_fs, nperseg=spec_nperseg, detrend="constant")
    f_ref, pxx_ref = welch(ref_beat, fs=spec_fs, nperseg=spec_nperseg, detrend="constant")
    max_spec_freq = min(60.0, spec_fs / 2.0)

    print("Non-causal ILC template view (real data)")
    print(f"  {src_label}")
    print("  window: %.2f-%.2f s  fs=%.3f Hz  n=%d" % (start, end, fs, len(t)))
    print("  template rms: %.3e V  p2p: %.3e V" % (tpl_rms, tpl_p2p))
    print("  reference ECG beat rms: %.3e V  p2p: %.3e V" % (ref_rms, ref_p2p))
    print("  learnable beats used: %d" % len(fids))
    print("  peak-aligned ECG beats used: %d" % len(beat_windows))
    if highpass_hz is not None or notch_hz is not None:
        hp_text = "none" if highpass_hz is None else f"{highpass_hz:.2f} Hz"
        notch_text = "none" if notch_hz is None else f"{notch_hz:.2f} Hz"
        print(f"  preprocessing: high-pass={hp_text} notch={notch_text}")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=False)
    axes[0].plot(ts, template, color="C0", lw=1.6, label="Non-causal ILC template (EEG artifact)")
    axes[0].axvline(0.0, color="0.5", lw=0.8, ls="--")
    axes[0].axvspan(-0.06, 0.06, color="C1", alpha=0.08, label="QRS +/-60 ms")
    axes[0].set_ylabel("Template amplitude (V)")
    axes[0].set_title("Non-causal ILC average template vs ECG beat phase reference")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(ts, ref_beat, color="C3", lw=1.2, label="One ECG beat (phase reference)")
    axes[1].axvline(0.0, color="0.5", lw=0.8, ls="--")
    axes[1].axvspan(-0.06, 0.06, color="C1", alpha=0.08)
    axes[1].set_ylabel("ECG amplitude (V)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].semilogy(f_tpl[f_tpl <= max_spec_freq], pxx_tpl[f_tpl <= max_spec_freq], color="C0", lw=1.5, label="Template spectrum")
    axes[2].semilogy(f_ref[f_ref <= max_spec_freq], pxx_ref[f_ref <= max_spec_freq], color="C3", lw=1.2, alpha=0.9, label="ECG beat spectrum")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("PSD (V^2/Hz)")
    axes[2].set_xlim(0.0, max_spec_freq)
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=130)
        print(f"Saved figure: {out_path}")

    # Figure 3: peak-aligned ECG average and comparison against the EEG template.
    fig3, axes3 = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True)
    for beat in beat_windows[: min(12, len(beat_windows))]:
        axes3[0].plot(ts, beat, color="0.7", lw=0.55, alpha=0.35)
    axes3[0].plot(ts, ecg_beat_avg, color="C3", lw=1.5, label="Peak-aligned ECG average")
    axes3[0].axvline(0.0, color="0.5", lw=0.8, ls="--")
    axes3[0].axvspan(-0.06, 0.06, color="C1", alpha=0.08)
    axes3[0].set_ylabel("ECG amplitude (V)")
    axes3[0].set_title("Peak-aligned ECG averaging")
    axes3[0].grid(alpha=0.3)
    axes3[0].legend(loc="upper right", fontsize=8)

    axes3[1].plot(ts, template, color="C0", lw=1.5, label="EEG non-causal template")
    axes3[1].plot(ts, ecg_beat_avg, color="C3", lw=1.2, alpha=0.9, label="Peak-aligned ECG average")
    axes3[1].axvline(0.0, color="0.5", lw=0.8, ls="--")
    axes3[1].axvspan(-0.06, 0.06, color="C1", alpha=0.08)
    axes3[1].set_xlabel("Time around R (s)")
    axes3[1].set_ylabel("Amplitude (V)")
    axes3[1].grid(alpha=0.3)
    axes3[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if out_path is not None:
        out3 = out_path.with_name(f"{out_path.stem}_figure3_ecg_avg{out_path.suffix}")
        plt.savefig(out3, dpi=130)
        print(f"Saved figure: {out3}")

    # Figure 4: one-beat coherence between EEG/template and ECG.
    f_raw_coh, coh_raw = _coherence_curve(y[lo:hi], ref_beat)
    f_tpl_coh, coh_tpl = _coherence_curve(template, ref_beat)
    fig4, ax4 = plt.subplots(figsize=(10.5, 4.4))
    ax4.plot(f_raw_coh, coh_raw, color="0.7", lw=1.0, label="Raw EEG vs ECG beat")
    ax4.plot(f_tpl_coh, coh_tpl, color="C0", lw=1.6, label="Template vs ECG beat")
    ax4.set_xlim(0.0, min(60.0, fs / 2.0))
    ax4.set_ylim(0.0, 1.05)
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Coherence")
    ax4.set_title("One-beat coherence: template vs ECG")
    ax4.grid(alpha=0.3)
    ax4.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if out_path is not None:
        out4 = out_path.with_name(f"{out_path.stem}_figure4_coherence{out_path.suffix}")
        plt.savefig(out4, dpi=130)
        print(f"Saved figure: {out4}")

    # Figure 5: causal and buffered templates in the time domain.
    clean_buffered, template_buffered = rt.ilc_clean(y, fs, est, mode="buffered", use_ilc=True)
    clean_causal, template_causal = rt.ilc_clean(y, fs, est, mode="causal", use_ilc=True)
    f_buf_coh, coh_buf = _coherence_curve(template_buffered, ref_beat)
    f_cau_coh, coh_cau = _coherence_curve(template_causal, ref_beat)
    f_non_coh, coh_non = _coherence_curve(template, ref_beat)

    fig5, ax5 = plt.subplots(figsize=(10.5, 4.8))
    ax5.plot(ts, ref_beat, color="C3", lw=1.1, alpha=0.85, label="ECG beat")
    ax5.plot(ts, template, color="C0", lw=1.4, label="Non-causal template")
    ax5.plot(ts, template_buffered, color="C2", lw=1.2, label="Buffered template")
    ax5.plot(ts, template_causal, color="C4", lw=1.2, label="Causal template")
    ax5.axvline(0.0, color="0.5", lw=0.8, ls="--")
    ax5.axvspan(-0.06, 0.06, color="C1", alpha=0.08)
    ax5.set_xlabel("Time around R (s)")
    ax5.set_ylabel("Amplitude (V)")
    ax5.set_title("Causal and buffered templates in the time domain")
    ax5.grid(alpha=0.3)
    ax5.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if out_path is not None:
        out5 = out_path.with_name(f"{out_path.stem}_figure5_causal_buffered_time{out_path.suffix}")
        plt.savefig(out5, dpi=130)
        print(f"Saved figure: {out5}")

    # Figure 6: coherence-only comparison on its own axis.
    fig6, ax6 = plt.subplots(figsize=(10.5, 4.4))
    ax6.plot(f_non_coh, coh_non, color="C0", lw=1.4, label="Non-causal template vs ECG")
    ax6.plot(f_buf_coh, coh_buf, color="C2", lw=1.2, label="Buffered template vs ECG")
    ax6.plot(f_cau_coh, coh_cau, color="C4", lw=1.2, label="Causal template vs ECG")
    ax6.set_xlim(0.0, min(60.0, fs / 2.0))
    ax6.set_ylim(0.0, 1.05)
    ax6.set_xlabel("Frequency (Hz)")
    ax6.set_ylabel("Coherence")
    ax6.set_title("Template coherence against ECG beat")
    ax6.grid(alpha=0.3)
    ax6.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if out_path is not None:
        out6 = out_path.with_name(f"{out_path.stem}_figure6_causal_buffered_coherence{out_path.suffix}")
        plt.savefig(out6, dpi=130)
        print(f"Saved figure: {out6}")

    mean_raw_coh = _mean_band_coh(f_raw_coh, coh_raw)
    mean_tpl_coh = _mean_band_coh(f_tpl_coh, coh_tpl)
    mean_buf_coh = _mean_band_coh(f_buf_coh, coh_buf)
    mean_cau_coh = _mean_band_coh(f_cau_coh, coh_cau)
    print("  one-beat mean coherence 1-20 Hz:")
    print("    raw EEG vs ECG: %.3f" % mean_raw_coh)
    print("    template vs ECG: %.3f" % mean_tpl_coh)
    print("    buffered template vs ECG: %.3f" % mean_buf_coh)
    print("    causal template vs ECG: %.3f" % mean_cau_coh)

    # Figure 2: ECG on top, EEG before/after cleanup in the middle, residual error at the bottom.
    fig2, axes = plt.subplots(3, 1, figsize=(11.2, 8.2), sharex=True)
    axes[0].plot(t, r, color="C3", lw=0.9, label="ECG")
    axes[0].set_ylabel("ECG amplitude (V)")
    axes[0].set_title("Aligned ECG and EEG cleanup over the selected duration")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, y, color="C0", lw=1.0, label="Original EEG")
    axes[1].plot(t, clean_ilc, color="C2", lw=1.0, label="Non-causal cleaned EEG")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("EEG amplitude (V)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    residual = y - clean_ilc
    axes[2].plot(t, residual, color="C4", lw=0.9, label="Residual: original EEG - cleaned EEG")
    axes[2].axhline(0.0, color="0.5", lw=0.8, ls="--")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Residual (V)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8)

    if start > 0.0 or end < t[-1]:
        axes[0].set_xlim(start, end)
        axes[1].set_xlim(start, end)
        axes[2].set_xlim(start, end)

    if figure_label := f"{dataset_root.name} | {subject} | {session} | run-{run}":
        fig2.suptitle(f"{figure_label} | ECG reference and EEG cleanup detail", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if out_path is not None:
        out2 = out_path.with_name(f"{out_path.stem}_comparison{out_path.suffix}")
        plt.savefig(out2, dpi=130)
        print(f"Saved figure: {out2}")

    if show:
        plt.show()
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
    plt.close(fig)
    plt.close(fig2)


def run_pi_tracker_comparison(ecg_trace: SignalTrace) -> TrackerComparison:
    """Run PI/phase tracker on ECG and return fiducial diagnostics."""
    from ekg_ilc_rt import phase_estimator

    ecg = ecg_trace.values
    fs = ecg_trace.fs
    t = _time_axis(len(ecg), fs)

    rr0 = 0.85
    est = phase_estimator(ecg, fs, use_pll=True, use_override=True, rr0=rr0, debug=True)
    fids = np.array([int(fid) for fid, _, _ in est["fids"]], dtype=int)
    fid_times = fids / fs

    rr_meas_idx = np.where(~np.isnan(est["rr_meas"]))[0]
    rr_meas_times = rr_meas_idx / fs
    rr_meas_values = est["rr_meas"][rr_meas_idx]

    detect_idx = np.where(est["detect_events"] > 0)[0]
    rr_meas_detect_times = detect_idx / fs
    params = est.get("params", {})
    thr_hi = float(params.get("THR_HI", 0.40))
    thr_lo = float(params.get("THR_LO", 0.15))
    ab = float(params.get("AB", 1.0 / (1.5 * fs)))

    return TrackerComparison(
        t=t,
        ecg=ecg,
        emph=est["emph"],
        fs=fs,
        fids=fids,
        fid_times=fid_times,
        rrhat=est["rrhat"],
        rr_meas_times=rr_meas_times,
        rr_meas_values=rr_meas_values,
        peak=est["peak"],
        base=est["base"],
        gate=est["gate"],
        armed=est["armed"],
        thresh_met=est["thresh_met"],
        detect_events=est["detect_events"],
        rr_meas_detect_times=rr_meas_detect_times,
        thr_hi=thr_hi,
        thr_lo=thr_lo,
        ab=ab,
    )


def plot_tracker_comparison(
    cmp: TrackerComparison,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    out_path: Path | None = None,
    figure_label: str | None = None,
    show: bool = False,
) -> None:
    """Plot ECG + tracker internals with linked time axis."""
    t = cmp.t
    fs = cmp.fs
    start = max(0.0, start_sec)
    end = t[-1] if duration_sec is None else min(t[-1], start + max(0.0, duration_sec))
    if end <= start:
        raise ValueError(f"Invalid time window start={start} end={end}")

    mask = (t >= start) & (t <= end)
    t_win = t[mask]
    ecg_win = cmp.ecg[mask]
    emph_win = cmp.emph[mask]

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)

    ax = axes[0]
    ax.plot(t_win, ecg_win, color="0.25", lw=0.7, label="ECG")
    keep = (cmp.fids / fs >= start) & (cmp.fids / fs <= end)
    sel = cmp.fids[keep]
    if len(sel) > 0:
        ax.scatter(sel / fs, cmp.ecg[sel], s=24, c="C0", marker="o", label="fids (beat detections)", alpha=0.9)
    ax.set_title("EKG with fid markers")
    ax.set_ylabel("Amplitude (V)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(start, end)

    ax = axes[1]
    ax.plot(t_win, emph_win, color="0.6", lw=0.6, label="filtered ECG (emph)")
    ax.plot(t_win, cmp.peak[mask], color="C6", lw=0.7, ls="--", alpha=0.9, label="peak")
    ax.plot(t_win, cmp.base[mask], color="C2", lw=0.7, ls=":", alpha=0.95, label="base")
    ax2 = ax.twinx()
    gate_win = cmp.gate[mask]
    ax2.plot(t_win, gate_win, color="C1", lw=1.2, label="gate")
    ax2.axhline(cmp.thr_hi, color="C3", lw=0.9, ls="--", label=f"THR_HI={cmp.thr_hi:.2f}")
    ax2.axhline(cmp.thr_lo, color="C5", lw=0.9, ls=":", label=f"THR_LO={cmp.thr_lo:.2f}")
    keep = (cmp.fids / fs >= start) & (cmp.fids / fs <= end)
    sel = cmp.fids[keep]
    if sel.size > 0:
        ax.scatter(
            sel / fs,
            cmp.emph[sel],
            s=18,
            c="C0",
            marker="o",
            alpha=0.8,
            label="fids",
            zorder=4,
        )
    thresh_win_idx = np.where(cmp.thresh_met[mask] > 0)[0]
    if thresh_win_idx.size > 0:
        ax2.scatter(
            t_win[thresh_win_idx],
            gate_win[thresh_win_idx],
            s=16,
            c="C3",
            marker="o",
            alpha=0.75,
            label="threshold-met samples",
            zorder=4,
        )
    armed_on = cmp.armed[mask] > 0
    if np.any(armed_on):
        ax.fill_between(t_win, emph_win.min(), emph_win.max(), where=armed_on, color="C0", alpha=0.08, label="armed")
    thr_times = t[np.where(cmp.thresh_met > 0)]
    thr_times = thr_times[(thr_times >= start) & (thr_times <= end)]
    if len(thr_times) > 0:
        for tt in thr_times:
            ax.axvline(tt, color="C3", lw=0.6, alpha=0.3)
    ab_tau_sec = 1.0 / (cmp.ab * cmp.fs)
    ax.set_title(f"Filtered ECG (emph) with normalized gate and thresholds (AB tau~{ab_tau_sec:.2f}s)")
    ax.set_ylabel("Filtered ECG (V)")
    ax2.set_ylabel("Normalized amplitude")
    gate_max = float(np.nanmax(gate_win)) if gate_win.size else 1.0
    ax2.set_ylim(-0.05, max(1.1, gate_max + 0.1))
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if thr_times.size > 0:
        from matplotlib.lines import Line2D
        lines1.append(Line2D([0], [0], color="C3", lw=1.0, alpha=0.6))
        labels1.append("threshold met")
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")

    if figure_label:
        fig.suptitle(f"{figure_label} | PI tracker fids", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120)
        print(f"Saved figure: {out_path}")

    # Figure 2: only the bottom tracker panel over the same time window,
    # with independent autoscaling so details remain visible when zoomed.
    fig2, axb = plt.subplots(1, 1, figsize=(12, 4.6))
    axb.plot(t_win, emph_win, color="0.6", lw=0.6, label="filtered ECG (emph)")
    axb.plot(t_win, cmp.peak[mask], color="C6", lw=0.7, ls="--", alpha=0.9, label="peak")
    axb.plot(t_win, cmp.base[mask], color="C2", lw=0.7, ls=":", alpha=0.95, label="base")
    axb2 = axb.twinx()
    axb2.plot(t_win, gate_win, color="C1", lw=1.2, label="gate")
    axb2.axhline(cmp.thr_hi, color="C3", lw=0.9, ls="--", label=f"THR_HI={cmp.thr_hi:.2f}")
    axb2.axhline(cmp.thr_lo, color="C5", lw=0.9, ls=":", label=f"THR_LO={cmp.thr_lo:.2f}")

    keep = (cmp.fids / fs >= start) & (cmp.fids / fs <= end)
    sel = cmp.fids[keep]
    if sel.size > 0:
        axb.scatter(
            sel / fs,
            cmp.emph[sel],
            s=18,
            c="C0",
            marker="o",
            alpha=0.8,
            label="fids",
            zorder=4,
        )

    thresh_win_idx = np.where(cmp.thresh_met[mask] > 0)[0]
    if thresh_win_idx.size > 0:
        axb2.scatter(
            t_win[thresh_win_idx],
            gate_win[thresh_win_idx],
            s=16,
            c="C3",
            marker="o",
            alpha=0.75,
            label="threshold-met samples",
            zorder=4,
        )

    armed_on = cmp.armed[mask] > 0
    if np.any(armed_on):
        axb.fill_between(t_win, emph_win.min(), emph_win.max(), where=armed_on, color="C0", alpha=0.08, label="armed")

    thr_times = t[np.where(cmp.thresh_met > 0)]
    thr_times = thr_times[(thr_times >= start) & (thr_times <= end)]
    if len(thr_times) > 0:
        for tt in thr_times:
            axb.axvline(tt, color="C3", lw=0.6, alpha=0.3)

    axb.set_title("Tracker bottom panel (filtered ECG emph, zoom window)")
    axb.set_ylabel("Filtered ECG (V)")
    axb2.set_ylabel("Normalized amplitude")
    axb.set_xlabel("Time (s)")
    axb.set_xlim(start, end)
    axb.grid(alpha=0.3)
    # Fixed limits for Figure 2: left in ECG volts (-2 to 10 mV), right normalized gate axis.
    axb.set_ylim(-2e-3, 10e-3)
    axb2.set_ylim(-3.0, 3.0)

    lines1, labels1 = axb.get_legend_handles_labels()
    lines2, labels2 = axb2.get_legend_handles_labels()
    if thr_times.size > 0:
        from matplotlib.lines import Line2D
        lines1.append(Line2D([0], [0], color="C3", lw=1.0, alpha=0.6))
        labels1.append("threshold met")
    axb2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    if out_path is not None:
        out2 = out_path.with_name(f"{out_path.stem}_figure2_bottom{out_path.suffix}")
        plt.savefig(out2, dpi=140)
        print(f"Saved figure: {out2}")

    if show:
        # Live-link Figure 2 x-axis to Figure 1 zoom/pan so the focused interval
        # stays synchronized while interacting with Figure 1.
        def _sync_xlim(changed_ax):
            x0, x1 = changed_ax.get_xlim()
            axb.set_xlim(x0, x1)
            fig2.canvas.draw_idle()

        for src_ax in axes:
            src_ax.callbacks.connect("xlim_changed", _sync_xlim)

    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)


def _noncausal_locked_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Find an offline zero-phase R-peak reference for predictor scoring."""
    sos = signal.butter(2, [8.0, 20.0], btype="band", fs=fs, output="sos")
    envelope = np.abs(signal.sosfiltfilt(sos, ecg))
    smooth_len = max(3, int(0.05 * fs) | 1)
    envelope = signal.savgol_filter(envelope, smooth_len, 2)
    threshold = np.median(envelope) + 0.5 * (np.percentile(envelope, 98) - np.median(envelope))
    peaks, _ = signal.find_peaks(envelope, height=threshold, distance=int(0.30 * fs))
    refine = int(round(0.06 * fs))
    locked = []
    for peak in peaks:
        lo = max(0, peak - refine)
        hi = min(len(ecg), peak + refine + 1)
        local = ecg[lo:hi] - np.mean(ecg[lo:hi])
        locked.append(lo + int(np.argmax(np.abs(local))))
    return np.asarray(locked, dtype=float)


def _nearest_reference(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the nearest offline reference sample for each candidate sample."""
    idx = np.searchsorted(reference, samples)
    lo = np.clip(idx - 1, 0, len(reference) - 1)
    hi = np.clip(idx, 0, len(reference) - 1)
    return np.where(
        np.abs(samples - reference[hi]) < np.abs(samples - reference[lo]),
        reference[hi],
        reference[lo],
    )


def run_kalman_predictor_comparison(
    ecg_trace: SignalTrace,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    out_path: Path | None = None,
    figure_label: str | None = None,
    show: bool = False,
) -> None:
    """Compare causal Kalman R predictions with an offline locked R reference."""
    ecg = np.asarray(ecg_trace.values, dtype=float)
    fs = float(ecg_trace.fs)
    if duration_sec is not None:
        ecg = ecg[: max(2, int(round(duration_sec * fs)))]
    estimate = rt.phase_estimator(ecg, fs, use_pll=False, use_override=True, debug=False)
    observed = np.asarray([fid for fid, _, _ in estimate["fids"]], dtype=float)
    reference = _noncausal_locked_rpeaks(ecg, fs)
    if observed.size < 5 or reference.size < 5:
        raise RuntimeError("Not enough R-peaks for Kalman predictor comparison")

    q, r = rt.estimate_kalman_qr(observed, reference, fs)
    kalman = rt.kalman_predict_rpeaks(observed, fs, q, r)
    prediction = kalman["predicted"]
    sigma_tau = kalman["sigma_tau"]
    valid = np.isfinite(prediction)
    matched = _nearest_reference(prediction[valid], reference)
    error_ms = (prediction[valid] - matched) / fs * 1000.0
    sigma_ms = sigma_tau[valid] / fs * 1000.0
    # Reject gross sequence mismatches while retaining ordinary predictor errors.
    scored = np.abs(error_ms) <= 0.45 * 1000.0 * np.median(np.diff(reference)) / fs
    error_ms = error_ms[scored]
    predicted = prediction[valid][scored]
    matched = matched[scored]
    sigma_ms = sigma_ms[scored]
    innovation_z = kalman["innovations"][valid][scored]
    if error_ms.size < 3:
        raise RuntimeError("Too few matched predicted/reference beats to score")

    median_error = float(np.median(error_ms))
    prior_variance = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
    metrics = {
        "q_ms2": q * 1e6,
        "r_ms2": r * 1e6,
        "K_ss": prior_variance / (prior_variance + r),
        "n_observed": int(observed.size),
        "n_reference": int(reference.size),
        "n_scored": int(error_ms.size),
        "bias_ms": float(np.mean(error_ms)),
        "median_error_ms": median_error,
        "mae_ms": float(np.mean(np.abs(error_ms))),
        "rmse_ms": float(np.sqrt(np.mean(error_ms ** 2))),
        "sd_debiased_ms": float(np.std(error_ms - median_error)),
        "abs_p90_ms": float(np.percentile(np.abs(error_ms), 90)),
        "within_5ms": float(np.mean(np.abs(error_ms) <= 5.0)),
        "within_10ms": float(np.mean(np.abs(error_ms) <= 10.0)),
    }
    print("Kalman R-peak predictor metrics")
    print(f"  q = {metrics['q_ms2']:.3f} ms^2, r = {metrics['r_ms2']:.3f} ms^2, K_ss = {metrics['K_ss']:.4f}")
    print(f"  observed/reference/scored = {metrics['n_observed']}/{metrics['n_reference']}/{metrics['n_scored']}")
    print(f"  bias={metrics['bias_ms']:.2f} ms, median={metrics['median_error_ms']:.2f} ms, "
          f"MAE={metrics['mae_ms']:.2f} ms, RMSE={metrics['rmse_ms']:.2f} ms")
    print(f"  debiased SD={metrics['sd_debiased_ms']:.2f} ms, abs P90={metrics['abs_p90_ms']:.2f} ms, "
          f"within 5/10 ms={100*metrics['within_5ms']:.1f}%/{100*metrics['within_10ms']:.1f}%")
    innovation_score = np.abs(innovation_z)
    print(f"  normalized innovation |z|: median={np.median(innovation_score):.2f}, "
          f"P90={np.percentile(innovation_score, 90):.2f}, max={np.max(innovation_score):.2f}")
    print("  Innovation gate against observed prediction error:")
    for error_threshold_ms in (10.0, 20.0, 30.0):
        bad = np.abs(error_ms) > error_threshold_ms
        if not np.any(bad) or np.all(bad):
            print(f"    bad = |error| > {error_threshold_ms:.0f} ms: insufficient class variation")
            continue
        ranks = np.argsort(np.argsort(innovation_score))
        n_bad = int(np.sum(bad))
        auc = float((np.sum(ranks[bad]) - n_bad * (n_bad - 1) / 2.0)
                / (np.sum(~bad) * n_bad))
        print(f"    bad = |error| > {error_threshold_ms:.0f} ms: prevalence={100*np.mean(bad):.1f}%, AUC={auc:.3f}")
    for z_threshold in (1.5, 2.0, 2.5, 3.0):
        flagged = innovation_score > z_threshold
        bad = np.abs(error_ms) > 20.0
        true_positive = np.sum(flagged & bad)
        precision = true_positive / max(np.sum(flagged), 1)
        recall = true_positive / max(np.sum(bad), 1)
        print(f"    |z| > {z_threshold:.1f}: flag={100*np.mean(flagged):.1f}%, "
              f"precision={100*precision:.1f}%, recall={100*recall:.1f}% for |error|>20 ms")
    print(f"  Section 6 sigma_tau range = {np.min(sigma_ms):.2f}..{np.max(sigma_ms):.2f} ms "
          f"(median {np.median(sigma_ms):.2f} ms)")
    print("  Gate sweep (sigma_tau <= threshold; rejected beats would use buffered/freeze):")
    for threshold_ms in (5.0, 10.0, 15.0, 20.0, 30.0):
        keep = sigma_ms <= threshold_ms
        if np.any(keep):
            gated_error = error_ms[keep]
            gate_rmse = float(np.sqrt(np.mean(gated_error ** 2)))
            gate_p90 = float(np.percentile(np.abs(gated_error), 90))
            print(f"    {threshold_ms:4.0f} ms: keep {100*np.mean(keep):5.1f}% "
                  f"RMSE {gate_rmse:6.2f} ms, abs P90 {gate_p90:6.2f} ms")
        else:
            print(f"    {threshold_ms:4.0f} ms: keep   0.0% (all beats gated)")

    t = _time_axis(len(ecg), fs)
    start = max(0.0, start_sec)
    end = t[-1] if duration_sec is None else min(t[-1], start + max(0.0, duration_sec))
    mask = (t >= start) & (t <= end)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
    axes[0].plot(t[mask], ecg[mask], color="0.25", lw=0.7, label="ECG")
    ref_keep = (matched / fs >= start) & (matched / fs <= end)
    pred_keep = (predicted / fs >= start) & (predicted / fs <= end)
    axes[0].scatter(matched[ref_keep] / fs, ecg[np.rint(matched[ref_keep]).astype(int)],
                    s=28, color="C0", marker="o", label="non-causal locked R")
    axes[0].scatter(predicted[pred_keep] / fs, ecg[np.clip(np.rint(predicted[pred_keep]).astype(int), 0, len(ecg)-1)],
                    s=28, color="C3", marker="x", label="Kalman predicted R")
    axes[0].set_ylabel("ECG amplitude")
    axes[0].set_title("ECG with predicted and non-causal locked R-peaks")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    error_keep = (matched / fs >= start) & (matched / fs <= end)
    axes[1].axhline(0.0, color="0.2", lw=0.8)
    axes[1].plot(matched[error_keep] / fs, error_ms[error_keep], "o-", ms=3, lw=0.8, color="C4")
    axes[1].set_ylabel("Prediction error (ms)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Kalman predicted R minus non-causal locked R")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(start, end)
    if figure_label:
        fig.suptitle(f"{figure_label} | scalar Kalman R-peak predictor", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140)
        print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def _legend_under_traces(ax, ncol: int, pad: float = 0.03) -> None:
    """Park the legend in an empty strip added below the traces.

    The traces span the full width of these panels, so the only place a legend
    cannot cover a curve is a band reserved just above the x-axis. The band is
    measured from the rendered legend so the traces are compressed no more than
    the legend actually needs.
    """
    leg = ax.legend(loc="lower center", ncol=ncol, framealpha=0.85,
                    borderaxespad=0.25, columnspacing=1.2, handletextpad=0.5)
    ax.figure.canvas.draw()
    height = leg.get_window_extent().transformed(ax.transAxes.inverted()).height
    band = min(0.5, height + pad)
    lo, hi = ax.get_ylim()
    ticks = ax.get_yticks()
    ax.set_ylim(lo - band * (hi - lo) / (1.0 - band), hi)
    # Keep the tick labels the traces had before the band was added.
    new_lo, new_hi = ax.get_ylim()
    ax.set_yticks([tk for tk in ticks if new_lo <= tk <= new_hi])


def run_real_cancellation_figure(
    dataset_root: Path,
    subject: str,
    session: str,
    run: str,
    start_sec: float,
    duration_sec: float,
    eeg_index: int = 1,
    ecg_index: int = 0,
    full_duration_sec: float = 600.0,
    out_path: Path | None = None,
    show: bool = False,
) -> None:
    """Plot original, buffered, and Kalman strict-real-time cancellation.

    q/r are estimated once from the full filtered recording; only the display
    window is cropped. This is the canonical real-data cancellation plot used
    by both the command line and manuscript figure generation.
    """
    t_full, _, eeg, ecg, fs, _, _, eeg_name, ecg_name = load_aligned_real_segment(
        dataset_root=dataset_root,
        subject=subject,
        session=session,
        run=run,
        pull=True,
        eeg_indices=(0, eeg_index),
        ecg_index=ecg_index,
        start_sec=0.0,
        duration_sec=full_duration_sec,
        highpass_hz=0.5,
        notch_hz=50.0,
    )
    detected = rt.phase_estimator(ecg, fs, use_pll=False, use_override=True)
    observed = np.asarray([fid for fid, _, _ in detected["fids"]], dtype=float)
    q, r = rt.estimate_kalman_qr(observed, fs=fs)
    estimate = rt.phase_estimator(
        ecg, fs, use_pll=True, use_override=True, kalman_q=q, kalman_r=r
    )
    buffered, _ = rt.ilc_clean(eeg, fs, estimate, mode="buffered", use_ilc=True)
    causal, _ = rt.ilc_clean(eeg, fs, estimate, mode="causal", use_ilc=True)

    i0 = max(0, min(int(round(start_sec * fs)), len(t_full) - 1))
    i1 = max(i0 + 1, min(int(round((start_sec + duration_sec) * fs)), len(t_full)))
    t = t_full[i0:i1]
    ecg_view = ecg[i0:i1]
    fid_idx = np.asarray([int(fid) for fid, _, _ in estimate["fids"]], dtype=int)
    fid_idx = fid_idx[(fid_idx >= i0) & (fid_idx < i1)]
    buffered_times = fid_idx / fs
    predicted_idx = []
    for fid in fid_idx:
        phase = ((float(estimate["phi"][fid]) + 0.5) % 1.0) - 0.5
        predicted_idx.append(float(fid) - phase * float(estimate["rrhat"][fid]) * fs)
    predicted_idx = np.asarray(predicted_idx, dtype=float)
    predicted_idx = predicted_idx[(predicted_idx >= i0) & (predicted_idx < i1)]
    predicted_times = predicted_idx / fs

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})
    axes[0].plot(t, ecg_view, lw=0.9, color="C2", label="ECG")
    if buffered_times.size:
        axes[0].scatter(buffered_times, ecg[fid_idx], s=34, color="C0", marker="o",
                        zorder=4, label="buffered R peak")
    if predicted_times.size:
        axes[0].scatter(predicted_times, np.interp(predicted_times, t, ecg_view),
                        s=42, color="C4", marker="x", zorder=5,
                        label="predicted R peak")
    axes[0].set_ylabel("V")
    axes[0].set_title("ECG/EKG: buffered and predicted R-peaks")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, eeg[i0:i1], lw=0.8, color="C3", alpha=0.75, label="original")
    axes[1].plot(t, buffered[i0:i1], lw=1.0, color="C0", label="buffered")
    axes[1].plot(t, causal[i0:i1], lw=1.0, color="C4", label="strict real-time")
    axes[1].set_ylabel("V")
    axes[1].set_title(f"EEG channel: {eeg_name}")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        for peak_time in buffered_times:
            ax.axvspan(peak_time - 0.06, peak_time + 0.06, color="C1", alpha=0.10, lw=0)
            ax.axvline(peak_time, color="0.45", lw=0.7, ls="--", alpha=0.55)
    fig.suptitle("Buffered and Strict Real-Time CFA Cancellation (SeizeIT2 ds005873)",
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    _legend_under_traces(axes[0], ncol=3)
    _legend_under_traces(axes[1], ncol=3)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved cancellation figure: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_run_data(
    run_data: RunData,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    out_path: Path | None = None,
    figure_label: str | None = None,
    show: bool = False,
) -> None:
    """Plot two EEG traces on top and ECG/EKG trace(s) on bottom with aligned time axes."""
    eeg1, eeg2 = run_data.eeg[0], run_data.eeg[1]

    all_durations = [len(eeg1.values) / eeg1.fs, len(eeg2.values) / eeg2.fs]
    all_durations.extend(len(x.values) / x.fs for x in run_data.ecg)
    max_end = min(all_durations)

    start = max(0.0, start_sec)
    end = max_end if duration_sec is None else min(max_end, start + max(0.0, duration_sec))
    if end <= start:
        raise ValueError(f"Invalid time window start={start} end={end}")

    fig_psd, ax_psd = plt.subplots(figsize=(11, 4.4))
    traces = [
        (eeg1, f"EEG 1: {eeg1.name}", "C0"),
        (eeg2, f"EEG 2: {eeg2.name}", "C1"),
    ]
    if run_data.ecg:
        traces.extend(
            (trace, f"ECG/EKG: {trace.name}", f"C{2 + idx}")
            for idx, trace in enumerate(run_data.ecg)
        )

    for trace, label, color in traces:
        t = _time_axis(len(trace.values), trace.fs)
        mask = (t >= start) & (t <= end)
        x = np.asarray(trace.values[mask], dtype=float)
        if x.size < 2:
            continue
        fs = float(trace.fs)
        nperseg = min(x.size, max(64, int(round(fs * 4.0))))
        if nperseg < 2:
            continue
        f, pxx = welch(x, fs=fs, nperseg=nperseg, detrend="constant")
        keep = f <= min(60.0, fs / 2.0)
        if np.any(keep):
            ax_psd.semilogy(f[keep], pxx[keep], lw=1.0, color=color, label=label)

    max_plot_freq = min([60.0, eeg1.fs / 2.0, eeg2.fs / 2.0] + [trace.fs / 2.0 for trace in run_data.ecg])
    ax_psd.set_xlim(0.0, max_plot_freq)
    ax_psd.set_title("Welch power spectral density of selected window")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (V^2/Hz)")
    ax_psd.grid(alpha=0.3, which="both")
    ax_psd.legend(loc="upper right", fontsize=8)

    if figure_label:
        fig_psd.suptitle(figure_label + " | Welch spectrum", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_psd = out_path.with_name(f"{out_path.stem}_spectrum{out_path.suffix}")
        plt.savefig(out_psd, dpi=120)
        print(f"Saved figure: {out_psd}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    for ax, trace, title, color in [
        (axes[0], eeg1, f"EEG 1: {eeg1.name}", "C0"),
        (axes[1], eeg2, f"EEG 2: {eeg2.name}", "C1"),
    ]:
        t = _time_axis(len(trace.values), trace.fs)
        mask = (t >= start) & (t <= end)
        ax.plot(t[mask], trace.values[mask], lw=0.7, color=color)
        ax.set_ylabel(trace.unit or "uV")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    ecg_ax = axes[2]
    for idx, trace in enumerate(run_data.ecg):
        t = _time_axis(len(trace.values), trace.fs)
        mask = (t >= start) & (t <= end)
        ecg_ax.plot(t[mask], trace.values[mask], lw=0.8, label=trace.name)

    ecg_ax.set_ylabel(run_data.ecg[0].unit or "uV")
    ecg_ax.set_title("ECG/EKG")
    ecg_ax.set_xlabel("Time (s)")
    ecg_ax.grid(alpha=0.3)
    ecg_ax.legend(loc="upper right", fontsize=8)
    ecg_ax.set_xlim(start, end)

    if figure_label:
        fig.suptitle(figure_label, fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120)
        print(f"Saved figure: {out_path}")
    if show:
        plt.figure(fig_psd.number)
        plt.show()
    plt.close(fig_psd)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load ds005873 sub-001 run-01 EEG/ECG and plot aligned subplots."
    )
    parser.add_argument(
        "--dataset-root",
        default="ds005873",
        help="Dataset root path. Relative paths are resolved under <project>/datasets",
    )
    parser.add_argument("--subject", default="sub-001", help="Subject ID")
    parser.add_argument("--session", default="ses-01", help="Session ID")
    parser.add_argument("--run", default="01", help="Run number (e.g., 01)")
    parser.add_argument(
        "--pull",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If run EDF is missing, try to pull it with datalad get",
    )
    parser.add_argument("--start-sec", type=float, default=0.0, help="Plot start time in seconds")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=30.0,
        help="Plot duration in seconds (default 30)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Save plot to this path (e.g., outputs/sub-001_run-01.png)",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display interactive plot window",
    )
    parser.add_argument(
        "--plot-tracker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run PI tracker on ECG and plot fid/peak/base/state and RR tracking",
    )
    parser.add_argument(
        "--plot-kalman-predictor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare scalar-Kalman predicted R-peaks with an offline non-causal lock",
    )
    parser.add_argument(
        "--plot-cancellation-figure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot original, buffered, and Kalman strict-real-time cancellation",
    )
    parser.add_argument(
        "--plot-passloss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run real-data pass-loss analysis (Figures 1-3) using ekg_ilc_rt algorithms",
    )
    parser.add_argument(
        "--plot-ilc-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lock beats, run non-causal ILC, and plot the averaged template against one ECG beat",
    )
    parser.add_argument("--eeg-index", type=int, default=0, help="EEG trace index for pass-loss mode")
    parser.add_argument("--ecg-index", type=int, default=0, help="ECG trace index for pass-loss mode")
    parser.add_argument("--band-lo", type=float, default=8.0, help="Pass-loss lower band edge (Hz)")
    parser.add_argument("--band-hi", type=float, default=20.0, help="Pass-loss upper band edge (Hz)")
    parser.add_argument("--win-sec", type=float, default=8.0, help="Time-varying pass-loss window (s)")
    parser.add_argument("--hop-sec", type=float, default=2.0, help="Time-varying pass-loss hop (s)")
    parser.add_argument("--beats-per-est", type=int, default=12, help="Beat-synchronous pooled beats")
    parser.add_argument("--beat-step", type=int, default=1, help="Beat-synchronous step in beats")
    parser.add_argument("--out-prefix", default=None, help="Pass-loss output prefix (writes *_freq/timevary/figure3_beatsync)")
    parser.add_argument(
        "--ilc-highpass-hz",
        type=float,
        default=0.5,
        help="High-pass cutoff for real-data ILC template preprocessing (default 0.5 Hz)",
    )
    parser.add_argument(
        "--ilc-notch-hz",
        type=float,
        default=50.0,
        help="Notch frequency for real-data ILC template preprocessing (default 50 Hz)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    datasets_root = project_root / "datasets"
    ds_arg = Path(args.dataset_root).expanduser()
    if ds_arg.is_absolute():
        dataset_root = ds_arg.resolve()
    elif ds_arg.parts and ds_arg.parts[0] == "datasets":
        dataset_root = (project_root / ds_arg).resolve()
    else:
        dataset_root = (datasets_root / ds_arg).resolve()

    if args.plot_passloss:
        run_real_passloss(
            dataset_root=dataset_root,
            subject=args.subject,
            session=args.session,
            run=args.run,
            pull=args.pull,
            eeg_index=args.eeg_index,
            ecg_index=args.ecg_index,
            band_lo=args.band_lo,
            band_hi=args.band_hi,
            win_sec=args.win_sec,
            hop_sec=args.hop_sec,
            beats_per_est=args.beats_per_est,
            beat_step=args.beat_step,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            out_prefix=args.out_prefix,
            show=args.show,
        )
        return

    if args.plot_ilc_template:
        out_path = Path(args.out).expanduser().resolve() if args.out else None
        run_real_ilc_template_view(
            dataset_root=dataset_root,
            subject=args.subject,
            session=args.session,
            run=args.run,
            pull=args.pull,
            eeg_index=args.eeg_index,
            ecg_index=args.ecg_index,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            out_path=out_path,
            show=args.show,
            highpass_hz=args.ilc_highpass_hz,
            notch_hz=args.ilc_notch_hz,
        )
        return

    run_data = load_run_data(
        dataset_root=dataset_root,
        subject=args.subject,
        session=args.session,
        run=args.run,
        pull=args.pull,
    )

    out_path = Path(args.out).expanduser().resolve() if args.out else None
    figure_label = (
        f"{dataset_root.name} | {args.subject} | {args.session} | run-{args.run}"
    )
    if args.plot_cancellation_figure:
        out_path = Path(args.out).expanduser().resolve() if args.out else None
        run_real_cancellation_figure(
            dataset_root=dataset_root,
            subject=args.subject,
            session=args.session,
            run=args.run,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            eeg_index=args.eeg_index,
            ecg_index=args.ecg_index,
            full_duration_sec=600.0,
            out_path=out_path,
            show=args.show,
        )
    elif args.plot_kalman_predictor:
        out_path = Path(args.out).expanduser().resolve() if args.out else None
        run_kalman_predictor_comparison(
            ecg_trace=run_data.ecg[0],
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            out_path=out_path,
            figure_label=figure_label,
            show=args.show,
        )
    elif args.plot_tracker:
        tracker_cmp = run_pi_tracker_comparison(run_data.ecg[0])
        plot_tracker_comparison(
            cmp=tracker_cmp,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            out_path=out_path,
            figure_label=figure_label,
            show=args.show,
        )
    else:
        plot_run_data(
            run_data=run_data,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            out_path=out_path,
            figure_label=figure_label,
            show=args.show,
        )


if __name__ == "__main__":
    main()
