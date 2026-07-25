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
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load ds005873 sub-001 run-01 EEG/ECG and plot aligned subplots."
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/ds005873",
        help="Path to ds005873 dataset root",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
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
