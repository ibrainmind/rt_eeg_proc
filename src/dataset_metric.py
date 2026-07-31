#!/usr/bin/env python3
"""
dataset_metric.py  --  "how big is the CFA" across runs of one subject
======================================================================
Basic scaffold, built on the existing workbench:
  - R-peak timing            : ekg_ilc_rt.phase_estimator   (PLL + arrhythmia supervisor)
  - template window / dims    : ekg_ilc_rt._template_dims    ([-0.3s .. +0.5s], R at index ra)
  - non-causal template build : mirrors ekg_ilc_rt.ilc_clean(mode="noncausal") averaging
  - data loading              : process_data.load_run_data   (OpenNeuro ds005873 EDF)

What it measures (per run, then across runs of one subject):
  The non-causal (RRO) averaged template d_hat, and its MAIN-LOBE SNR relative to the
  RRO-removed noise floor. The floor is measured directly by rebuilding the template on
  PHASE-SCRAMBLED R-peaks (circular shift): that template contains only leakage, so

      template_SNR_dB = 10*log10( E_lobe[true] / E_lobe[scrambled] )

  is exactly the "how far the CFA stands above the leakage floor" number, bias-free
  (leakage is in BOTH the numerator template and the scramble floor, so the ratio is honest).
  A bias-corrected CFA amplitude (A_cfa/SD) and power fraction are also reported, so you can
  see the physical size of the artifact, not just the estimate quality.

Two plots:
  (1) template overlay across runs (time since R), main lobe shaded
  (2) main-lobe template SNR (dB) over the RRO-removed floor, per run, with the
      scramble-null significance band

Two entry points to build on:
  subject_cfa_metrics(...)   -> list of per-run dicts (the numbers)
  plot_subject_cfa(...)      -> the two-panel figure

CLI:
  python dataset_metric.py --subject sub-001 --eeg-index 1 --ecg-index 0 \
         --duration-sec 600 --out outputs/sub-001_cfa_overview.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import ekg_ilc_rt as rt   # phase_estimator, _template_dims  (dataset-free import)


# ---------------------------------------------------------------------------
# Core: template averaging (mirrors ilc_clean noncausal branch) + scramble null
# ---------------------------------------------------------------------------
def build_template(y: np.ndarray, r_samples: np.ndarray, ra: int, Lw: int) -> np.ndarray:
    """Phase-synchronous average of y over R-peak sample indices r_samples.

    Identical construction to ekg_ilc_rt.ilc_clean(mode='noncausal'): per-offset count,
    divide by count so edge beats do not bias the mean.
    """
    N = len(y)
    acc = np.zeros(Lw)
    cnt = np.zeros(Lw)
    for fid in r_samples:
        base = int(fid) - ra
        for o in range(Lw):
            k = base + o
            if 0 <= k < N:
                acc[o] += y[k]
                cnt[o] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


def scramble_samples(r_samples: np.ndarray, N: int, rng: np.random.Generator,
                     min_shift_samples: int) -> np.ndarray:
    """Circular time-shift of the whole R-peak train by a random offset.

    Preserves the inter-beat-interval sequence (RSA correlation included), destroys
    absolute R-phase -> the resulting template contains only leakage.
    """
    hi = max(min_shift_samples + 1, N - min_shift_samples)
    shift = int(rng.integers(min_shift_samples, hi))
    return np.sort((r_samples.astype(int) + shift) % N)


def _lobe_slice(ra: int, fs: float, lobe_pre_s: float, lobe_post_s: float):
    lo = max(0, ra - int(round(lobe_pre_s * fs)))
    hi = ra + int(round(lobe_post_s * fs)) + 1
    return slice(lo, hi)


def run_cfa_metric(
    y: np.ndarray,
    r: np.ndarray,
    fs: float,
    label: str = "",
    n_scramble: int = 200,
    lobe_pre_s: float = 0.05,
    lobe_post_s: float = 0.15,
    min_shift_s: float = 2.0,
    seed: int = 0,
    use_pll: bool = True,
    use_override: bool = True,
) -> dict:
    """Compute the non-causal template + scramble-null main-lobe SNR for one (y, r) pair.

    y : EEG channel (already aligned to r's sampling rate, preprocessed as desired)
    r : ECG/EKG reference at the same fs
    Returns a dict of per-run numbers plus the template and its time axis.
    """
    N = len(y)
    ra, Lw = rt._template_dims(fs)
    tsr = (np.arange(Lw) - ra) / fs

    est = rt.phase_estimator(r, fs, use_pll=use_pll, use_override=use_override)
    fids = [f[0] for f in est["fids"] if f[2]]        # learnable beats only (lg == True)
    r_samples = np.asarray(fids, dtype=int)
    n_beats = len(r_samples)

    result = {
        "label": label, "fs": fs, "n_beats": n_beats, "tsr": tsr,
        "template": np.zeros(Lw), "snr_db": np.nan, "e_true": np.nan,
        "e_scr_mean": np.nan, "e_scr_p95": np.nan, "p_value": np.nan,
        "a_cfa_over_sd": np.nan, "cfa_power_fraction": np.nan,
        "lobe_slice": _lobe_slice(ra, fs, lobe_pre_s, lobe_post_s),
    }
    if n_beats < 5:
        return result   # not enough beats to average / scramble meaningfully

    template = build_template(y, r_samples, ra, Lw)
    lobe = result["lobe_slice"]
    e_true = float(np.sum(template[lobe] ** 2))

    rng = np.random.default_rng(seed)
    min_shift = int(round(min_shift_s * fs))
    e_scr = np.empty(n_scramble)
    for b in range(n_scramble):
        scr = scramble_samples(r_samples, N, rng, min_shift)
        tscr = build_template(y, scr, ra, Lw)
        e_scr[b] = float(np.sum(tscr[lobe] ** 2))

    e_scr_mean = float(np.mean(e_scr))
    p_value = float(np.mean(e_scr >= e_true))            # null: scramble beats true?
    snr_db = 10.0 * np.log10(e_true / (e_scr_mean + 1e-30))

    # bias-corrected physical CFA size
    L_lobe = lobe.stop - lobe.start
    cfa_energy = max(e_true - e_scr_mean, 0.0)
    a_cfa = np.sqrt(cfa_energy / L_lobe)                 # RMS CFA deflection in the lobe
    sd_bg = float(np.std(y))                             # background+all, ratio currency
    a_over_sd = a_cfa / (sd_bg + 1e-30)
    f_power = a_over_sd ** 2 / (1.0 + a_over_sd ** 2)    # approx CFA share of power

    result.update({
        "template": template, "e_true": e_true, "e_scr_mean": e_scr_mean,
        "e_scr_p95": float(np.percentile(e_scr, 95)), "p_value": p_value,
        "snr_db": snr_db, "a_cfa_over_sd": a_over_sd, "cfa_power_fraction": f_power,
    })
    return result


# ---------------------------------------------------------------------------
# Loading: one subject across runs (lazy import of process_data so the core above
# stays usable without mne / the dataset present)
# ---------------------------------------------------------------------------
def _resolve_dataset_root(dataset_root: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    ds = Path(dataset_root).expanduser()
    if ds.is_absolute():
        return ds.resolve()
    if ds.parts and ds.parts[0] == "datasets":
        return (project_root / ds).resolve()
    return (project_root / "datasets" / ds).resolve()


def discover_runs(dataset_root: Path, subject: str, session: str) -> list[str]:
    """List run ids that have BOTH an eeg and an ecg edf present (skips ECG-missing runs)."""
    import re
    eeg_dir = dataset_root / subject / session / "eeg"
    ecg_dir = dataset_root / subject / session / "ecg"
    runs = []
    if not eeg_dir.exists():
        return runs
    for p in sorted(eeg_dir.glob(f"{subject}_{session}_task-szMonitoring_run-*_eeg.edf")):
        m = re.search(r"run-(\w+)_eeg\.edf$", p.name)
        if not m:
            continue
        run = m.group(1)
        ecg = ecg_dir / f"{subject}_{session}_task-szMonitoring_run-{run}_ecg.edf"
        if ecg.exists():
            runs.append(run)
    return runs


def load_eeg_ecg(
    dataset_root: Path, subject: str, session: str, run: str,
    eeg_index: int, ecg_index: int, pull: bool,
    highpass_hz: float | None, notch_hz: float | None,
    start_sec: float, duration_sec: float | None,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Load one EEG + one ECG channel, EEG resampled to ECG fs and MNE-preprocessed."""
    import process_data as pd
    rd = pd.load_run_data(dataset_root, subject=subject, session=session, run=run, pull=pull)
    eeg = rd.eeg[eeg_index]
    ecg = rd.ecg[ecg_index]
    fs = float(ecg.fs)

    max_dur = min(len(eeg.values) / eeg.fs, len(ecg.values) / ecg.fs)
    n_out = max(2, int(np.floor(max_dur * fs)))
    r = np.asarray(ecg.values[:n_out], dtype=float)
    if eeg.fs == fs:
        y = np.asarray(eeg.values[:n_out], dtype=float)
    else:
        y = pd._resample_linear(np.asarray(eeg.values, dtype=float), eeg.fs, fs, n_out)
    y = pd._mne_preprocess_trace(y, fs, highpass_hz=highpass_hz, notch_hz=notch_hz)

    n = min(len(y), len(r))
    y, r = y[:n], r[:n]
    t = np.arange(n) / fs
    a = max(0.0, start_sec)
    b = t[-1] if duration_sec is None else min(t[-1], a + duration_sec)
    m = (t >= a) & (t <= b)
    label = f"{subject}/{session}/run-{run} EEG={eeg.name} ECG={ecg.name}"
    return y[m], r[m], fs, label


def subject_cfa_metrics(
    dataset_root: str = "ds005873",
    subject: str = "sub-001",
    session: str = "ses-01",
    runs: list[str] | None = None,
    eeg_index: int = 1,
    ecg_index: int = 0,
    pull: bool = True,
    highpass_hz: float | None = 0.5,
    notch_hz: float | None = 50.0,
    start_sec: float = 0.0,
    duration_sec: float | None = 600.0,
    n_scramble: int = 200,
) -> list[dict]:
    """Run the CFA metric on every run of one subject. Returns a list of per-run dicts."""
    root = _resolve_dataset_root(dataset_root)
    if runs is None:
        runs = discover_runs(root, subject, session)
        if not runs:
            runs = ["01"]   # fall back to the canonical run if glob finds nothing
    out = []
    for run in runs:
        try:
            y, r, fs, label = load_eeg_ecg(
                root, subject, session, run, eeg_index, ecg_index, pull,
                highpass_hz, notch_hz, start_sec, duration_sec,
            )
            m = run_cfa_metric(y, r, fs, label=f"run-{run}", n_scramble=n_scramble)
            m["run"] = run
            out.append(m)
            print("  %-8s beats=%4d  SNR=%6.2f dB  A_cfa/SD=%.3f  f=%.3f  p=%.3f"
                  % (f"run-{run}", m["n_beats"], m["snr_db"],
                     m["a_cfa_over_sd"], m["cfa_power_fraction"], m["p_value"]))
        except Exception as e:
            print("  run-%s SKIPPED: %s" % (run, e))
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_subject_cfa(metrics: list[dict], subject: str = "", out: str | None = None,
                     show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in metrics if np.isfinite(m["snr_db"])]
    if not metrics:
        print("no valid runs to plot")
        return

    fig, ax = plt.subplots(2, 1, figsize=(11, 8))

    # (1) template overlay (uV), main lobe shaded
    lobe = metrics[0]["lobe_slice"]
    tsr0 = metrics[0]["tsr"]
    for m in metrics:
        ax[0].plot(m["tsr"], m["template"] * 1e6, lw=0.9, alpha=0.85, label=m["label"])
    ax[0].axvspan(tsr0[lobe.start], tsr0[min(lobe.stop, len(tsr0) - 1)],
                  color="orange", alpha=0.12, label="main lobe")
    ax[0].axvline(0, color="0.5", lw=0.6)
    ax[0].set_title("(1) non-causal (RRO) template overlay across runs  —  %s" % subject)
    ax[0].set_xlabel("time since R-peak (s)")
    ax[0].set_ylabel("amplitude (uV)")
    ax[0].grid(alpha=0.3)
    if len(metrics) <= 8:
        ax[0].legend(fontsize=7, ncol=2)

    # (2) main-lobe template SNR over the RRO-removed floor, per run
    runs = [m.get("run", m["label"]) for m in metrics]
    snr = [m["snr_db"] for m in metrics]
    x = np.arange(len(metrics))
    floor_db = [10 * np.log10(m["e_scr_p95"] / (m["e_scr_mean"] + 1e-30)) for m in metrics]
    ax[1].bar(x, snr, color="tab:blue", alpha=0.8, width=0.6)
    ax[1].plot(x, floor_db, "o--", color="tab:red", ms=4, lw=1,
               label="null 95th pct (significance)")
    ax[1].axhline(0, color="0.5", lw=0.6)
    for xi, m in zip(x, metrics):
        ax[1].annotate("N=%d" % m["n_beats"], (xi, m["snr_db"]),
                       textcoords="offset points", xytext=(0, 3),
                       ha="center", fontsize=7)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
    ax[1].set_title("(2) main-lobe template SNR over RRO-removed floor (higher = larger/cleaner CFA)")
    ax[1].set_ylabel("template SNR (dB)")
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        print("wrote", out)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Per-subject CFA-size metric across runs")
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--subject", default="sub-001")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run ids; default = discover")
    ap.add_argument("--eeg-index", type=int, default=1)
    ap.add_argument("--ecg-index", type=int, default=0)
    ap.add_argument("--pull", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--highpass-hz", type=float, default=0.5)
    ap.add_argument("--notch-hz", type=float, default=50.0)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--n-scramble", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    print("subject %s: computing CFA metric across runs" % args.subject)
    metrics = subject_cfa_metrics(
        dataset_root=args.dataset_root, subject=args.subject, session=args.session,
        runs=args.runs, eeg_index=args.eeg_index, ecg_index=args.ecg_index,
        pull=args.pull, highpass_hz=args.highpass_hz, notch_hz=args.notch_hz,
        start_sec=args.start_sec, duration_sec=args.duration_sec, n_scramble=args.n_scramble,
    )
    out = args.out if args.out else (None if args.show else "cfa_overview.png")
    plot_subject_cfa(metrics, subject=args.subject, out=out, show=args.show)


if __name__ == "__main__":
    main()
