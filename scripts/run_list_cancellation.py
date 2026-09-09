#!/usr/bin/env python3
"""Generate buffered and strict-real-time metrics for dateset_list.txt."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import dataset_metric as dm
import ekg_ilc_rt as rt

ENTRY_RE = re.compile(r"\s+(sub-\d+)\s+run-(\S+)\s+eeg=(\d+)")


def metric(y: np.ndarray, r: np.ndarray, fs: float, n_scramble: int = 200) -> dict:
    n = len(y)
    ra, width = rt._template_dims(fs)
    estimate = rt.phase_estimator(r, fs, use_pll=True, use_override=True)
    peaks = np.asarray([fid for fid, _, learnable in estimate["fids"] if learnable], dtype=int)
    offsets = np.arange(width)

    def build(peak_set: np.ndarray) -> np.ndarray:
        indices = peak_set[:, None] - ra + offsets
        valid = (indices >= 0) & (indices < n)
        values = np.where(valid, y[np.clip(indices, 0, n - 1)], 0.0)
        return values.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)

    template = build(peaks)
    lobe = slice(max(0, ra - round(0.05 * fs)), ra + round(0.15 * fs) + 1)
    true_energy = float(np.sum(template[lobe] ** 2))
    rng = np.random.default_rng(0)
    scrambled_energy = []
    for _ in range(n_scramble):
        shifted = dm.scramble_samples(peaks, n, rng, round(2.0 * fs))
        scrambled_energy.append(float(np.sum(build(shifted)[lobe] ** 2)))
    floor = float(np.mean(scrambled_energy))
    return {
        "n_beats": int(len(peaks)),
        "snr_db": float(10.0 * np.log10(true_energy / (floor + 1e-30))),
        "a_over_sd": float(np.sqrt(max(true_energy - floor, 0.0) / (lobe.stop - lobe.start)) / (np.std(y) + 1e-30)),
    }


def stats(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows if row["status"] == "ok"], dtype=float)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "above_20_db": int(np.sum(values >= 20.0)),
    }


list_path = Path("paper/dateset_list.txt")
out_path = Path("outputs/cancellation_metrics_365_200scramble.json")
entries = []
for line in list_path.read_text().splitlines():
    match = ENTRY_RE.match(line)
    if match:
        entries.append({"subject": match.group(1), "run": match.group(2), "eeg_index": int(match.group(3))})

existing = {}
if out_path.exists():
    existing = {(row["subject"], row["run"], row["eeg_index"]): row for row in json.loads(out_path.read_text()).get("rows", [])}
rows = []
for index, entry in enumerate(entries, 1):
    key = (entry["subject"], entry["run"], entry["eeg_index"])
    if existing.get(key, {}).get("status") == "ok":
        rows.append(existing[key])
        continue
    row = dict(entry)
    try:
        y, r, fs, label = dm.load_eeg_ecg(Path("datasets/ds005873"), entry["subject"], "ses-01", entry["run"], entry["eeg_index"], 0, False, 0.5, 50.0, 0.0, 600.0)
        estimate = rt.phase_estimator(r, fs, use_pll=True, use_override=True)
        for mode in ("buffered", "causal"):
            cleaned, _ = rt.ilc_clean(y, fs, estimate, mode=mode, use_ilc=True)
            result = metric(cleaned, r, fs)
            row[f"{mode}_snr_db"] = result["snr_db"]
            row[f"{mode}_a_over_sd"] = result["a_over_sd"]
        row["status"] = "ok"
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    rows.append(row)
    payload = {"source": str(list_path), "entries": len(entries), "n_scramble": 200, "rows": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    if index % 25 == 0 or index == len(entries):
        print(f"processed {index}/{len(entries)}", flush=True)

ok = [row for row in rows if row["status"] == "ok"]
payload = {
    "source": str(list_path), "entries": len(entries), "successful": len(ok), "n_scramble": 200,
    "buffered_snr_db": stats(rows, "buffered_snr_db"), "strict_snr_db": stats(rows, "causal_snr_db"),
    "buffered_a_over_sd": stats(rows, "buffered_a_over_sd"), "strict_a_over_sd": stats(rows, "causal_a_over_sd"),
    "rows": rows,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
