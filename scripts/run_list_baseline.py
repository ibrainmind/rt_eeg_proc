#!/usr/bin/env python3
"""Regenerate raw phase-scramble metrics for entries in dateset_list.txt."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import dataset_metric as dm


ENTRY_RE = re.compile(
    r"\s+(sub-\d+)\s+run-(\S+)\s+eeg=(\d+)\s+"
    r"SNR=([0-9.]+)\s+dB\s+A/SD=([0-9.]+)"
)


def entries_from_list(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text().splitlines():
        match = ENTRY_RE.match(line)
        if match:
            entries.append({
                "subject": match.group(1),
                "run": match.group(2),
                "eeg_index": int(match.group(3)),
                "listed_snr_db": float(match.group(4)),
                "listed_a_over_sd": float(match.group(5)),
            })
    return entries


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row.get("status") == "ok"]

    def stats(key: str) -> dict:
        values = np.asarray([row[key] for row in successful], dtype=float)
        return {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "above_20_db": int(np.sum(values >= 20.0)),
        }

    return {
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "snr_db": stats("snr_db") if successful else {},
        "a_over_sd": stats("a_over_sd") if successful else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/ds005873"))
    parser.add_argument("--list", type=Path, default=Path("paper/dateset_list.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline_metrics_365.json"))
    parser.add_argument("--n-scramble", type=int, default=200)
    parser.add_argument("--max-entries", type=int, default=None)
    args = parser.parse_args()

    entries = entries_from_list(args.list)
    if args.max_entries is not None:
        entries = entries[:args.max_entries]
    existing = {}
    if args.out.exists():
        existing = {(
            row["subject"], row["run"], row["eeg_index"]
        ): row for row in json.loads(args.out.read_text()).get("rows", [])}

    rows = []
    for index, entry in enumerate(entries, 1):
        key = (entry["subject"], entry["run"], entry["eeg_index"])
        if existing.get(key, {}).get("status") == "ok":
            rows.append(existing[key])
            continue
        row = dict(entry)
        try:
            y, r, fs, label = dm.load_eeg_ecg(
                args.dataset_root, entry["subject"], "ses-01", entry["run"],
                entry["eeg_index"], 0, False, 0.5, 50.0, 0.0, 600.0,
            )
            metric = dm.run_cfa_metric(y, r, fs, label=label, n_scramble=args.n_scramble)
            row.update({
                "status": "ok",
                "n_beats": metric["n_beats"],
                "snr_db": metric["snr_db"],
                "a_over_sd": metric["a_cfa_over_sd"],
                "p_value": metric["p_value"],
            })
        except Exception as exc:
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
        payload = {
            "source": str(args.list),
            "n_scramble": args.n_scramble,
            "entries": len(entries),
            "summary": summarize(rows),
            "rows": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        if index % 10 == 0 or index == len(entries):
            print(f"processed {index}/{len(entries)}; {payload['summary']}", flush=True)

    payload = {
        "source": str(args.list),
        "n_scramble": args.n_scramble,
        "entries": len(entries),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()