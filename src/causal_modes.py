#!/usr/bin/env python3
"""
causal_modes.py -- non-causal vs buffered causal vs strict causal cancellation.

All three modes are scored the same ground-truth-free way used elsewhere in this
repo: the template that cancels a beat is never built from that beat, and the
residual is measured on identical time-domain samples, so the modes differ only
in what information the canceller is allowed to use.

  non-causal  batch average over the other half of the beats; anchor and warp
              from the DETECTED peak and the MEASURED period.
  buffered    exponential ILC recursion over beats < k (one beat of latency);
              anchor and warp still from the detected peak / measured period,
              which the one-beat wait makes available.
  strict      same recursion, but the anchor is the PREDICTED peak
              t_{k-1} + That_k and the warp uses the predicted period, since at
              zero latency neither t_k nor T_k has arrived yet.

The gap between the last two is the price of zero latency, and it is set by
R-peak prediction error, so we also sweep an injected anchoring jitter
sigma_j = 0..40 ms and report where the measured prediction error falls on that
curve.

  python causal_modes.py --scan ../outputs/phase_hep_gated.json \
      --out ../outputs/causal_modes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dataset_metric as dm
import phase_hep_analysis as ph

JITTERS_MS = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
WARMUP_BEATS = 60
A_GAIN = 0.05
EWMA = 0.10


def _beat_window(fid: int, Tk: float, fs: float, n: int):
    """Evaluation offsets (s) for one beat, clipped at the next R and at the record."""
    hi = min(ph.POST_S, Tk)
    off = np.arange(-int(round(ph.PRE_S * fs)), int(round(hi * fs))) / fs
    ts = fid / fs + off
    ok = (ts >= 0.0) & (ts <= (n - 1) / fs)
    return off[ok], ts[ok]


def run_record(
    y: np.ndarray,
    r_samples: np.ndarray,
    T: np.ndarray,
    fs: float,
    gamma: float,
    T_bar: float,
    c_grid: np.ndarray,
    a_gain: float = A_GAIN,
    ewma: float = EWMA,
    jitters_ms=JITTERS_MS,
    seed: int = 0,
) -> dict:
    """Held-out variance reduction for each streaming mode on one record."""
    n = len(y)
    tn = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(seed)

    # --- streaming state: one template per arm, all fed the same beats -------
    arms = ["buffered", "strict"] + [f"jit{j:g}" for j in jitters_ms]
    m = {k: np.zeros(len(c_grid)) for k in arms}
    num = {k: 0.0 for k in arms}
    den = {k: 0.0 for k in arms}
    num_q = {k: 0.0 for k in arms}
    den_q = {k: 0.0 for k in arms}

    T_hat = float(T[0])
    pred_err_s = []

    for k in range(len(r_samples)):
        fid, Tk = int(r_samples[k]), float(T[k])
        w = (Tk / T_bar) ** gamma
        off, ts = _beat_window(fid, Tk, fs, n)
        if len(off) == 0:
            continue
        seg = np.interp(ts, tn, y)

        # predicted anchor and period, available before t_k arrives
        if k == 0:
            fid_pred, w_pred = fid, w
        else:
            t_pred = r_samples[k - 1] / fs + T_hat
            fid_pred = t_pred * fs
            w_pred = (T_hat / T_bar) ** gamma
            pred_err_s.append(t_pred - fid / fs)

        if k >= WARMUP_BEATS:
            preds = {
                "buffered": np.interp(off / w, c_grid, m["buffered"]),
                "strict": np.interp(
                    (ts - fid_pred / fs) / w_pred, c_grid, m["strict"]),
            }
            for j in jitters_ms:
                e = rng.normal(0.0, j * 1e-3)
                preds[f"jit{j:g}"] = np.interp(
                    (off - e) / w, c_grid, m[f"jit{j:g}"])
            qm = (off >= ph.QRS_LO_S) & (off < ph.QRS_HI_S)
            for kk, pr in preds.items():
                res = seg - pr
                num[kk] += float(np.sum(res ** 2))
                den[kk] += float(np.sum(seg ** 2))
                if np.any(qm):
                    num_q[kk] += float(np.sum(res[qm] ** 2))
                    den_q[kk] += float(np.sum(seg[qm] ** 2))

        # --- learn from beat k (after it has been scored) -------------------
        ts_l = fid / fs + c_grid * w
        okl = (ts_l >= 0.0) & (ts_l <= tn[-1])
        yl = np.interp(ts_l[okl], tn, y)
        for kk in arms:
            m[kk][okl] += a_gain * (yl - m[kk][okl])

        T_hat += ewma * (Tk - T_hat)

    def vr(nu, de):
        return 10.0 * np.log10(de / nu) if nu > 0 and de > 0 else np.nan

    out = {
        "gamma": gamma,
        "pred_err_sd_ms": float(np.std(pred_err_s) * 1e3) if pred_err_s else np.nan,
        "pred_err_mad_ms": float(
            np.median(np.abs(np.asarray(pred_err_s) - np.median(pred_err_s))) * 1e3
        ) if pred_err_s else np.nan,
    }
    for kk in arms:
        out[f"vr_{kk}"] = vr(num[kk], den[kk])
        out[f"vrq_{kk}"] = vr(num_q[kk], den_q[kk])
    return out


def analyse(root, subject, session, run, eeg_index, duration_sec, gammas=(0.0, 1.0)) -> dict | None:
    y, r, fs, label = dm.load_eeg_ecg(
        root, subject, session, run, eeg_index, 0, False,
        highpass_hz=0.5, notch_hz=50.0, start_sec=0.0, duration_sec=duration_sec,
    )
    prep = ph._prepare(y, r, fs, 0)
    if prep is None:
        return None
    r_samples, T, T_bar, c_grid = prep
    if len(r_samples) < WARMUP_BEATS + 40:
        return None
    meta = ph._stats(r_samples, T, T_bar, fs, label, subject, session, run, eeg_index)

    # non-causal reference on the same window definition
    meta["vr_noncausal"] = {}
    for g in gammas:
        e = ph.eval_gamma(y, r_samples, T, fs, g, T_bar, c_grid)
        meta["vr_noncausal"][str(g)] = {"all": e["vr_db_all"], "qrs": e["vr_db_qrs"]}
    meta["modes"] = {
        str(g): run_record(y, r_samples, T, fs, g, T_bar, c_grid) for g in gammas
    }
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="../outputs/phase_hep_gated.json")
    ap.add_argument("--out", default="../outputs/causal_modes.json")
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    recs = json.loads(Path(a.scan).read_text())
    if a.limit:
        recs = recs[: a.limit]
    root = dm._resolve_dataset_root(a.dataset_root)
    out = []
    for r in recs:
        try:
            m = analyse(root, r["subject"], r["session"], r["run"], r["eeg_index"],
                        a.duration_sec)
        except Exception as e:
            print(f"[skip] {r['label']}: {type(e).__name__}: {e}", flush=True)
            continue
        if m is None:
            continue
        out.append(m)
        d1 = m["modes"]["1.0"]
        print(f"{r['subject']}/run-{r['run']}/eeg{r['eeg_index']}  "
              f"nc={m['vr_noncausal']['1.0']['all']:+.2f} "
              f"buf={d1['vr_buffered']:+.2f} strict={d1['vr_strict']:+.2f}dB  "
              f"predSD={d1['pred_err_sd_ms']:.1f}ms", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out} ({len(out)} records)")


if __name__ == "__main__":
    main()
