#!/usr/bin/env python3
"""
phase_estimator_eval.py -- how well does the PI phase estimator PREDICT the next R-peak?

The buffered (one-beat-delay) canceller anchors on the DETECTED fiducial t_k,
which ekg_ilc_rt.phase_estimator sets to the argmax of the raw ECG inside the
armed window.  That anchor is exact by construction; its cost is that it is not
available until the beat is over.  The strict-causal canceller cannot wait, so it
anchors on the PLL's own phase: the sample at which phi wraps through zero is the
estimator's prediction of where the R-peak will be.

This script measures the gap between the two on real ECG:

  * detection latency        how long after the R-peak the detector declares it
                             (why a zero-latency canceller must predict at all)
  * PLL prediction error     phi-wrap instant minus the detected fiducial
  * open-loop comparison     t_{k-1} + RRhat, and two alternative RR predictors,
                             to show whether the loop is leaving anything on the
                             table or is up against genuine beat-to-beat variance
  * fiducial quality         agreement of the fiducial with the local |ECG| max,
                             and the sub-sample refinement offset, i.e. the floor
                             the buffered anchor is quantised to

  python phase_estimator_eval.py --subject sub-070 --session ses-01 \
      --out ../outputs/phase_estimator_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dataset_metric as dm
import ekg_ilc_rt as C

REFINE_WIN_S = 0.06          # +-60 ms around the fiducial, for the local |ECG| max
THRESHOLDS_MS = (2.0, 5.0, 10.0, 20.0, 40.0)


def load_ecg(dataset_root: Path, subject: str, session: str, run: str):
    """ECG trace only -- avoids pulling the much larger EEG file."""
    import process_data as pd
    _, ecg_file = pd.build_run_file_paths(dataset_root, subject=subject,
                                          session=session, run=run)
    raw = pd.load_raw_edf(ecg_file)
    ch = pd.select_ecg_channels(raw)[0]
    r = raw.get_data(picks=[ch])[0].astype(float)
    return r, float(raw.info["sfreq"]), ch


def parabolic_offset(x, i):
    """Sub-sample offset of a local max at index i by 3-point parabolic fit."""
    if i <= 0 or i >= len(x) - 1:
        return 0.0
    a, b, c = x[i - 1], x[i], x[i + 1]
    den = a - 2 * b + c
    return 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def _wrap(x):
    """Signed phase error in (-0.5, 0.5], the same wrap the PLL applies."""
    return ((x + 0.5) % 1.0) - 0.5


def analyse_run(r, fs, label, duration_sec=None) -> dict | None:
    if duration_sec:
        r = r[: int(round(duration_sec * fs))]
    est = C.phase_estimator(r, fs, use_pll=True, use_override=True, debug=True)
    fids = est["fids"]
    det = np.flatnonzero(est["detect_events"])
    phi = est["phi"]; rrh = est["rrhat"]
    if len(fids) < 40 or len(det) != len(fids):
        return None

    F = np.array([f[0] for f in fids], dtype=int)
    LG = np.array([f[2] for f in fids], dtype=bool)
    TY = [f[1] for f in fids]
    N = len(r)
    w = int(round(REFINE_WIN_S * fs))

    rows = []
    for k in range(1, len(F)):
        n0, n1 = int(det[k - 1]), int(det[k])
        fid = int(F[k])
        # The strict-causal canceller reads its anchor off the loop state: at any
        # sample it believes the R-peak was phi*RRhat ago, equivalently that the
        # next one is (1-phi)*RRhat away.  So the predicted R-peak nearest the
        # true one is displaced by -wrap(phi at the R-peak) * RRhat.  This is the
        # loop's own perr, in milliseconds.
        phi_pk = float(phi[fid])
        rr_now = float(rrh[fid])
        e_pll_ms = -_wrap(phi_pk) * rr_now * 1e3
        # A predictor that used the previous fiducial and the RR estimate instead
        # of the free-running phase: what a re-anchored tracker could achieve.
        rr_prev = float(rrh[min(n0 + 1, N - 1)])
        e_ol_ms = (F[k - 1] + rr_prev * fs - fid) / fs * 1e3
        a, b = max(0, fid - w), min(N, fid + w + 1)
        loc = a + int(np.argmax(np.abs(r[a:b] - np.mean(r[a:b]))))
        rows.append({
            "fid": fid, "type": TY[k], "lg": bool(LG[k] and LG[k - 1]),
            "phi_at_R": phi_pk,
            "err_pll_ms": e_pll_ms,
            "err_ol_ms": e_ol_ms,
            "latency_ms": (n1 - fid) / fs * 1e3,
            "fid_vs_localmax_ms": (loc - fid) / fs * 1e3,
            "subsample_ms": parabolic_offset(r, fid) / fs * 1e3,
            "T_prev_ms": (fid - F[k - 1]) / fs * 1e3,
        })

    T = np.diff(F) / fs
    good = [x for x in rows if x["lg"] and np.isfinite(x["err_pll_ms"])]
    if len(good) < 30:
        return None
    out = {
        "label": label, "fs": fs, "dur_s": len(r) / fs,
        "n_beats": int(len(F)),
        "n_premature": int(sum(t == "premature" for t in TY)),
        "n_missed": int(sum(t == "missed" for t in TY)),
        "n_scored": len(good),
        "rr_mean_ms": float(np.mean(T) * 1e3), "rr_sd_ms": float(np.std(T) * 1e3),
        "rr_cv": float(np.std(T) / np.mean(T)),
        # phi at the R-peak is circular and sits right on the wrap when the loop
        # is locked, so summarise the wrapped error, not phi itself.
        "phase_err_cycles_median": float(np.median([_wrap(x["phi_at_R"]) for x in good])),
        "phase_err_cycles_mad": 1.4826 * float(np.median(np.abs(
            np.array([_wrap(x["phi_at_R"]) for x in good])
            - np.median([_wrap(x["phi_at_R"]) for x in good])))),
    }
    for key, name in (("err_pll_ms", "pll"), ("err_ol_ms", "openloop"),
                      ("latency_ms", "latency"), ("fid_vs_localmax_ms", "fidmax"),
                      ("subsample_ms", "subsample")):
        v = np.array([x[key] for x in good], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        med = float(np.median(v))
        out[f"{name}_bias_ms"] = float(np.mean(v))
        out[f"{name}_median_ms"] = med
        out[f"{name}_sd_ms"] = float(np.std(v))
        out[f"{name}_mad_ms"] = 1.4826 * float(np.median(np.abs(v - med)))
        out[f"{name}_abs_median_ms"] = float(np.median(np.abs(v)))
        out[f"{name}_abs_p90_ms"] = float(np.percentile(np.abs(v), 90))
        out[f"{name}_abs_p95_ms"] = float(np.percentile(np.abs(v), 95))
        # after removing the standing bias -- what a fixed offset could not fix
        out[f"{name}_debiased_abs_median_ms"] = float(np.median(np.abs(v - med)))
        out[f"{name}_debiased_abs_p90_ms"] = float(np.percentile(np.abs(v - med), 90))
    for key, name in (("err_pll_ms", "pll"), ("err_ol_ms", "openloop")):
        v = np.array([x[key] for x in good], dtype=float)
        for th in THRESHOLDS_MS:
            out[f"{name}_within_{th:g}ms"] = float(np.mean(np.abs(v) <= th))
    out["_rows"] = good
    out["_rr_ms"] = (T * 1e3).tolist()
    return out


def rr_predictors(rr_ms: np.ndarray) -> dict:
    """One-step-ahead RR prediction error for the loop's rule and two alternatives."""
    T = np.asarray(rr_ms, dtype=float)
    if len(T) < 20:
        return {}
    out = {}
    # what the PLL does: EWMA with gain 0.30
    for a in (0.30, 0.10):
        p = np.empty(len(T)); p[0] = T[0]
        for i in range(1, len(T)):
            p[i] = p[i - 1] + a * (T[i - 1] - p[i - 1])
        e = T[1:] - p[1:]
        out[f"ewma{a:g}_sd_ms"] = float(np.std(e))
    e = T[1:] - T[:-1]                       # last-RR predictor
    out["lastrr_sd_ms"] = float(np.std(e))
    out["mean_sd_ms"] = float(np.std(T - np.mean(T)))   # constant-mean predictor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="ds005873")
    ap.add_argument("--subject", default="sub-070")
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--runs", default="", help="comma list; default = all local runs")
    ap.add_argument("--highlight-run", default="20", help="the Figure-1 record")
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--out", default="../outputs/phase_estimator_eval.json")
    a = ap.parse_args()

    root = dm._resolve_dataset_root(a.dataset_root)
    runs = [x for x in a.runs.split(",") if x] or dm.discover_runs(root, a.subject, a.session)
    print(f"{a.subject}/{a.session}: {len(runs)} runs")

    results = []
    for run in runs:
        try:
            r, fs, ch = load_ecg(root, a.subject, a.session, run)
        except Exception as e:
            print(f"  [skip] run-{run}: {type(e).__name__}: {e}", flush=True)
            continue
        res = analyse_run(r, fs, f"{a.subject}/{a.session}/run-{run}",
                          duration_sec=a.duration_sec)
        if res is None:
            print(f"  [skip] run-{run}: too few beats or detector mismatch", flush=True)
            continue
        res["run"] = run
        res["ecg_channel"] = ch
        res.update({f"rrpred_{k}": v for k, v in
                    rr_predictors(np.asarray(res["_rr_ms"])).items()})
        results.append(res)
        print(f"  run-{run}: N={res['n_beats']:4d} RR={res['rr_mean_ms']:.0f}"
              f"+-{res['rr_sd_ms']:.0f} ms  latency med {res['latency_median_ms']:.0f} ms  "
              f"phase err {res['phase_err_cycles_median']:+.3f} cyc  "
              f"PLL err {res['pll_median_ms']:+.1f} ms (MAD {res['pll_mad_ms']:.1f}), "
              f"open-loop {res['openloop_median_ms']:+.1f} (MAD {res['openloop_mad_ms']:.1f})",
              flush=True)

    Path(a.out).write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {a.out} ({len(results)} runs)")


if __name__ == "__main__":
    main()
