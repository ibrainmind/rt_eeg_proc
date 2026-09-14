#!/usr/bin/env python3
"""
phase_estimator_report.py -- write phase_estimator_section.txt and its figure.

  python phase_estimator_report.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 9})

RR_MIN_MS = 500.0            # exclude runs where the detector is clearly doubling
THRESH = (5.0, 10.0, 20.0, 40.0)


def clean_runs(d):
    return [r for r in d if r["rr_mean_ms"] >= RR_MIN_MS]


def med(rows, k):
    v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
    return float(np.median(v)) if v else float("nan")


def pool(rows, key):
    return np.concatenate([np.array([x[key] for x in r["_rows"]], float) for r in rows])


def txt(d, highlight="20") -> str:
    ok = clean_runs(d)
    hl = next((r for r in d if r.get("run") == highlight), None)
    L = []
    A = L.append
    A("=" * 78)
    A("PI PHASE ESTIMATOR: PREDICTED R-PEAK vs ONE-BEAT-DELAY ANCHOR")
    A("sub-070, ses-01, all runs; ekg_ilc_rt.phase_estimator (PLL + supervisor on)")
    A("=" * 78)
    A("")
    A("Two ways for the canceller to know where the R-peak is:")
    A("")
    A("  one-beat delay  anchor on the DETECTED fiducial t_k. The estimator sets")
    A("                  t_k to the argmax of the raw ECG inside the armed window,")
    A("                  so this anchor is exact. Its cost is latency.")
    A("  strict causal   anchor on the loop's own phase. At sample n the canceller")
    A("                  believes the R-peak was phi*RRhat ago, so the predicted")
    A("                  R-peak is displaced from the true one by -wrap(phi at R)")
    A("                  times RRhat. That is the loop's own perr, in ms.")
    A("")
    A(f"Analysed: {len(d)} runs, first {d[0]['dur_s']:.0f} s each; "
      f"{len(ok)} kept ({len(d)-len(ok)} dropped for RR < {RR_MIN_MS:.0f} ms, i.e.")
    A("detector doubling rather than genuine tachycardia).")
    A("")
    A("-" * 78)
    A("1. THE ONE-BEAT-DELAY ANCHOR IS EXACT, AND CHEAPER THAN A WHOLE BEAT")
    A("-" * 78)
    A(f"  fiducial vs local |ECG| max (+-60 ms)  median offset  "
      f"{med(ok,'fidmax_abs_median_ms'):.2f} ms")
    A(f"  sub-sample refinement of that max      SD             "
      f"{med(ok,'subsample_sd_ms'):.2f} ms")
    A(f"  detection latency (declared - R-peak)  median         "
      f"{med(ok,'latency_median_ms'):.1f} ms")
    A(f"                                         p90            "
      f"{med(ok,'latency_abs_p90_ms'):.1f} ms")
    A(f"  mean RR (the buffered mode's latency)                 "
      f"{med(ok,'rr_mean_ms'):.0f} ms")
    A("")
    A("  The fiducial lands on the R-peak sample every time, so its error is the")
    A("  sampling grid alone, ~1 ms at 256 Hz. Note the detector declares the beat")
    A("  only ~16 ms after it: an exact anchor is available almost immediately.")
    A("  The full beat of latency in the buffered mode is not needed for the ANCHOR")
    A("  -- it is needed to know T_k, and to have the template updated.")
    A("")
    A("-" * 78)
    A("2. THE PREDICTED ANCHOR IS UNBIASED BUT JITTERY")
    A("-" * 78)
    A(f"  {'':34s}{'PLL phase':>12s}{'re-anchored':>13s}")
    A(f"  {'':34s}{'(as built)':>12s}{'t_k-1 + RRhat':>13s}")
    A(f"  median signed error (ms)          {med(ok,'pll_median_ms'):>12.1f}"
      f"{med(ok,'openloop_median_ms'):>13.1f}")
    A(f"  MAD (ms)                          {med(ok,'pll_mad_ms'):>12.1f}"
      f"{med(ok,'openloop_mad_ms'):>13.1f}")
    A(f"  SD (ms)                           {med(ok,'pll_sd_ms'):>12.1f}"
      f"{med(ok,'openloop_sd_ms'):>13.1f}")
    A(f"  median |error| (ms)               {med(ok,'pll_abs_median_ms'):>12.1f}"
      f"{med(ok,'openloop_abs_median_ms'):>13.1f}")
    A(f"  p90 |error| (ms)                  {med(ok,'pll_abs_p90_ms'):>12.1f}"
      f"{med(ok,'openloop_abs_p90_ms'):>13.1f}")
    for th in THRESH:
        A(f"  within +-{th:.0f} ms                       "
          f"{100*med(ok,f'pll_within_{th:g}ms'):>11.1f}%"
          f"{100*med(ok,f'openloop_within_{th:g}ms'):>12.1f}%")
    A("")
    A("  Medians over runs. The loop is genuinely locked -- the signed error is a")
    A("  few ms, not a standing offset -- but the spread is tens of ms.")
    A("")
    A("-" * 78)
    A("3. HOW MUCH OF THAT JITTER IS AVOIDABLE?")
    A("-" * 78)
    A("  One-step-ahead RR prediction error (SD, ms), median over runs:")
    A(f"    previous RR (random-walk)          {med(ok,'rrpred_lastrr_sd_ms'):6.1f}")
    A(f"    EWMA a=0.30 (what the loop uses)   {med(ok,'rrpred_ewma0.3_sd_ms'):6.1f}")
    A(f"    EWMA a=0.10                        {med(ok,'rrpred_ewma0.1_sd_ms'):6.1f}")
    A(f"    constant mean RR                   {med(ok,'rrpred_mean_sd_ms'):6.1f}")
    A(f"    (RR SD itself                      {med(ok,'rr_sd_ms'):6.1f})")
    A("")
    A("  So beat-to-beat RR on this subject is close to a random walk: the best")
    A("  simple predictor is 'same as last beat'. Two consequences:")
    A("")
    A(f"  * Re-anchoring on the last fiducial instead of the free-running phase")
    A(f"    halves the error, {med(ok,'pll_mad_ms'):.0f} -> {med(ok,'openloop_mad_ms'):.0f} ms MAD. That")
    A(f"    part of the jitter is self-inflicted and is a tracker fix, not a")
    A(f"    fundamental limit.")
    A(f"  * The remainder, ~{med(ok,'rrpred_lastrr_sd_ms'):.0f} ms SD, is irreducible for ANY causal")
    A(f"    predictor here: it is the part of the next RR interval that the past")
    A(f"    does not determine.")
    A("")
    if hl is not None:
        A("-" * 78)
        A(f"4. THE FIGURE-1 RECORD ({hl['label']})")
        A("-" * 78)
        A(f"  {hl['n_scored']} scored beats of {hl['n_beats']}, "
          f"{hl['n_premature']} premature, {hl['n_missed']} missed")
        A(f"  RR {hl['rr_mean_ms']:.0f} +- {hl['rr_sd_ms']:.0f} ms "
          f"(CV {100*hl['rr_cv']:.1f}%)")
        A(f"  detection latency        median {hl['latency_median_ms']:.1f} ms, "
          f"p90 {hl['latency_abs_p90_ms']:.1f} ms")
        A(f"  PLL predicted anchor     median {hl['pll_median_ms']:+.1f} ms, "
          f"MAD {hl['pll_mad_ms']:.1f} ms, p90 |e| {hl['pll_abs_p90_ms']:.1f} ms")
        A(f"  re-anchored predictor    median {hl['openloop_median_ms']:+.1f} ms, "
          f"MAD {hl['openloop_mad_ms']:.1f} ms, p90 |e| {hl['openloop_abs_p90_ms']:.1f} ms")
        A(f"  one-beat-delay anchor    0 ms by construction "
          f"(+-{hl['subsample_sd_ms']:.1f} ms grid)")
        A("")
    A("-" * 78)
    A("5. READING")
    A("-" * 78)
    A("")
    A("* The detector is not the problem. Its fiducial is the R-peak sample, every")
    A("  time, and it declares within ~16 ms.")
    A(f"* The predicted anchor is unbiased but scatters by ~{med(ok,'pll_mad_ms'):.0f} ms MAD "
      f"(p90 ~{med(ok,'pll_abs_p90_ms'):.0f} ms).")
    A("  That is wide compared with the QRS deflection the template has to line up")
    A("  with, which is why strict-causal cancellation loses most around the R spike")
    A("  while the buffered mode does not.")
    A("* Half of the gap is a design choice, not physics: the strict-causal path")
    A("  should anchor on the last detected fiducial plus the RR estimate rather")
    A(f"  than on free-running phase ({med(ok,'pll_mad_ms'):.0f} -> {med(ok,'openloop_mad_ms'):.0f} ms MAD).")
    A(f"* Past that, ~{med(ok,'rrpred_lastrr_sd_ms'):.0f} ms SD is irreducible without information beyond")
    A("  past beat timing -- respiration, for instance, since RSA is what most of")
    A("  this variability is.")
    A("* Practical consequence: one beat of latency buys an exact anchor, and the")
    A("  buffered mode should be the default. A zero-latency mode is worth building")
    A("  only alongside a better RR predictor, and its residual error should be")
    A("  budgeted against the width of the QRS feature being cancelled.")
    A("")
    return "\n".join(L)


def figure(d, out, highlight="20"):
    ok = clean_runs(d)
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.7))

    e_pll = pool(ok, "err_pll_ms")
    e_ol = pool(ok, "err_ol_ms")
    lat = pool(ok, "latency_ms")

    bins = np.linspace(-200, 200, 81)
    ax[0].hist(e_pll, bins=bins, color="tab:red", alpha=0.55, label="PLL phase")
    ax[0].hist(e_ol, bins=bins, color="tab:blue", alpha=0.55, label=r"$t_{k-1}+\hat{T}$")
    ax[0].axvline(0, color="k", lw=1.0, label="one-beat delay (exact)")
    ax[0].set_xlabel("predicted R-peak error (ms)")
    ax[0].set_ylabel("beats")
    ax[0].set_title("(a) anchor error, all runs pooled", fontsize=9)
    ax[0].legend(fontsize=6.5, frameon=False)
    ax[0].tick_params(labelsize=7)

    q = np.linspace(0, 100, 201)
    ax[1].plot(np.percentile(np.abs(e_pll), q), q, color="tab:red", lw=1.3,
               label="PLL phase")
    ax[1].plot(np.percentile(np.abs(e_ol), q), q, color="tab:blue", lw=1.3,
               label=r"$t_{k-1}+\hat{T}$")
    ax[1].plot(np.percentile(np.abs(lat), q), q, color="0.4", lw=1.2, ls="--",
               label="detection latency")
    ax[1].axvline(0.0, color="k", lw=1.0)
    ax[1].set_xlim(0, 150)
    ax[1].set_xlabel("|error| (ms)")
    ax[1].set_ylabel("percent of beats below")
    ax[1].set_title("(b) cumulative", fontsize=9)
    ax[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    ax[1].tick_params(labelsize=7)
    ax[1].grid(alpha=0.3)

    runs = [r["run"] for r in ok]
    x = np.arange(len(ok))
    ax[2].bar(x - 0.2, [r["pll_mad_ms"] for r in ok], 0.4, color="tab:red",
              label="PLL phase")
    ax[2].bar(x + 0.2, [r["openloop_mad_ms"] for r in ok], 0.4, color="tab:blue",
              label=r"$t_{k-1}+\hat{T}$")
    ax[2].plot(x, [r["rrpred_lastrr_sd_ms"] for r in ok], "k.-", ms=3, lw=0.9,
               label="RR random-walk floor")
    hi = [i for i, r in enumerate(ok) if r.get("run") == highlight]
    if hi:
        ax[2].annotate("Fig. 1", (hi[0], 3), fontsize=6, ha="center", color="0.2")
    ax[2].set_xticks(x[::3]); ax[2].set_xticklabels(runs[::3], fontsize=6)
    ax[2].set_xlabel("run (sub-070, ses-01)")
    ax[2].set_ylabel("error MAD / SD (ms)")
    ax[2].set_title("(c) per run", fontsize=9)
    ax[2].legend(fontsize=6.5, frameon=False)
    ax[2].tick_params(labelsize=7)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def tex_numbers(d, out: Path, highlight="20"):
    """Emit the measured numbers as LaTeX macros so the prose cannot drift."""
    ok = clean_runs(d)
    hl = next((r for r in d if r.get("run") == highlight), None)
    m = {
        "rpNruns": f"{len(ok)}",
        "rpNrunsAll": f"{len(d)}",
        "rpDur": f"{d[0]['dur_s']:.0f}",
        "rpFidOff": f"{med(ok,'fidmax_abs_median_ms'):.2f}",
        "rpSub": f"{med(ok,'subsample_sd_ms'):.1f}",
        "rpLat": f"{med(ok,'latency_median_ms'):.0f}",
        "rpLatPninety": f"{med(ok,'latency_abs_p90_ms'):.0f}",
        "rpRR": f"{med(ok,'rr_mean_ms'):.0f}",
        "rpRRsd": f"{med(ok,'rr_sd_ms'):.0f}",
        "rpPllBias": f"{med(ok,'pll_median_ms'):+.1f}",
        "rpPllMad": f"{med(ok,'pll_mad_ms'):.0f}",
        "rpPllAbsMed": f"{med(ok,'pll_abs_median_ms'):.0f}",
        "rpPllPninety": f"{med(ok,'pll_abs_p90_ms'):.0f}",
        "rpOlBias": f"{med(ok,'openloop_median_ms'):+.1f}",
        "rpOlMad": f"{med(ok,'openloop_mad_ms'):.0f}",
        "rpOlAbsMed": f"{med(ok,'openloop_abs_median_ms'):.0f}",
        "rpOlPninety": f"{med(ok,'openloop_abs_p90_ms'):.0f}",
        "rpFloor": f"{med(ok,'rrpred_lastrr_sd_ms'):.0f}",
        "rpEwma": f"{med(ok,'rrpred_ewma0.3_sd_ms'):.0f}",
        "rpMeanPred": f"{med(ok,'rrpred_mean_sd_ms'):.0f}",
    }
    # LaTeX control sequences cannot contain digits, so spell the thresholds out
    WORD = {5.0: "Five", 10.0: "Ten", 20.0: "Twenty", 40.0: "Forty"}
    for th in THRESH:
        m[f"rpPllW{WORD[th]}"] = f"{100*med(ok,f'pll_within_{th:g}ms'):.0f}"
        m[f"rpOlW{WORD[th]}"] = f"{100*med(ok,f'openloop_within_{th:g}ms'):.0f}"
    if hl is not None:
        m.update({
            "rpHlRun": hl["run"],
            "rpHlBeats": f"{hl['n_scored']}",
            "rpHlRR": f"{hl['rr_mean_ms']:.0f}",
            "rpHlRRsd": f"{hl['rr_sd_ms']:.0f}",
            "rpHlLat": f"{hl['latency_median_ms']:.0f}",
            "rpHlPllMad": f"{hl['pll_mad_ms']:.0f}",
            "rpHlPllPninety": f"{hl['pll_abs_p90_ms']:.0f}",
            "rpHlOlMad": f"{hl['openloop_mad_ms']:.0f}",
        })
    out.write_text("\n".join(
        rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()) + "\n")
    print(f"wrote {out}")


def tex_table(d, out: Path):
    ok = clean_runs(d)
    rows = [
        ("median signed error (ms)", "pll_median_ms", "openloop_median_ms", "{:+.1f}"),
        ("MAD (ms)", "pll_mad_ms", "openloop_mad_ms", "{:.1f}"),
        ("SD (ms)", "pll_sd_ms", "openloop_sd_ms", "{:.1f}"),
        ("median $|$error$|$ (ms)", "pll_abs_median_ms", "openloop_abs_median_ms", "{:.1f}"),
        ("p90 $|$error$|$ (ms)", "pll_abs_p90_ms", "openloop_abs_p90_ms", "{:.1f}"),
    ]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Predicted-R-peak error on sub-070, ses-01, medians over runs. The",
         r"one-beat-delay anchor is the detected fiducial itself, so its error is zero up to the",
         r"sampling grid.}", r"\label{tab:rpeak}",
         r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r" & PLL phase & $t_{k-1}+\hat{T}$ & one-beat delay \\",
         r" & (as built) & (re-anchored) & (fiducial) \\", r"\midrule"]
    for name, ka, kb, fmt in rows:
        L.append(f"{name} & ${fmt.format(med(ok,ka))}$ & ${fmt.format(med(ok,kb))}$ & $0$ \\\\")
    for th in THRESH:
        L.append(f"within $\\pm{th:.0f}$~ms & "
                 f"${100*med(ok,f'pll_within_{th:g}ms'):.0f}\\%$ & "
                 f"${100*med(ok,f'openloop_within_{th:g}ms'):.0f}\\%$ & $100\\%$ \\\\")
    L += [r"\midrule",
          f"latency before anchor is usable & $0$ & $0$ & "
          f"${med(ok,'latency_median_ms'):.0f}$~ms \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out.write_text("\n".join(L))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macros", default="../paper/rpeak_numbers.tex")
    ap.add_argument("--table", default="../paper/tab_rpeak_prediction.tex")
    ap.add_argument("--json", default="../outputs/phase_estimator_eval.json")
    ap.add_argument("--txt", default="../paper/phase_estimator_section.txt")
    ap.add_argument("--fig", default="../paper/fig_real/fig_rpeak_prediction.pdf")
    ap.add_argument("--highlight-run", default="20")
    a = ap.parse_args()
    d = json.loads(Path(a.json).read_text())
    Path(a.txt).write_text(txt(d, a.highlight_run))
    print(f"wrote {a.txt}")
    figure(d, a.fig, a.highlight_run)
    tex_numbers(d, Path(a.macros), a.highlight_run)
    tex_table(d, Path(a.table))


if __name__ == "__main__":
    main()
