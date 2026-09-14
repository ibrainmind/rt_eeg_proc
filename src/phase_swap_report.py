#!/usr/bin/env python3
"""
phase_swap_report.py -- turn outputs/phase_swap.json into phase_swap.txt + a figure.

  python phase_swap_report.py
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

MODES = ("noncausal", "buffered")
MODE_LABEL = {"noncausal": "non-causal", "buffered": "one-beat delay"}
GAMMAS = (0.0, 0.5, 1.0)


def txt_tables(d) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A("WHAT DOES PHASE WARPING BUY THE FIGURE-1 CANCELLER?")
    A("=" * 78)
    A("")
    A("gamma = 0  : fixed lag from the R-peak (the shipped ilc_clean path, and")
    A("             what Figure 1 of the paper is drawn with)")
    A("gamma = 1  : normalized cardiac phase (template stretched to each beat)")
    A("gamma = 0.5: square-root warp, halfway between")
    A("")
    A("-" * 78)
    A("1. SYNTHETIC, WITH GROUND TRUTH")
    A("-" * 78)
    A("SNR improvement over the contaminated input (dB), measured against the true")
    A("neural signal, mean of 3 seeds, 180 s each.")
    A("")
    A("The shipped generator (gen_eeg) places a FIXED-SHAPE PQRST at every beat, so")
    A("its artifact is strictly time-locked to R and gamma=1 cannot help by")
    A("construction. The stretched variant scales each beat's waveform by T_k/Tbar,")
    A("where gamma=1 is correct by construction. Reality is somewhere between.")
    A("")
    hdr = f"{'artifact':>16s} {'CV_RR':>6s} | " + " | ".join(
        f"{MODE_LABEL[m]:>28s}" for m in MODES)
    A(hdr)
    A(f"{'':>16s} {'':>6s} | " + " | ".join(
        f"{'g=0':>8s}{'g=0.5':>9s}{'g=1':>8s}{'D':>7s}" for m in MODES))
    A("-" * len(hdr))
    for r in d["synthetic"]:
        cells = []
        for m in MODES:
            cells.append(f"{r[f'{m}_g0.0']:>+8.2f}{r[f'{m}_g0.5']:>+9.2f}"
                         f"{r[f'{m}_g1.0']:>+8.2f}{r[f'{m}_gain']:>+7.2f}")
        A(f"{r['artifact']:>16s} {100*r['cv_rr']:>5.1f}% | " + " | ".join(cells))
    A("")
    A("D = gamma=1 minus gamma=0. Positive means phase warping helps.")
    A("")
    A("-" * 78)
    A("2. REAL DATA (no ground truth)")
    A("-" * 78)
    A("Variance reduction inside the beat window (dB): how much signal energy the")
    A("canceller actually removes there. Coordinate free, so it privileges neither")
    A("gamma, and computed on the signal ilc_clean returns.")
    A("")
    A("HELD OUT: the non-causal template is built on half the beats and applied to")
    A("the other half. Two traps were live here and both are avoided. Scored in")
    A("sample, gamma=0 looks perfect because a template subtracted from the beats")
    A("that built it annihilates their average identically. And scoring the residual")
    A("by its R-triggered average over ALL beats also reads zero, because the two")
    A("half-templates average back to the full one. The one-beat-delay arm needs no")
    A("split: it subtracts before it updates.")
    A("")
    A("main lobe = -50..+150 ms, late = +200..+500 ms (the template ends at +500 ms).")
    A("")
    for r in d["real"]:
        A(f"{r['label']}")
        A(f"    {r['n_beats']} learnable beats, RR {r['rr_mean_s']:.3f} s, "
          f"CV_RR {100*r['cv_rr_robust']:.1f}%, EEG SD {r['sd_uV']:.1f} uV")
        A(f"    raw R-locked deflection: lobe {r['raw_lobe_rms_uV']:.2f} uV "
          f"({r['raw_lobe_db_over_floor']:.1f} dB over the shift null), "
          f"late {r['raw_late_rms_uV']:.2f} uV "
          f"({r['raw_late_db_over_floor']:.1f} dB)")
        A(f"    {'mode':>14s} {'band':>6s} | " + "".join(f"{'g='+str(g):>10s}" for g in GAMMAS)
          + f"{'D(1-0)':>9s}")
        for m in MODES:
            for b in ("lobe", "late"):
                vals = [r[f"{m}_g{g}_{b}_removed_db"] for g in GAMMAS]
                A(f"    {MODE_LABEL[m]:>14s} {b:>6s} | "
                  + "".join(f"{v:>10.2f}" for v in vals)
                  + f"{vals[-1]-vals[0]:>+9.2f}")
        A("")
    A("Removed (dB) is 10*log10(input energy / residual energy) in the band;")
    A("bigger is better. D is gamma=1 minus gamma=0, positive means warping helps.")
    A("")
    A("-" * 78)
    A("3. READING")
    A("-" * 78)
    A("")
    syn_t = [r for r in d["synthetic"] if r["artifact"] == "time-locked"]
    syn_p = [r for r in d["synthetic"] if r["artifact"] == "phase-stretched"]
    A(f"* Synthetic, time-locked artifact: warping COSTS "
      f"{min(r['noncausal_gain'] for r in syn_t):+.2f} to "
      f"{max(r['noncausal_gain'] for r in syn_t):+.2f} dB non-causal,")
    A(f"  and the cost grows with RR variability -- warping misaligns an artifact")
    A(f"  that was already aligned.")
    A(f"* Synthetic, phase-stretched artifact: warping BUYS "
      f"{min(r['noncausal_gain'] for r in syn_p):+.2f} to "
      f"{max(r['noncausal_gain'] for r in syn_p):+.2f} dB,")
    A(f"  again growing with RR variability. The sign of D is set entirely by which")
    A(f"  regime the artifact is in, not by the canceller.")
    A(f"* One beat of latency costs about "
      f"{np.mean([r['noncausal_g0.0']-r['buffered_g0.0'] for r in d['synthetic']]):.1f} dB "
      f"against the batch average, and that")
    A(f"  cost is essentially independent of gamma: the two choices are separable.")
    A("")
    A("* Real data: the sign is NOT uniform, and it is predictable. Sorting the")
    A("  records by where their R-locked energy sits:")
    A("")
    A(f"    {'record':>26s} {'late/lobe':>10s} {'D lobe':>8s} {'D late':>8s}")
    rows = sorted(d["real"], key=lambda r: r["raw_late_rms_uV"] / max(r["raw_lobe_rms_uV"], 1e-9))
    for r in rows:
        ratio = r["raw_late_rms_uV"] / max(r["raw_lobe_rms_uV"], 1e-9)
        dl = r["noncausal_g1.0_lobe_removed_db"] - r["noncausal_g0.0_lobe_removed_db"]
        dt = r["noncausal_g1.0_late_removed_db"] - r["noncausal_g0.0_late_removed_db"]
        A(f"    {r['label'].split(' EEG')[0]:>26s} {ratio:>10.2f} {dl:>+8.2f} {dt:>+8.2f}")
    A("")
    A("  The three lobe-dominated records lose a little from warping (D between")
    A("  -0.19 and 0.00 dB). The one late-dominated record gains clearly in its late")
    A("  window (+0.59 dB non-causal, +0.49 dB one-beat delay). That is the")
    A("  phase-stretched regime appearing in real data.")
    A("")
    A("  The mechanism is simple. Stretching the template by T_k/Tbar displaces bin")
    A("  o by (o-ra)*(T_k/Tbar - 1) samples. Near the R-peak that displacement is")
    A("  ~0, so warping cannot help there and can only blur a sharp, time-locked")
    A("  QRS deflection. Late in the cycle the displacement is large, which is where")
    A("  a genuinely phase-locked component gets realigned. So the late/lobe energy")
    A("  ratio predicts the sign, and it is measurable before choosing gamma.")
    A("")
    A("* Bottom line for Figure 1: it is drawn at gamma=0 on a lobe-dominated")
    A("  record, which is the regime where gamma=0 is right. Adding the warp would")
    A("  change it by -0.19 dB. Figure 1 is not leaving performance on the table.")
    A("")
    return "\n".join(L)


def make_figure(d, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))

    # (a) synthetic: advantage of warping vs RR variability
    for arti, ls in (("time-locked", "--"), ("phase-stretched", "-")):
        rows = [r for r in d["synthetic"] if r["artifact"] == arti]
        rows.sort(key=lambda r: r["cv_rr"])
        x = [100 * r["cv_rr"] for r in rows]
        for m, col in (("noncausal", "tab:blue"), ("buffered", "tab:orange")):
            ax[0].plot(x, [r[f"{m}_gain"] for r in rows], ls, color=col, marker="o",
                       ms=3, lw=1.2, label=f"{MODE_LABEL[m]}, {arti}")
    ax[0].axhline(0.0, color="0.5", lw=0.8)
    ax[0].set_xlabel(r"RR variability $\mathrm{CV}_{RR}$ (%)")
    ax[0].set_ylabel(r"$\Delta$ SNR improvement (dB)" "\n" r"$\gamma=1$ minus $\gamma=0$")
    ax[0].set_title("(a) synthetic, ground truth", fontsize=9)
    ax[0].legend(fontsize=6, frameon=False, loc="upper left")
    ax[0].tick_params(labelsize=7)

    # (b) real: R-locked energy removed, per arm
    reals = d["real"]
    labels = [r["label"].split(" EEG")[0].replace("/ses-01", "") for r in reals]
    x = np.arange(len(reals))
    w = 0.2
    for i, (m, b, col) in enumerate([
            ("noncausal", "lobe", "tab:blue"), ("noncausal", "late", "tab:cyan"),
            ("buffered", "lobe", "tab:orange"), ("buffered", "late", "tab:red")]):
        d0 = [r[f"{m}_g0.0_{b}_removed_db"] for r in reals]
        d1 = [r[f"{m}_g1.0_{b}_removed_db"] for r in reals]
        ax[1].bar(x + (i - 1.5) * w, np.array(d1) - np.array(d0), w, color=col,
                  label=f"{MODE_LABEL[m]}, {b}")
    ax[1].axhline(0.0, color="0.5", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=6, rotation=15, ha="right")
    ax[1].set_ylabel(r"$\Delta$ variance reduction (dB)")
    ax[1].set_title("(b) real recordings, held out", fontsize=9)
    ax[1].legend(fontsize=6, frameon=False, loc="lower left")
    ax[1].tick_params(labelsize=7)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def tex_tables(d, paper_dir: Path):
    syn = [
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{Synthetic sweep with ground truth: SNR improvement over the contaminated input",
        r"(dB), measured against the true neural signal, mean of three seeds at 180~s each.",
        r"$\Delta$ is $\gamma{=}1$ minus $\gamma{=}0$.}",
        r"\label{tab:synth}",
        r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
        r"artifact & mode & $\mathrm{CV}_{RR}$ & $\gamma{=}0$ & $\gamma{=}1$ & $\Delta$ \\",
        r"\midrule",
    ]
    prev = None
    for r in d["synthetic"]:
        for m in MODES:
            head = r["artifact"] if (r["artifact"], m) != prev and m == MODES[0] else ""
            syn.append(
                f"{head} & {MODE_LABEL[m]} & ${100*r['cv_rr']:.1f}\\%$ & "
                f"${r[f'{m}_g0.0']:+.2f}$ & ${r[f'{m}_g1.0']:+.2f}$ & "
                f"${r[f'{m}_gain']:+.2f}$ \\\\")
        prev = (r["artifact"], MODES[0])
        syn.append(r"\addlinespace[2pt]")
    syn += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (paper_dir / "tab_phase_swap_synth.tex").write_text("\n".join(syn))

    rea = [
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{Real recordings, held out: variance reduction inside the beat window (dB).",
        r"Main lobe is $-50$ to $+150$~ms, late is $+200$ to $+500$~ms. $\Delta$ is",
        r"$\gamma{=}1$ minus $\gamma{=}0$; negative means the warp removes less.}",
        r"\label{tab:real}",
        r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
        r"record & mode & band & $\gamma{=}0$ & $\gamma{=}1$ & $\Delta$ \\",
        r"\midrule",
    ]
    for r in d["real"]:
        lab = r["label"].split(" EEG")[0].replace("/ses-01", "").replace("_", r"\_")
        first = True
        for m in MODES:
            for b in ("lobe", "late"):
                v0 = r[f"{m}_g0.0_{b}_removed_db"]
                v1 = r[f"{m}_g1.0_{b}_removed_db"]
                rea.append(f"{lab if first else ''} & {MODE_LABEL[m]} & {b} & "
                           f"${v0:+.2f}$ & ${v1:+.2f}$ & ${v1-v0:+.2f}$ \\\\")
                first = False
        rea.append(r"\addlinespace[2pt]")
    rea += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (paper_dir / "tab_phase_swap_real.tex").write_text("\n".join(rea))
    print(f"wrote {paper_dir}/tab_phase_swap_{{synth,real}}.tex")


def fill_tex(d, paper_dir: Path):
    tpl = paper_dir / "phase_swap.tex.tokens"
    if not tpl.exists():
        return
    syn_t = [r for r in d["synthetic"] if r["artifact"] == "time-locked"]
    syn_p = [r for r in d["synthetic"] if r["artifact"] == "phase-stretched"]
    vals = {
        "SYNCOST": f"{abs(min(r['noncausal_gain'] for r in syn_t)):.1f}",
        "SYNGAIN": f"{max(r['noncausal_gain'] for r in syn_p):.1f}",
        "LATCOST": f"{np.mean([r['noncausal_g0.0'] - r['buffered_g0.0'] for r in d['synthetic']]):.1f}",
    }
    txt = tpl.read_text()
    for k, v in vals.items():
        txt = txt.replace(f"@@{k}@@", v)
    (paper_dir / "phase_swap.tex").write_text(txt)
    left = sum("@@" in ln for ln in txt.splitlines())
    print(f"wrote {paper_dir}/phase_swap.tex ({left} unfilled)", vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../outputs/phase_swap.json")
    ap.add_argument("--txt", default="../paper/phase_swap.txt")
    ap.add_argument("--fig", default="../paper/fig_real/fig_phase_swap.pdf")
    ap.add_argument("--paper-dir", default="../paper")
    a = ap.parse_args()
    d = json.loads(Path(a.json).read_text())
    Path(a.txt).write_text(txt_tables(d))
    print(f"wrote {a.txt}")
    make_figure(d, a.fig)
    tex_tables(d, Path(a.paper_dir))
    fill_tex(d, Path(a.paper_dir))


if __name__ == "__main__":
    main()
