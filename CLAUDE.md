# Project: Real-Time Cardiac Field Artifact (CFA) Removal for Low-Channel Ear-EEG

This file summarizes a long design conversation (Claude.ai) that led to a working
prototype and an ICASSP paper draft. It is a technical brief, not a transcript —
read it once, then work from the actual files it references.

## 1. The problem and the architecture

Target: an 8-channel (or fewer) near-ear EEG earable contaminated by the cardiac
field artifact (CFA) — the volume-conducted electrical potential of the heart,
distinct from the ballistocardiogram/pulse artifact studied in EEG-fMRI. Goal:
real-time, low-power, causal removal, with an ECG (or watch-derived heartbeat)
reference available.

**Notation used throughout the code and paper (all plain Latin letters):**
- `y(t)` measured EEG, `s(t)` true neural signal, `d(t)` cardiac disturbance,
  `n(t)` noise, `r(t)` ECG/heartbeat reference, `u(t)` cardiac source, `v(t)`
  reference noise.
- `N(·)` static electrode nonlinearity, `H` linear volume-conduction channel.
- `t_k` k-th R-peak instant, `RR_k` k-th RR interval, `p(t)` cardiac phase in [0,1).
- State-space residual canceller (control-custom): `x[n]=A x[n-1]+B w[n]`,
  `d_r(t)=C[n] x[n]+n(t)`, Kalman/RLS gain `K[n]`, covariance `P[n]`.
- ILC template `m(o)` indexed by **time-since-R** offset `o` (not normalized
  phase — systole is time-locked to R, diastole absorbs RR variation).
- Learn-gate `l[k]` (arrhythmia supervisor), amplitude limiter `β·pk(t)`
  (reuses the peak tracker), leaky-NRLS factor `δ`.

**Pipeline (original, full version):** two-level min/max envelope tracker →
PI phase-locked loop (PLL) with arrhythmia supervisor → ILC/repetitive
R-anchored template (buffered ~1-beat-latency or causal/zero-latency mode) →
Hammerstein basis expansion of the reference → NRLS state-space residual
canceller.

## 2. CRITICAL PIVOT — read this before editing the paper

Real-data test: coherence between the ear-EEG and the ECG reference was
**measured near zero, even in R-peak-locked windows.** This means:

- The cardiac artifact at the ear is **phase-locked but waveform-decorrelated**
  from the reference (different volume-conduction projection than the ECG lead
  sees).
- **Reference-driven stages (Hammerstein nonlinearity model, NRLS residual
  canceller) cannot contribute on this data** — they need the artifact to be
  linearly (or nonlinearly-but-regressor-) predictable from `r(t)`; coherence
  ≈ 0 means there's nothing to regress against.
- **ILC/synchronous-averaging still works**, because it only needs R-peak
  *timing*, never ECG *waveform*. This is exactly HDD RRO/NRRO cancellation:
  synchronous average over "revolutions" (heartbeats) extracts the repeatable
  disturbance; non-repeatable neural signal + noise average out.
- **This makes a watch/wrist heartbeat source usable as the reference**, even
  though a separate consumer device can't give a sample-aligned *waveform*
  reference (unknown cross-device latency/jitter) — because only *timing*
  crosses the device boundary, and timing is all ILC needs.

## 3. Current paper scope decision

Given the finding above, the ICASSP draft is being **narrowed** to:
**PLL + min/max tracker + ILC only** (cardiac artifact, ear-EEG, watch-timing
feasible). The Hammerstein/NRLS material is being **cut or demoted to future
work** in this paper — do not defend it as a contribution here; the coherence
result is the reason, and it should be stated as a motivating negative result
("we tested reference-driven cancellation; near-zero R-locked coherence shows
it cannot help; hence a timing-only ILC/repetitive approach").

**Separate, later, speculative paper (not this one):** unify ILC, Hammerstein,
and NRLS as instances of one **disturbance-observer** framework (choose a
reference → choose an observer structure → cancel what's reference-explainable),
and extend to **motion artifact via IMU (accelerometer/gyro) reference** — motion
should be IMU-coherent (physical cause, unlike the heart's differently-projected
field), which is exactly the regime where the reference-driven machinery
belongs. **Gate on measuring IMU-vs-motion-artifact coherence before investing
in this** — same diagnostic as the cardiac coherence check. This is a guess,
not yet verified.

## 4. File inventory (from this session — confirm these are present in this repo)

- `cfa_tool.py` — consolidated CLI: `gen-ecg`, `gen-eeg`, `clean` subcommands;
  `clean --mode {noncausal,buffered,causal}` and per-block toggles
  (`--phase-tracker`, `--override`, `--ilc`, `--nonlinear`, `--rls`).
- `ecg_phase_demo.py` — standalone streaming phase estimator demo (two-level
  gate + PI-PLL + arrhythmia supervisor), diagnostic plots.
- `cfa_realtime_demo.py` — single-pass causal pipeline demo, buffered vs
  causal R-anchoring, per-region (R-spike/P-wave/tail) residual analysis.
- `rro_fig.py` — generates the RRO/NRRO illustration: ECG beats → EEG with
  R-anchored template table → synchronous average recovers the artifact.
  This is the figure/logic to re-run on the real dataset to confirm a
  genuine repeatable template exists there (not just in synthetic data).
- `generate_figures.py` — regenerates the paper's Fig. 1 (convergence +
  spectrum) and Fig. 2 (time-domain PQRST) as vector PDFs from `cfa_tool`.
- `paper.tex` — ICASSP draft, IEEEtran conference class, ~5 pages compiled
  (content 1–4/5, references spill to last page). Contains a TikZ system
  diagram (Fig. 1, two-column `figure*`) with blocks labeled by equation
  number, and a "Safeguards against cancellation of neural signal" subsection
  (phase-lock argument, reference-orthogonality argument, amplitude limiter
  Eq. 16, leaky-NRLS Eq. 17) — **this subsection's reasoning still applies
  even in the narrowed scope for the ILC limiter; the NRLS-specific parts
  will need to move if NRLS is cut.**
- Compiled `paper.pdf` / `ICASSP_draft_latex.pdf` and earlier docx versions
  exist from iterative revisions — `paper.tex` is the current source of truth.

## 5. Immediate next steps (in rough priority order)

1. Run `rro_fig.py`'s methodology (synchronous R-locked averaging) on the
   **real** dataset to confirm a genuine, consistent template emerges (not
   flat) — this is the ground-truth-free evidence figure for the paper and
   the sanity check that cardiac artifact is even worth removing there.
2. Rewrite `paper.tex` Sections IV–V to the narrowed scope: keep signal model,
   phase estimator, ILC/repetitive template, the limiter safeguard; cut or
   move to "Future Work" the Hammerstein and NRLS subsections and their
   equations/table rows; update the TikZ diagram to match (remove the
   Basis/NRLS blocks or mark them as future work).
3. Add the coherence result itself as a figure/table (coherence vs frequency,
   or R-locked coherence value) — it is now a key motivating result, not a
   footnote.
4. Reconfirm the paper still fits the 4-content-page + references-page ICASSP
   norm after removing material (it was ~5 pages with everything included;
   removing NRLS/Hammerstein should recover room).
5. All literature citations in `paper.tex` are **unverified** (built from
   model knowledge during a session where live search was unavailable) —
   run a real search/DOI check before submission. Do not add new citations
   from memory without flagging them the same way.

## 6. Working conventions

- Author is a controls/DSP engineer (PhD, nonlinear control; prior HDD servo
  firmware background) — comfortable with, and prefers, precise
  control-theoretic framing (state-space, ILC/repetitive control, PLL,
  observer terminology). No need to explain basic control concepts.
- Prefers concrete measurements over reassurance — when a design choice's
  value is uncertain (e.g., nonlinearity significance, motion-IMU coherence),
  the right move is to propose the specific cheap diagnostic, not argue from
  intuition.
- Keep equation notation consistent with Section 1 above across any new code
  or paper edits — do not reintroduce Greek-letter or script notation the
  paper deliberately moved away from.
