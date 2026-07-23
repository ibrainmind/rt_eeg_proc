#!/usr/bin/env python3
"""
cfa_tool.py  --  Cardiac-Field-Artifact cleanup workbench
=========================================================
A single controllable tool wrapping everything we prototyped:

  synthesis      gen-ecg / gen-eeg   (RSA-modulated ECG, EEG + Hammerstein CFA)
  processing     clean               with three run modes and per-block toggles

Run modes (clean --mode):
  noncausal   offline ceiling: template = average over ALL beats, batch ridge RLS
  buffered    ~1-beat latency: cancellation anchored to the DETECTED R (full R-spike removal)
  causal      zero latency:    cancellation anchored to the PREDICTED R (R-spike limited)

Per-block toggles (clean, all default ON; use --no-XXX to disable):
  --phase-tracker   PLL rate smoothing + phase prediction (off = raw per-beat intervals)
  --override        arrhythmia supervisor (off = every beat trusted -> anomalies corrupt)
  --ilc             repetitive/ILC template subtraction
  --nonlinear       Hammerstein (Chebyshev) basis on the RLS reference (off = linear reference)
  --rls             linear residual canceller

Examples:
  python cfa_tool.py gen-ecg --duration 20 --out ecg.png
  python cfa_tool.py gen-eeg --duration 30 --out eeg.png
  python cfa_tool.py clean --mode buffered --out clean.png
  python cfa_tool.py clean --mode causal --no-rls
  python cfa_tool.py clean --mode buffered --no-ilc --no-nonlinear
  python cfa_tool.py clean --mode buffered --no-override      # watch anomalies corrupt the template
"""
import argparse
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# 1. SIGNAL SYNTHESIS
# ============================================================================
def _pqrst(fs, wide=False):
    tt = np.arange(-0.30, 0.50, 1.0 / fs)
    g = lambda c, a, w: a * np.exp(-0.5 * ((tt - c) / w) ** 2)
    if wide:                                    # PVC-like: no P, wide odd QRS
        return g(0.0, 1.1, 0.028) + g(0.05, -0.5, 0.030)
    return (g(-0.20, 0.10, 0.025) + g(-0.025, -0.15, 0.008) + g(0.0, 1.0, 0.010)
            + g(0.025, -0.25, 0.010) + g(0.18, 0.30, 0.040))

def gen_sources(fs, dur, seed, anomalies=True, rr0=0.85, f_resp=0.25, rsa=0.08):
    """Cardiac source d(t), R-peak times, beat types."""
    rng = np.random.default_rng(seed)
    N = int(fs * dur)
    beat_t, btype, tk = [], [], 1.0
    while tk < dur - 1.0:
        rr = rr0 * (1 + rsa * np.sin(2 * np.pi * f_resp * tk)) + rng.normal(0, 0.008)
        beat_t.append(tk); btype.append("normal"); tk += rr
    beat_t = np.array(beat_t)
    if anomalies and dur > 30:
        ip = np.searchsorted(beat_t, min(dur * 0.45, dur - 10))
        beat_t = np.insert(beat_t, ip, beat_t[ip] - 0.35 * rr0); btype.insert(ip, "premature")
        idp = np.searchsorted(beat_t, min(dur * 0.65, dur - 6))
        beat_t = np.delete(beat_t, idp); del btype[idp]
    d = np.zeros(N); r0 = int(0.30 * fs)
    for bt, ty in zip(beat_t, btype):
        w = _pqrst(fs, wide=(ty == "premature")); idx = int(round(bt * fs)); lo = idx - r0
        a0 = max(0, -lo); a1 = min(len(w), N - lo)
        if a1 > a0:
            d[max(lo, 0):max(lo, 0) + (a1 - a0)] += w[a0:a1]
    return d, beat_t, btype, rng

def gen_ecg(fs, dur, seed, anomalies=True):
    d, beat_t, btype, rng = gen_sources(fs, dur, seed, anomalies)
    N = len(d); t = np.arange(N) / fs
    r = d + 0.15 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.02, N)  # + wander + noise
    return t, r, beat_t, btype

def gen_eeg(fs, dur, seed, anomalies=True, drift=0.20, artifact_gain=3.0):
    d, beat_t, btype, rng = gen_sources(fs, dur, seed, anomalies)
    N = len(d); t = np.arange(N) / fs
    f_nl = lambda x: np.tanh(1.6 * x) + 0.18 * x ** 2         # electrode nonlinearity
    h = np.array([0.05, 0.20, 0.35, 0.25, 0.10, 0.05]); h /= h.sum()
    dr = 1.0 + drift * np.sin(2 * np.pi * 0.02 * t)
    cfa = dr * signal.lfilter(h, 1.0, f_nl(d)) * artifact_gain
    white = rng.normal(0, 1, N)
    pink = signal.lfilter([0.049922, -0.095993, 0.050612, -0.004408],
                          [1, -2.494956, 2.017265, -0.522189], white); pink /= np.std(pink)
    alpha = (0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t)) * np.sin(2 * np.pi * 10 * t)
    s = 0.7 * pink + 0.6 * alpha
    y = s + cfa + rng.normal(0, 0.10, N)
    r = d + rng.normal(0, 0.03, N)
    return t, y, s, cfa, r, beat_t, btype

# ============================================================================
# 2. PHASE ESTIMATOR  (streaming: gate + optional PLL + optional supervisor)
# ============================================================================
def phase_estimator(r, fs, use_pll=True, use_override=True, rr0=0.85):
    N = len(r)
    sos = signal.butter(2, [8, 20], btype="band", fs=fs, output="sos")
    emph = np.abs(signal.sosfilt(sos, r))
    DEC = np.exp(-1.0 / (0.35 * fs)); AB = 1.0 / (1.5 * fs)
    THR_HI, THR_LO = 0.40, 0.15; REFR = int(0.25 * fs); WARM = int(2.0 * fs)
    peak = 1e-6; base = 0.0; armed = False; refr = 0
    run_max = -1e9; rmi = 0; phi_pk = 0.0
    phi = 0.0; RR_hat = rr0; omega = 1.0 / (RR_hat * fs); last = None
    fids = []; phi_log = np.zeros(N); rr_log = np.zeros(N)
    wrap = lambda p: ((p + 0.5) % 1.0) - 0.5
    for n in range(N):
        e = emph[n]
        peak = e if e > peak else peak * DEC
        base = base + AB * (e - base)
        gate = (e - base) / (peak - base + 1e-9)
        phi = (phi + omega) % 1.0
        phi_log[n] = phi; rr_log[n] = RR_hat
        if refr > 0: refr -= 1
        if not armed and gate > THR_HI and refr == 0 and n > WARM:
            armed = True; run_max = -1e9
        if armed:
            if r[n] > run_max: run_max = r[n]; rmi = n; phi_pk = phi
            if gate < THR_LO:
                armed = False; refr = REFR; fid = rmi; rr_pre = RR_hat
                rr_meas = RR_hat if last is None else (fid - last) / fs
                ratio = rr_meas / RR_hat; perr = wrap(phi_pk)
                if use_override and ratio < 0.70:
                    ty = "premature"; phi = 0.0; lg = False
                elif use_override and ratio > 1.50:
                    ty = "missed"; phi = 0.0; lg = False
                else:
                    ty = "normal"; lg = True
                    if use_pll:
                        phi = (phi - 0.30 * perr) % 1.0
                        RR_hat = RR_hat + 0.30 * (rr_meas - RR_hat)
                    else:                                    # no PLL: raw interval, hard reseed
                        phi = 0.0
                        RR_hat = rr_meas if last is not None else RR_hat
                    omega = 1.0 / (RR_hat * fs)
                fids.append((fid, ty, lg)); last = fid
    return {"fids": fids, "phi": phi_log, "rrhat": rr_log}

# ============================================================================
# 3. ILC  (three anchoring strategies) + RLS residual
# ============================================================================
def _template_dims(fs):
    ra = int(0.30 * fs); Lw = ra + int(0.50 * fs)          # [-0.3s .. +0.5s] around R
    return ra, Lw

def ilc_clean(y, fs, est, mode, use_ilc, MU=0.06):
    """Return (clean_after_ilc, template). Honors run mode for anchoring."""
    N = len(y); ra, Lw = _template_dims(fs)
    fids = est["fids"]; phi = est["phi"]; rrh = est["rrhat"]
    Rs = np.array([f[0] for f in fids]); Rlg = [f[2] for f in fids]
    template = np.zeros(Lw); out = y.copy()
    if not use_ilc:
        return out, template

    if mode == "noncausal":                                 # acausal average of all learnable beats
        cnt = np.zeros(Lw)
        for fid, ty, lg in fids:
            if not lg: continue
            for o in range(Lw):
                k = fid - ra + o
                if 0 <= k < N: template[o] += y[k]; cnt[o] += 1
        cnt[cnt == 0] = 1; template /= cnt
        for n in range(N):
            j = np.searchsorted(Rs, n); idx = -1; bd = 1e18
            for jj in (j - 1, j):
                if 0 <= jj < len(Rs):
                    o = n - Rs[jj] + ra
                    if 0 <= o < Lw and abs(n - Rs[jj]) < bd: bd = abs(n - Rs[jj]); idx = o
            out[n] = y[n] - template[idx] if idx >= 0 else y[n]
        return out, template

    for n in range(N):                                      # streaming: buffered or causal
        idx = -1
        if mode == "buffered":                              # anchor to nearest DETECTED R
            j = np.searchsorted(Rs, n); bd = 1e18; lg = True
            for jj in (j - 1, j):
                if 0 <= jj < len(Rs):
                    o = n - Rs[jj] + ra
                    if 0 <= o < Lw and abs(n - Rs[jj]) < bd:
                        bd = abs(n - Rs[jj]); idx = o; lg = Rlg[jj]
        else:                                               # causal: anchor to PREDICTED R (phi,rrhat)
            tau = phi[n] * rrh[n]; ttn = (1 - phi[n]) * rrh[n]
            op = ra + int(round(tau * fs)); opre = ra - int(round(ttn * fs))
            if tau <= 0.50 and 0 <= op < Lw: idx = op
            elif ttn <= 0.30 and 0 <= opre < ra: idx = opre
            lg = True
        if idx >= 0:
            e1 = y[n] - template[idx]
            if lg: template[idx] += MU * e1
            out[n] = e1
        else:
            out[n] = y[n]
    return out, template

def rls_clean(e1, r, fs, mode, use_rls, use_nonlinear, M=8, lam=0.9998, ridge=0.3):
    N = len(e1)
    if not use_rls:
        return e1.copy()
    orders = range(1, 5) if use_nonlinear else range(1, 2)  # Chebyshev T1..T4, or linear only
    rn = r / (np.max(np.abs(r)) + 1e-9)
    Phi = np.vstack([np.polynomial.chebyshev.Chebyshev.basis(k)(rn) for k in orders])
    Phi = Phi - Phi.mean(axis=1, keepdims=True)             # zero-mean -> no DC collinearity
    K = Phi.shape[0]; Dd = K * M
    X = np.zeros((Dd, N))
    for k in range(K):
        for m in range(M):
            X[k * M + m, m:] = Phi[k, :N - m]
    if mode == "noncausal":                                 # batch ridge
        A = X @ X.T + ridge * np.trace(X @ X.T) / Dd * np.eye(Dd)
        w = np.linalg.solve(A, X @ e1)
        return e1 - X.T @ w
    w = np.zeros(Dd); P = np.eye(Dd) * ridge; out = np.zeros(N)  # streaming RLS
    for n in range(N):
        xn = X[:, n]; err = e1[n] - w @ xn
        Px = P @ xn; g = Px / (lam + xn @ Px); w += g * err; P = (P - np.outer(g, Px)) / lam
        out[n] = err
    return out

# ============================================================================
# 4. METRICS + PLOTS
# ============================================================================
def snr_db(s, est, mask):
    return 10 * np.log10(np.sum(s[mask] ** 2) / np.sum((s[mask] - est[mask]) ** 2))

def plot_ecg(t, r, beat_t, out):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, r, lw=0.6, color="0.3")
    for bt in beat_t:
        ax.axvline(bt, color="C2", lw=0.6, alpha=0.5)
    ax.set_xlim(0, min(t[-1], 20)); ax.set_title("Synthetic ECG reference (green = R peaks)")
    ax.set_xlabel("time (s)"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()

def plot_eeg(t, y, s, cfa, out):
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t, cfa, lw=0.6, color="C3"); ax[0].set_title("Cardiac field artifact (nonlinear + volume conduction)")
    ax[1].plot(t, s, lw=0.6, color="C0"); ax[1].set_title("True neural signal (pink + alpha)")
    ax[2].plot(t, y, lw=0.6, color="0.3"); ax[2].set_title("Contaminated EEG = neural + CFA + noise")
    ax[2].set_xlabel("time (s)"); ax[2].set_xlim(0, min(t[-1], 20))
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()

def plot_clean(t, y, s, cfa, clean_ilc, clean, template, fs, args, out):
    N = len(t); ra, Lw = _template_dims(fs); tsr = (np.arange(Lw) - ra) / fs
    ss = t > (t[-1] * 0.5)
    def wsnr(est, win=int(8 * fs)):
        c, o = [], []
        for a in range(0, N - win, win // 2):
            b = a + win; o.append(snr_db(s, est, slice(a, b))); c.append((a + b) / 2 / fs)
        return np.array(c), np.array(o)
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    tc, sr = wsnr(y); _, sa = wsnr(clean_ilc); _, sb = wsnr(clean)
    ax[0, 0].plot(tc, sr, label="raw"); ax[0, 0].plot(tc, sa, label="+ILC"); ax[0, 0].plot(tc, sb, label="+ILC+RLS")
    ax[0, 0].set_title("Convergence (8s-window SNR)"); ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("dB"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(tsr, template, label="learned template")
    ax[0, 1].axvspan(-0.25, -0.05, color="orange", alpha=0.12, label="pre-R"); ax[0, 1].axvline(0, color="0.5", lw=0.6)
    ax[0, 1].set_title("ILC template (time since R)"); ax[0, 1].set_xlabel("s"); ax[0, 1].legend(fontsize=8)
    z0, z1 = int(min(t[-1] * 0.8, t[-1] - 3) * fs), int(min(t[-1] * 0.8 + 3, t[-1]) * fs)
    ax[1, 0].plot(t[z0:z1], y[z0:z1], lw=0.6, label="contaminated")
    ax[1, 0].plot(t[z0:z1], clean[z0:z1], lw=0.8, label="cleaned")
    ax[1, 0].plot(t[z0:z1], s[z0:z1], lw=0.8, alpha=0.7, label="true")
    ax[1, 0].set_title("Cleaned vs truth (zoom)"); ax[1, 0].set_xlabel("time (s)"); ax[1, 0].legend(fontsize=8)
    fa, Py = signal.welch(y[ss], fs, nperseg=1024); _, Pc = signal.welch(clean[ss], fs, nperseg=1024); _, Ps = signal.welch(s[ss], fs, nperseg=1024)
    ax[1, 1].semilogy(fa, Py, alpha=0.8, label="contaminated"); ax[1, 1].semilogy(fa, Pc, lw=2, label="cleaned"); ax[1, 1].semilogy(fa, Ps, ":", lw=2, label="true")
    ax[1, 1].axvspan(9, 11, color="green", alpha=0.12); ax[1, 1].set_xlim(0, 45)
    ax[1, 1].set_title("Spectrum"); ax[1, 1].set_xlabel("Hz"); ax[1, 1].legend(fontsize=8)
    fig.suptitle("mode=%s | phase=%s override=%s ilc=%s nonlinear=%s rls=%s"
                 % (args.mode, args.phase_tracker, args.override, args.ilc, args.nonlinear, args.rls),
                 fontsize=11)
    for a in ax.flat: a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()

# ============================================================================
# 5. CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(prog="cfa_tool", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    def add_common(sp):
        sp.add_argument("--fs", type=float, default=250.0)
        sp.add_argument("--duration", type=float, default=60.0)
        sp.add_argument("--seed", type=int, default=11)
        sp.add_argument("--anomalies", action=argparse.BooleanOptionalAction, default=True)
        sp.add_argument("--out", default=None, help="output PNG path")
    sp = sub.add_parser("gen-ecg", help="generate & plot synthetic ECG"); add_common(sp)
    sp = sub.add_parser("gen-eeg", help="generate & plot synthetic EEG + CFA"); add_common(sp)
    sp.add_argument("--drift", type=float, default=0.20); sp.add_argument("--artifact-gain", type=float, default=3.0)
    sp = sub.add_parser("clean", help="run the cleanup pipeline"); add_common(sp)
    sp.add_argument("--mode", choices=["noncausal", "buffered", "causal"], default="buffered")
    sp.add_argument("--phase-tracker", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--override", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--ilc", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--nonlinear", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--rls", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--drift", type=float, default=0.20); sp.add_argument("--artifact-gain", type=float, default=3.0)
    args = p.parse_args()

    if args.cmd == "gen-ecg":
        t, r, beat_t, btype = gen_ecg(args.fs, args.duration, args.seed, args.anomalies)
        print("ECG: %.0f s, %d beats (%d premature, %d after-drop)" %
              (args.duration, len(beat_t), btype.count("premature"), btype.count("missed")))
        out = args.out or "ecg.png"; plot_ecg(t, r, beat_t, out); print("wrote", out)

    elif args.cmd == "gen-eeg":
        t, y, s, cfa, r, beat_t, btype = gen_eeg(args.fs, args.duration, args.seed, args.anomalies,
                                                 args.drift, args.artifact_gain)
        ss = t > (t[-1] * 0.5)
        print("EEG: input artifact SNR (s vs y) = %.2f dB over %d beats" %
              (snr_db(s, y, ss), len(beat_t)))
        out = args.out or "eeg.png"; plot_eeg(t, y, s, cfa, out); print("wrote", out)

    elif args.cmd == "clean":
        t, y, s, cfa, r, beat_t, btype = gen_eeg(args.fs, args.duration, args.seed, args.anomalies,
                                                 args.drift, args.artifact_gain)
        est = phase_estimator(r, args.fs, args.phase_tracker, args.override)
        clean_ilc, template = ilc_clean(y, args.fs, est, args.mode, args.ilc)
        clean = rls_clean(clean_ilc, r, args.fs, args.mode, args.rls, args.nonlinear)
        ss = t > (t[-1] * 0.5)
        nP = sum(1 for f in est["fids"] if f[1] == "premature")
        nM = sum(1 for f in est["fids"] if f[1] == "missed")
        print("mode=%s  phase=%s override=%s ilc=%s nonlinear=%s rls=%s"
              % (args.mode, args.phase_tracker, args.override, args.ilc, args.nonlinear, args.rls))
        print("  detected beats: %d  (premature %d, missed %d)" % (len(est["fids"]), nP, nM))
        print("  raw y            : %6.2f dB" % snr_db(s, y, ss))
        print("  after ILC        : %6.2f dB" % snr_db(s, clean_ilc, ss))
        print("  after ILC + RLS  : %6.2f dB" % snr_db(s, clean, ss))
        if args.out:
            plot_clean(t, y, s, cfa, clean_ilc, clean, template, args.fs, args, args.out)
            print("wrote", args.out)

if __name__ == "__main__":
    main()