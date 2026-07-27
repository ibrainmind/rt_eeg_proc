#!/usr/bin/env python3
"""
ekg_ilc_rt.py  --  Cardiac-Field-Artifact cleanup workbench
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
    python ekg_ilc_rt.py gen-ecg --duration 20 --out ecg.png
    python ekg_ilc_rt.py gen-eeg --duration 30 --out eeg.png
    python ekg_ilc_rt.py clean --mode buffered --out clean.png
    python ekg_ilc_rt.py clean --mode causal --no-rls
    python ekg_ilc_rt.py clean --mode buffered --no-ilc --no-nonlinear
    python ekg_ilc_rt.py clean --mode buffered --no-override      # watch anomalies corrupt the template
    python ekg_ilc_rt.py gen-eeg --show                          # interactive window (zoom/pan)
    python ekg_ilc_rt.py passloss --duration 60 --out-prefix passloss_demo
    python ekg_ilc_rt.py passloss --duration 60 --band-lo 8 --band-hi 20 --show
"""
import argparse
import numpy as np
from scipy import signal
import matplotlib

if not hasattr(argparse, "BooleanOptionalAction"):
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            opts = []
            for opt in option_strings:
                opts.append(opt)
                if opt.startswith("--"):
                    opts.append("--no-" + opt[2:])
            kwargs.setdefault("nargs", 0)
            super().__init__(option_strings=opts, dest=dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

    argparse.BooleanOptionalAction = BooleanOptionalAction

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
def phase_estimator(
    r,
    fs,
    use_pll=True,
    use_override=True,
    rr0=0.85,
    debug=False,
    thr_hi=0.60,
    emph_delta=3.2e-5,
):
    N = len(r)
    sos = signal.butter(2, [8, 20], btype="band", fs=fs, output="sos")
    emph = np.abs(signal.sosfilt(sos, r))
    DEC = np.exp(-1.0 / (0.35 * fs)); AB = 1.0 / (1.5 * fs)
    THR_HI, THR_LO = float(thr_hi), 0.15; REFR = int(0.25 * fs); WARM = int(2.0 * fs)
    EMPH_DELTA = float(emph_delta)
    peak = 1e-6; base = 0.0; armed = False; refr = 0
    run_max = -1e9; rmi = 0; phi_pk = 0.0
    phi = 0.0; RR_hat = rr0; omega = 1.0 / (RR_hat * fs); last = None
    fids = []; phi_log = np.zeros(N); rr_log = np.zeros(N)
    if debug:
        emph_log = np.zeros(N)
        peak_log = np.zeros(N)
        base_log = np.zeros(N)
        gate_log = np.zeros(N)
        armed_log = np.zeros(N, dtype=int)
        thresh_met_log = np.zeros(N, dtype=int)
        detect_log = np.zeros(N, dtype=int)
        rr_meas_log = np.full(N, np.nan)
    wrap = lambda p: ((p + 0.5) % 1.0) - 0.5
    for n in range(N):
        e = emph[n]
        if debug:
            emph_log[n] = e
        peak = e if e > peak else peak * DEC
        base = base + AB * (e - base)
        gate = (e - base) / (peak - base + 1e-9)
        phi = (phi + omega) % 1.0
        phi_log[n] = phi; rr_log[n] = RR_hat
        if debug:
            peak_log[n] = peak
            base_log[n] = base
            gate_log[n] = gate
        if refr > 0: refr -= 1
        thresh_met = (gate > THR_HI) and (e > (base + EMPH_DELTA)) and (refr == 0) and (n > WARM)
        if debug:
            thresh_met_log[n] = int(thresh_met)
        if not armed and thresh_met:
            armed = True; run_max = -1e9
        if armed:
            if r[n] > run_max: run_max = r[n]; rmi = n; phi_pk = phi
            if gate < THR_LO:
                armed = False; refr = REFR; fid = rmi; rr_pre = RR_hat
                rr_meas = RR_hat if last is None else (fid - last) / fs
                if debug:
                    rr_meas_log[fid] = rr_meas
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
                if debug:
                    detect_log[n] = 1
        if debug:
            armed_log[n] = int(armed)

    out = {"fids": fids, "phi": phi_log, "rrhat": rr_log}
    if debug:
        out.update({
            "emph": emph_log,
            "peak": peak_log,
            "base": base_log,
            "gate": gate_log,
            "armed": armed_log,
            "thresh_met": thresh_met_log,
            "detect_events": detect_log,
            "rr_meas": rr_meas_log,
            "params": {
                "THR_HI": THR_HI,
                "THR_LO": THR_LO,
                "EMPH_DELTA": EMPH_DELTA,
                "AB": AB,
                "DEC": DEC,
                "REFR": REFR,
                "WARM": WARM,
            },
        })
    return out

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

def plot_ecg(t, r, beat_t, out=None, show=False):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, r, lw=0.6, color="0.3")
    if len(beat_t) > 0:
        bt = np.asarray(beat_t)
        bt = bt[(bt >= t[0]) & (bt <= t[-1])]
        ridx = np.searchsorted(t, bt)
        ridx = np.clip(ridx, 0, len(t) - 1)
        ax.scatter(t[ridx], r[ridx], s=18, c="C2", marker="o", alpha=0.85, label="R peaks")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, min(t[-1], 20)); ax.set_title("Synthetic ECG reference (green markers = R peaks)")
    ax.set_xlabel("time (s)"); ax.grid(alpha=0.3)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=110)
    if show:
        plt.show()
    plt.close()

def plot_eeg(t, y, s, cfa, out=None, show=False):
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t, cfa, lw=0.6, color="C3"); ax[0].set_title("Cardiac field artifact (nonlinear + volume conduction)")
    ax[1].plot(t, s, lw=0.6, color="C0"); ax[1].set_title("True neural signal (pink + alpha)")
    ax[2].plot(t, y, lw=0.6, color="0.3"); ax[2].set_title("Contaminated EEG = neural + CFA + noise")
    ax[2].set_xlabel("time (s)"); ax[2].set_xlim(0, min(t[-1], 20))
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=110)
    if show:
        plt.show()
    plt.close()

def plot_clean(t, y, s, cfa, clean_ilc, clean, template, fs, args, out=None, show=False):
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
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=110)
    if show:
        plt.show()
    plt.close()


def estimate_pass_loss_freq(y, r, fs, nperseg=2048):
    """Estimate EKG->EEG transfer and pass loss in frequency domain."""
    eps = 1e-12
    f, Sxx = signal.welch(r, fs=fs, nperseg=min(nperseg, len(r)))
    _, Syy = signal.welch(y, fs=fs, nperseg=min(nperseg, len(y)))
    _, Syx = signal.csd(y, r, fs=fs, nperseg=min(nperseg, len(r)))
    H = Syx / (Sxx + eps)
    coh = (np.abs(Syx) ** 2) / (Sxx * Syy + eps)
    pass_loss_db = -20.0 * np.log10(np.abs(H) + eps)
    return f, pass_loss_db, coh, np.abs(H)


def estimate_pass_loss_timevarying(y, r, fs, band_lo=8.0, band_hi=20.0, win_sec=8.0, hop_sec=2.0):
    """Sliding-window pass loss estimate over a selected frequency band."""
    eps = 1e-12
    win = max(64, int(round(win_sec * fs)))
    hop = max(1, int(round(hop_sec * fs)))
    if len(y) < win:
        return np.array([]), np.array([]), np.array([])

    tc, loss_db, coh_band = [], [], []
    for a in range(0, len(y) - win + 1, hop):
        b = a + win
        yy = y[a:b]
        rr = r[a:b]
        nps = min(1024, len(rr))
        f, Sxx = signal.welch(rr, fs=fs, nperseg=nps)
        _, Syy = signal.welch(yy, fs=fs, nperseg=nps)
        _, Syx = signal.csd(yy, rr, fs=fs, nperseg=nps)
        H = Syx / (Sxx + eps)
        coh = (np.abs(Syx) ** 2) / (Sxx * Syy + eps)

        m = (f >= band_lo) & (f <= band_hi)
        if not np.any(m):
            continue
        w = coh[m] + eps
        mag = np.sum(w * np.abs(H[m])) / np.sum(w)
        cb = float(np.mean(coh[m]))
        tc.append((a + b) / 2.0 / fs)
        loss_db.append(-20.0 * np.log10(mag + eps))
        coh_band.append(cb)

    return np.array(tc), np.array(loss_db), np.array(coh_band)


def estimate_pass_loss_beatsync(
    y,
    r,
    fs,
    est,
    band_lo=8.0,
    band_hi=20.0,
    pre_sec=0.30,
    post_sec=0.50,
    beats_per_est=12,
    step_beats=1,
):
    """Beat-synchronous pass loss estimate pooled over rolling beat batches."""
    eps = 1e-12
    pre = int(round(pre_sec * fs))
    post = int(round(post_sec * fs))
    L = pre + post
    if L < 32:
        return np.array([]), np.array([]), np.array([]), 0

    fids = est.get("fids", [])
    valid = [fid for fid, ty, lg in fids if lg]
    seg_y = []
    seg_r = []
    beat_t = []
    for fid in valid:
        a = fid - pre
        b = fid + post
        if a < 0 or b > len(y):
            continue
        seg_y.append(y[a:b])
        seg_r.append(r[a:b])
        beat_t.append(fid / fs)

    nb = len(seg_y)
    if nb < max(2, beats_per_est):
        return np.array([]), np.array([]), np.array([]), nb

    seg_y = np.asarray(seg_y)
    seg_r = np.asarray(seg_r)
    beat_t = np.asarray(beat_t)

    win = np.hanning(L)
    nfft = 1
    while nfft < L:
        nfft *= 2
    fr = np.fft.rfftfreq(nfft, d=1.0 / fs)
    m = (fr >= band_lo) & (fr <= band_hi)
    if not np.any(m):
        return np.array([]), np.array([]), np.array([]), nb

    tc = []
    loss_db = []
    coh_band = []
    K = int(max(2, beats_per_est))
    S = int(max(1, step_beats))
    for i0 in range(0, nb - K + 1, S):
        i1 = i0 + K
        Yb = seg_y[i0:i1]
        Rb = seg_r[i0:i1]
        Sxx = np.zeros_like(fr)
        Syy = np.zeros_like(fr)
        Syx = np.zeros_like(fr, dtype=np.complex128)
        for yy, rr in zip(Yb, Rb):
            X = np.fft.rfft(win * rr, n=nfft)
            Y = np.fft.rfft(win * yy, n=nfft)
            Sxx += np.abs(X) ** 2
            Syy += np.abs(Y) ** 2
            Syx += Y * np.conj(X)
        Sxx /= K
        Syy /= K
        Syx /= K
        H = Syx / (Sxx + eps)
        coh = (np.abs(Syx) ** 2) / (Sxx * Syy + eps)
        w = coh[m] + eps
        mag = np.sum(w * np.abs(H[m])) / np.sum(w)
        cb = float(np.mean(coh[m]))
        tc.append(float(np.mean(beat_t[i0:i1])))
        loss_db.append(-20.0 * np.log10(mag + eps))
        coh_band.append(cb)

    return np.array(tc), np.array(loss_db), np.array(coh_band), nb


def plot_pass_loss_frequency(f, pass_loss_db, coh, out=None, show=False, close=True):
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(f, pass_loss_db, color="C0", lw=1.5, label="pass loss (dB)")
    ax1.set_title("Estimated EKG->EEG pass loss (frequency domain)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Pass loss (dB)")
    ax1.set_xlim(0, min(60, f[-1]))
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(f, coh, color="C3", lw=1.0, alpha=0.85, label="coherence")
    ax2.set_ylabel("Coherence")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=120)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_pass_loss_timevarying(tc, loss_db, coh_band, band_lo, band_hi, out=None, show=False, close=True):
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(tc, loss_db, color="C0", lw=1.5, label="band pass loss (dB)")
    ax1.set_title(f"Time-varying EKG->EEG pass loss ({band_lo:.1f}-{band_hi:.1f} Hz)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pass loss (dB)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(tc, coh_band, color="C3", lw=1.0, alpha=0.85, label="band coherence")
    ax2.set_ylabel("Band coherence")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=120)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_pass_loss_beatsync(tc, loss_db, coh_band, band_lo, band_hi, out=None, show=False, close=True):
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(tc, loss_db, color="C0", lw=1.5, label="beat-synchronous pass loss (dB)")
    ax1.set_title(f"Beat-synchronous EKG->EEG pass loss ({band_lo:.1f}-{band_hi:.1f} Hz)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pass loss (dB)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(tc, coh_band, color="C3", lw=1.0, alpha=0.85, label="beat-band coherence")
    ax2.set_ylabel("Band coherence")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=120)
    if show:
        plt.show()
    if close:
        plt.close()

# ============================================================================
# 5. CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(prog="ekg_ilc_rt.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    def add_common(sp):
        sp.add_argument("--fs", type=float, default=250.0)
        sp.add_argument("--duration", type=float, default=60.0)
        sp.add_argument("--seed", type=int, default=11)
        sp.add_argument("--anomalies", action=argparse.BooleanOptionalAction, default=True)
        sp.add_argument("--out", default=None, help="output PNG path")
        sp.add_argument("--show", action=argparse.BooleanOptionalAction, default=False,
                        help="show interactive plot window (zoom/pan)")
    sp = sub.add_parser("gen-ecg", help="generate & plot synthetic ECG"); add_common(sp)
    sp = sub.add_parser("gen-eeg", help="generate & plot synthetic EEG + CFA"); add_common(sp)
    sp.add_argument("--drift", type=float, default=0.20); sp.add_argument("--artifact-gain", type=float, default=3.0)
    sp = sub.add_parser("passloss", help="estimate EKG->EEG pass loss (frequency + time-varying)"); add_common(sp)
    sp.add_argument("--drift", type=float, default=0.20)
    sp.add_argument("--artifact-gain", type=float, default=3.0)
    sp.add_argument("--band-lo", type=float, default=8.0, help="Lower frequency bound (Hz)")
    sp.add_argument("--band-hi", type=float, default=20.0, help="Upper frequency bound (Hz)")
    sp.add_argument("--win-sec", type=float, default=8.0, help="Sliding window length for time-varying estimate")
    sp.add_argument("--hop-sec", type=float, default=2.0, help="Sliding window hop for time-varying estimate")
    sp.add_argument("--beats-per-est", type=int, default=12, help="Beats pooled per beat-synchronous estimate")
    sp.add_argument("--beat-step", type=int, default=1, help="Beat hop for beat-synchronous estimate")
    sp.add_argument("--out-prefix", default=None, help="Output prefix for pass-loss figures")
    sp = sub.add_parser("clean", help="run the cleanup pipeline"); add_common(sp)
    sp.add_argument("--mode", choices=["noncausal", "buffered", "causal"], default="buffered")
    sp.add_argument("--phase-tracker", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--override", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--ilc", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--nonlinear", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--rls", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--drift", type=float, default=0.20); sp.add_argument("--artifact-gain", type=float, default=3.0)
    args = p.parse_args()

    global plt
    if args.show:
        import matplotlib.pyplot as plt
    else:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    if args.cmd == "gen-ecg":
        t, r, beat_t, btype = gen_ecg(args.fs, args.duration, args.seed, args.anomalies)
        print("ECG: %.0f s, %d beats (%d premature, %d after-drop)" %
              (args.duration, len(beat_t), btype.count("premature"), btype.count("missed")))
        out = args.out if args.out else (None if args.show else "ecg.png")
        plot_ecg(t, r, beat_t, out=out, show=args.show)
        if out:
            print("wrote", out)

    elif args.cmd == "gen-eeg":
        t, y, s, cfa, r, beat_t, btype = gen_eeg(args.fs, args.duration, args.seed, args.anomalies,
                                                 args.drift, args.artifact_gain)
        ss = t > (t[-1] * 0.5)
        print("EEG: input artifact SNR (s vs y) = %.2f dB over %d beats" %
              (snr_db(s, y, ss), len(beat_t)))
        out = args.out if args.out else (None if args.show else "eeg.png")
        plot_eeg(t, y, s, cfa, out=out, show=args.show)
        if out:
            print("wrote", out)

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
        out = args.out if args.out else (None if args.show else "clean.png")
        if out or args.show:
            plot_clean(t, y, s, cfa, clean_ilc, clean, template, args.fs, args, out=out, show=args.show)
        if out:
            print("wrote", out)

    elif args.cmd == "passloss":
        t, y, s, cfa, r, beat_t, btype = gen_eeg(args.fs, args.duration, args.seed, args.anomalies,
                                                 args.drift, args.artifact_gain)
        f, loss_db_f, coh_f, magH = estimate_pass_loss_freq(y, r, args.fs)
        tc, loss_db_t, coh_t = estimate_pass_loss_timevarying(
            y,
            r,
            args.fs,
            band_lo=args.band_lo,
            band_hi=args.band_hi,
            win_sec=args.win_sec,
            hop_sec=args.hop_sec,
        )
        est = phase_estimator(r, args.fs, use_pll=True, use_override=True)
        tc_b, loss_db_b, coh_b, nbeats_used = estimate_pass_loss_beatsync(
            y,
            r,
            args.fs,
            est,
            band_lo=args.band_lo,
            band_hi=args.band_hi,
            beats_per_est=args.beats_per_est,
            step_beats=args.beat_step,
        )

        m = (f >= args.band_lo) & (f <= args.band_hi)
        if np.any(m):
            print("Pass loss summary:")
            print("  band %.1f-%.1f Hz mean pass loss: %.2f dB" % (args.band_lo, args.band_hi, np.mean(loss_db_f[m])))
            print("  band %.1f-%.1f Hz mean coherence: %.3f" % (args.band_lo, args.band_hi, np.mean(coh_f[m])))
        if len(loss_db_b) > 0:
            print("  Figure 3 beat-synchronous mean pass loss: %.2f dB (nbeats=%d)" % (np.mean(loss_db_b), nbeats_used))
        else:
            print("  Figure 3 beat-synchronous estimate unavailable (not enough valid beats)")

        if args.out_prefix:
            out_f = f"{args.out_prefix}_freq.png"
            out_t = f"{args.out_prefix}_timevary.png"
            out_b = f"{args.out_prefix}_figure3_beatsync.png"
        else:
            out_f = None if args.show else "passloss_freq.png"
            out_t = None if args.show else "passloss_timevary.png"
            out_b = None if args.show else "passloss_figure3_beatsync.png"

        plot_pass_loss_frequency(f, loss_db_f, coh_f, out=out_f, show=False)
        plot_pass_loss_timevarying(tc, loss_db_t, coh_t, args.band_lo, args.band_hi, out=out_t, show=False)
        if len(tc_b) > 0:
            plot_pass_loss_beatsync(tc_b, loss_db_b, coh_b, args.band_lo, args.band_hi, out=out_b, show=False)

        if args.show:
            # Build both figures first, then show once so both windows appear together.
            plot_pass_loss_frequency(f, loss_db_f, coh_f, out=None, show=False, close=False)
            plot_pass_loss_timevarying(
                tc,
                loss_db_t,
                coh_t,
                args.band_lo,
                args.band_hi,
                out=None,
                show=False,
                close=False,
            )
            if len(tc_b) > 0:
                plot_pass_loss_beatsync(
                    tc_b,
                    loss_db_b,
                    coh_b,
                    args.band_lo,
                    args.band_hi,
                    out=None,
                    show=False,
                    close=False,
                )
            plt.show()
            plt.close("all")

        if out_f:
            print("wrote", out_f)
        if out_t:
            print("wrote", out_t)
        if out_b and len(tc_b) > 0:
            print("wrote", out_b)

if __name__ == "__main__":
    main()