#!/usr/bin/env python3
"""
generate_figures.py -- regenerate Fig. 1 and Fig. 2 of the ICASSP paper as
vector PDFs, ready to \\includegraphics in LaTeX.

Requires cfa_tool.py (the reference implementation of the synthesis:
signal generation, phase estimator, ILC/repetitive canceller, NRLS residual)
to be importable from the same directory.

Usage:
    python generate_figures.py
Produces:
    fig1_convergence_spectrum.pdf
    fig2_timedomain.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
import cfa_tool as C

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

FS = 250.0
DUR = 90.0
SEED = 11


def run_pipeline():
    t, y, s, d, r, beat_t, beat_type = C.gen_eeg(FS, DUR, SEED, True, 0.20, 3.0)
    est = C.phase_estimator(r, FS, True, True)

    results = {}
    for mode in ["noncausal", "buffered", "causal"]:
        e_ilc, templ = C.ilc_clean(y, FS, est, mode, True)
        e_final = C.rls_clean(e_ilc, r, FS, mode, True, True)
        results[mode] = (e_ilc, e_final, templ)
    return t, y, s, d, r, est, results


def fig1_convergence_spectrum(t, y, s, results, fs):
    N = len(t)
    ss = t > t[-1] * 0.5

    def windowed_snr(est, win=int(8 * fs)):
        centers, vals = [], []
        for a in range(0, N - win, win // 2):
            b = a + win
            vals.append(10 * np.log10(np.sum(s[a:b] ** 2) / np.sum((s[a:b] - est[a:b]) ** 2)))
            centers.append((a + b) / 2 / fs)
        return np.array(centers), np.array(vals)

    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.9))

    tc, snr_in = windowed_snr(y)
    ax[0].plot(tc, snr_in, label="input", lw=1.1, color="0.5")
    for mode, color in zip(["causal", "buffered", "noncausal"], ["C3", "C0", "C2"]):
        _, e_final, _ = results[mode]
        _, snr_out = windowed_snr(e_final)
        ax[0].plot(tc, snr_out, label=mode, lw=1.3, color=color)
    ax[0].set_title("(a) Online SNR convergence")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel(r"artifact SNR (dB)")
    ax[0].legend(ncol=2)
    ax[0].grid(alpha=0.3)

    _, e_final_buf, _ = results["buffered"]
    f_ax, P_y = signal.welch(y[ss], fs, nperseg=1024)
    _, P_clean = signal.welch(e_final_buf[ss], fs, nperseg=1024)
    _, P_s = signal.welch(s[ss], fs, nperseg=1024)
    ax[1].semilogy(f_ax, P_y, lw=1.0, alpha=0.8, label="contaminated", color="0.5")
    ax[1].semilogy(f_ax, P_clean, lw=1.4, label="cleaned", color="C0")
    ax[1].semilogy(f_ax, P_s, ":", lw=1.4, label="true", color="C2")
    ax[1].axvspan(9, 11, color="green", alpha=0.12)
    ax[1].set_xlim(0, 40)
    ax[1].set_title("(b) Spectrum (buffered mode)")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("PSD")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("fig1_convergence_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_timedomain(t, y, s, d, r, est, results, fs):
    e_ilc_buf, e_final_buf, _ = results["buffered"]
    R = np.array([f[0] for f in est["fids"]])

    z0, z1 = int(40 * fs), int(43.2 * fs)
    tt = t[z0:z1]

    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.0), sharex=True)

    ax[0].plot(tt, r[z0:z1], lw=0.8, color="0.2")
    for Rk in R:
        if z0 <= Rk < z1:
            ax[0].axvline(Rk / fs, color="C2", lw=0.7, alpha=0.6)
    ax[0].set_title("(a) Synthetic ECG reference (PQRST)")
    ax[0].set_ylabel(r"$r(t)$")
    ax[0].grid(alpha=0.3)

    ax[1].plot(tt, y[z0:z1], lw=0.7, color="0.55", label="contaminated")
    ax[1].plot(tt, e_final_buf[z0:z1], lw=1.0, color="C0", label="cleaned")
    ax[1].plot(tt, s[z0:z1], lw=1.0, color="C2", ls=":", label="true neural")
    ax[1].set_title("(b) EEG: contaminated vs. cleaned vs. truth")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel(r"$y(t)$")
    ax[1].legend(loc="upper right")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig2_timedomain.pdf", bbox_inches="tight")
    plt.close(fig)


def print_summary_table(t, y, s, results, fs):
    ss = t > t[-1] * 0.5

    def snr(est):
        return 10 * np.log10(np.sum(s[ss] ** 2) / np.sum((s[ss] - est[ss]) ** 2))

    print(f"raw y            : {snr(y):5.2f} dB")
    for mode in ["noncausal", "buffered", "causal"]:
        e_ilc, e_final, _ = results[mode]
        print(f"{mode:9s} +ILC   : {snr(e_ilc):5.2f} dB")
        print(f"{mode:9s} +ILC+NRLS: {snr(e_final):5.2f} dB")


if __name__ == "__main__":
    t, y, s, d, r, est, results = run_pipeline()
    fig1_convergence_spectrum(t, y, s, results, FS)
    fig2_timedomain(t, y, s, d, r, est, results, FS)
    print_summary_table(t, y, s, results, FS)
    print("wrote fig1_convergence_spectrum.pdf and fig2_timedomain.pdf")
