#%%
#!/usr/bin/env python3
"""
============================================================================
Fractional Fourier Transform ↔ Wigner-Ville Distribution Equivalence
============================================================================

Interactive demonstration with a slider for α ∈ [0, 2].

Dependencies
------------
  pip install numpy scipy matplotlib

Author : jacobas-del
Date   : 2026-02-11
"""

import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import rotate as ndrotate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider


# =========================================================================
# 0.  WIGNER-VILLE DISTRIBUTION
# =========================================================================
def wigner_ville_raw(x, n_fbins):
    """
    Raw WVD computation with zero-padded signal.
    Returns the WVD with n_fbins frequency rows × N time columns.
    """
    x = np.asarray(x, dtype=complex).ravel()
    N = x.shape[0]

    # Zero-pad signal into 3N buffer for full lag access
    xp = np.zeros(3 * N, dtype=complex)
    xp[N: 2 * N] = x

    tfr = np.zeros((n_fbins, N), dtype=complex)

    for icol in range(N):
        t_pad = icol + N
        tau_max = min(N - 1, n_fbins // 2 - 1)
        tau = np.arange(-tau_max, tau_max + 1).astype(int)
        freq_idx = np.remainder(n_fbins + tau, n_fbins).astype(int)
        tfr[freq_idx, icol] = xp[t_pad + tau] * np.conj(xp[t_pad - tau])

    tfr = np.fft.fft(tfr, axis=0)
    tfr = np.real(tfr)
    return tfr


def wigner_ville(x, n_fbins=None):
    """
    Compute the Wigner-Ville Distribution on the N-point DFT frequency
    grid, suitable for comparison with the FrFT.

    Uses 2× upsampling of the signal to correctly align the WVD frequency
    grid with the N-point DFT grid. The upsampled signal has half the
    normalized frequency, which compensates for the frequency-doubling
    effect of the bilinear WVD kernel.

    Returns
    -------
    tfr   : 2-D real array (N, N)
    ts    : 1-D int array  (N,)
    freqs : 1-D float array (N,)
    """
    from scipy.signal import resample
    
    x = np.asarray(x, dtype=complex).ravel()
    N = x.shape[0]
    if n_fbins is None:
        n_fbins = N

    # Upsample signal by 2× using sinc interpolation
    N2 = 2 * N
    x_up = resample(x, N2)

    # Zero-pad upsampled signal for full lag access
    xp = np.zeros(3 * N2, dtype=complex)
    xp[N2: 2 * N2] = x_up

    # Compute WVD of upsampled signal
    tfr = np.zeros((n_fbins, N2), dtype=complex)
    for icol in range(N2):
        t_pad = icol + N2
        tau_max = min(N2 - 1, n_fbins // 2 - 1)
        tau = np.arange(-tau_max, tau_max + 1).astype(int)
        freq_idx = np.remainder(n_fbins + tau, n_fbins).astype(int)
        tfr[freq_idx, icol] = xp[t_pad + tau] * np.conj(xp[t_pad - tau])

    tfr = np.fft.fft(tfr, axis=0)
    tfr = np.real(tfr)

    # Downsample time axis by 2 to get back to N columns
    tfr = tfr[:, ::2]

    ts = np.arange(N)
    freqs = np.arange(n_fbins, dtype=float) / n_fbins

    return tfr, ts, freqs


# =========================================================================
# 1.  ZERO-PAD WVD INTO A CIRCULAR CANVAS
# =========================================================================
def pad_wvd_for_rotation(wvd_matrix):
    M, N = wvd_matrix.shape
    D = int(np.ceil(np.sqrt(M**2 + N**2)))
    if D % 2 == 0:
        D += 1

    padded = np.zeros((D, D), dtype=wvd_matrix.dtype)
    pad_r = (D - M) // 2
    pad_c = (D - N) // 2
    padded[pad_r:pad_r + M, pad_c:pad_c + N] = wvd_matrix

    return padded, pad_r, pad_c


# =========================================================================
# 2.  RADON PROJECTION
# =========================================================================
def radon_projection(padded_wvd, angle_deg, output_len):
    D = padded_wvd.shape[0]

    rotated = ndrotate(padded_wvd, angle_deg, reshape=False,
                       order=3, mode='constant', cval=0.0)

    proj_full = rotated.sum(axis=0)

    centre = D // 2
    half = output_len // 2
    start = centre - half
    proj = proj_full[start: start + output_len]

    if len(proj) != output_len:
        proj = np.interp(
            np.linspace(0, len(proj_full) - 1, output_len),
            np.arange(len(proj_full)),
            proj_full,
        )
    return proj


# =========================================================================
# 3.  DISCRETE FRACTIONAL FOURIER TRANSFORM
# =========================================================================
def _dft_eigenvectors(N):
    d = np.cos(2.0 * np.pi * np.arange(N) / N)
    S = np.diag(d) + np.diag(np.ones(N - 1) * 0.5, 1) \
                    + np.diag(np.ones(N - 1) * 0.5, -1)
    S[0, N - 1] = 0.5
    S[N - 1, 0] = 0.5

    _, V = np.linalg.eigh(S)

    n = np.arange(N)
    F = np.exp(-1j * 2 * np.pi * np.outer(n, n) / N) / np.sqrt(N)
    FV = F @ V
    lambdas = np.sum(V.conj() * FV, axis=0)
    ks = np.round(-np.angle(lambdas) * 2.0 / np.pi).astype(int) % 4

    return V, ks


def dfrft(x, alpha):
    alpha = float(alpha) % 4
    N = len(x)
    x = np.asarray(x, dtype=complex)

    V, ks = _dft_eigenvectors(N)
    frac_eigenvalues = np.exp(-1j * np.pi * alpha * ks / 2.0)
    Fa_x = V @ (frac_eigenvalues * (V.T @ x))
    return Fa_x


# =========================================================================
# 4.  TEST SIGNAL
# =========================================================================
def build_test_signal(N=513, fs=1024.0):
    t = np.linspace(0, 1, N, endpoint=False)

    sigma = 0.03
    env1 = np.exp(-0.5 * ((t - 0.30) / sigma)**2)
    env2 = np.exp(-0.5 * ((t - 0.70) / sigma)**2)

    x_real = env1 * np.cos(2 * np.pi * 150 * t) \
           + env2 * np.cos(2 * np.pi * 350 * t)

    x_analytic = hilbert(x_real).astype(np.complex128)
    return t, x_analytic, fs


# =========================================================================
# 5.  PRE-COMPUTE
# =========================================================================
def precompute(x, N, padded_wvd, n_angles=361):
    alphas = np.linspace(0.0, 2.0, n_angles)
    frft_densities  = np.zeros((n_angles, N))
    radon_densities = np.zeros((n_angles, N))
    correlations    = np.zeros(n_angles)

    for i, a in enumerate(alphas):
        X_a = dfrft(x, a)
        fd  = np.abs(X_a)**2
        fd /= fd.max() if fd.max() > 0 else 1.0
        frft_densities[i] = fd

        th = a * 90.0
        rd = radon_projection(padded_wvd, th, N)
        rd = np.abs(rd)
        rd /= rd.max() if rd.max() > 0 else 1.0
        radon_densities[i] = rd

        correlations[i] = np.corrcoef(fd, rd)[0, 1]

    return alphas, frft_densities, radon_densities, correlations


# =========================================================================
# 6.  ROTATED WVD IMAGE
# =========================================================================
def get_rotated_wvd_image(padded_wvd, angle_deg):
    return ndrotate(padded_wvd, angle_deg, reshape=False,
                    order=3, mode='constant', cval=0.0)


# =========================================================================
# 7.  MAIN
# =========================================================================
def main():
    N  = 513
    fs = 1024.0
    t, x, fs = build_test_signal(N=N, fs=fs)
    freq = np.linspace(-fs / 2, fs / 2, N, endpoint=False)

    # ── Folded N×N WVD ───────────────────────────────────────────────────
    wvd_matrix, _, _ = wigner_ville(x, n_fbins=N)

    # ── Sanity checks ───────────────────────────────────────────────────
    print("=" * 72)
    print("  SANITY CHECKS")
    print("=" * 72)

    # Time marginal
    time_marg = wvd_matrix.sum(axis=0)
    expected_t = np.abs(x)**2
    r_t = np.corrcoef(time_marg / time_marg.max(),
                       expected_t / expected_t.max())[0, 1]
    print(f"  WVD time marginal  vs |x(t)|²  :  r = {r_t:.6f}")

    # Freq marginal vs N-point DFT
    freq_marg = wvd_matrix.sum(axis=1)
    X_N = np.fft.fft(x)
    expected_f = np.abs(X_N)**2
    r_f = np.corrcoef(freq_marg / np.max(np.abs(freq_marg)),
                       expected_f / expected_f.max())[0, 1]
    print(f"  WVD freq marginal  vs |FFT_N|²  :  r = {r_f:.6f}")

    # Pad WVD for rotation (no fftshift needed - WVD and FrFT use same freq grid)
    padded_wvd, pad_r, pad_c = pad_wvd_for_rotation(wvd_matrix)

    # Radon 0°
    proj0 = radon_projection(padded_wvd, 0.0, N)
    proj0 = np.abs(proj0); proj0 /= proj0.max()
    expected_t_norm = expected_t / expected_t.max()
    r_r0 = np.corrcoef(proj0, expected_t_norm)[0, 1]
    print(f"  Radon θ=0°         vs |x(t)|²  :  r = {r_r0:.6f}")

    # FrFT α=0
    frft0 = np.abs(dfrft(x, 0.0))**2; frft0 /= frft0.max()
    r_f0 = np.corrcoef(frft0, expected_t_norm)[0, 1]
    print(f"  FrFT α=0           vs |x(t)|²  :  r = {r_f0:.6f}")

    # FrFT α=1 vs freq marginal
    frft1 = np.abs(dfrft(x, 1.0))**2; frft1 /= frft1.max()
    r_f1 = np.corrcoef(frft1, expected_f / expected_f.max())[0, 1]
    print(f"  FrFT α=1           vs |FFT_N|²  :  r = {r_f1:.6f}")

    # Radon 90° vs FrFT α=1
    proj90 = radon_projection(padded_wvd, 90.0, N)
    proj90 = np.abs(proj90); proj90 /= proj90.max()
    r_90 = np.corrcoef(proj90, frft1)[0, 1]
    print(f"  Radon θ=90°        vs FrFT α=1  :  r = {r_90:.6f}")

    print("=" * 72)
    all_r = [r_t, r_f, r_r0, r_f0, r_f1]
    if min(all_r) < 0.90:
        print("  ⚠  SOME CHECKS FAILED")
    else:
        print("  ✓  All basic checks passed (r > 0.90)")
    print("=" * 72)

    # ── Pre-compute ──────────────────────────────────────────────────────
    print("Pre-computing FrFT & Radon for 361 angles … ", end="", flush=True)
    n_angles = 361
    alphas, frft_all, radon_all, corr_all = precompute(
        x, N, padded_wvd, n_angles=n_angles)
    print("done.")

    print(f"\n[α = 0]  r = {corr_all[0]:.6f}")
    print(f"[α = 1]  r = {corr_all[n_angles // 2]:.6f}")
    print(f"[Global] Mean r = {corr_all.mean():.6f}   "
          f"Min r = {corr_all.min():.6f}   "
          f"Max r = {corr_all.max():.6f}")
    print("=" * 72)

    # ══════════════════════════════════════════════════════════════════════
    #  INTERACTIVE FIGURE
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        r"FrFT – WVD Radon Equivalence   ▸ drag the $\alpha$ slider",
        fontsize=15, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(
        3, 2, hspace=0.40, wspace=0.30,
        left=0.06, right=0.97, top=0.91, bottom=0.12)

    ax_wvd = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(np.abs(padded_wvd), 99)
    im_wvd = ax_wvd.imshow(
        get_rotated_wvd_image(padded_wvd, 0.0),
        aspect='auto', origin='lower', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax)
    title_wvd = ax_wvd.set_title(r"WVD rotated by $\theta = 0°$", fontsize=12)
    ax_wvd.set_xlabel("Column  (projection axis →)")
    ax_wvd.set_ylabel("Row  (integration axis ↑)")

    ax_cmp = fig.add_subplot(gs[0, 1])
    line_frft, = ax_cmp.plot(
        np.arange(N), frft_all[0], lw=2, color='#2166ac',
        label=r"$|\mathrm{FrFT}_\alpha|^2$")
    line_radon, = ax_cmp.plot(
        np.arange(N), radon_all[0], lw=2, ls='--', color='#d6604d',
        label="Radon proj.")
    ax_cmp.set_ylim(-0.05, 1.15)
    ax_cmp.legend(fontsize=10, loc='upper right')
    ax_cmp.grid(True, alpha=0.3)
    title_cmp = ax_cmp.set_title(
        r"$\alpha = 0.00$   $\theta = 0.0°$   r = {:.4f}".format(corr_all[0]),
        fontsize=12)
    ax_cmp.set_ylabel("Normalised density")
    ax_cmp.set_xlabel("Sample index")

    ax_corr = fig.add_subplot(gs[1, :])
    ax_corr.plot(alphas * 90, corr_all, lw=2, color='#1a9850')
    marker_corr, = ax_corr.plot(0, corr_all[0], 'o', ms=10,
                                 color='#e31a1c', zorder=5)
    ax_corr.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax_corr.set_xlim(0, 180)
    ax_corr.set_ylim(min(corr_all.min() - 0.05, 0.80), 1.05)
    ax_corr.set_xlabel(r"Projection angle  $\theta = \alpha \cdot 90°$",
                       fontsize=11)
    ax_corr.set_ylabel("Pearson  r", fontsize=11)
    ax_corr.set_title("Geometric Invariance — Correlation across all angles",
                      fontsize=12)
    ax_corr.grid(True, alpha=0.3)

    for sp_idx, (a_key, x_axis, xlabel, note) in enumerate([
        (0.0, t,    "Time [s]",      "peaks @ 0.3, 0.7 s"),
        (1.0, freq, "Frequency [Hz]", "peaks @ 150, 350 Hz"),
    ]):
        ax = fig.add_subplot(gs[2, sp_idx])
        ki = int(round(a_key / 2.0 * (n_angles - 1)))
        ax.plot(x_axis, frft_all[ki],  lw=2, color='#2166ac',
                label=r"$|\mathrm{FrFT}|^2$")
        ax.plot(x_axis, radon_all[ki], lw=2, ls='--', color='#d6604d',
                label="Radon proj.")
        ax.set_title(rf"$\alpha = {a_key:.0f}$  ({note})", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalised density")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.90, f"r = {corr_all[ki]:.4f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    ax_slider = fig.add_axes([0.15, 0.02, 0.70, 0.025])
    slider = Slider(
        ax_slider, r'$\alpha$', 0.0, 2.0,
        valinit=0.0, valstep=2.0 / (n_angles - 1),
        color='#4393c3')

    def update(val):
        a  = slider.val
        th = a * 90.0
        ki = int(round(a / 2.0 * (n_angles - 1)))

        im_wvd.set_data(get_rotated_wvd_image(padded_wvd, th))
        title_wvd.set_text(rf"WVD rotated by $\theta = {th:.1f}°$")

        line_frft.set_ydata(frft_all[ki])
        line_radon.set_ydata(radon_all[ki])
        title_cmp.set_text(
            rf"$\alpha = {a:.2f}$   $\theta = {th:.1f}°$"
            f"   r = {corr_all[ki]:.4f}")

        marker_corr.set_data([th], [corr_all[ki]])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()