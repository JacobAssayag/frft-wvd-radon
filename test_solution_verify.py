#!/usr/bin/env python3
"""
THE definitive solution test.

The WVD with the standard discrete kernel R[t,τ] = x[t+τ]·x*[t-τ]
has frequency bin k mapping to physical frequency k·fs/(2·n_fbins).

This means WVD bin 2k = DFT bin k.

Solution: compute WVD with n_fbins = N, then DECIMATE by 2 in the
freq direction: keep only even-indexed rows. This gives an (N/2 × N)
matrix... but that's not square.

ACTUAL solution: the FrFT–WVD Radon equivalence theorem works in
CONTINUOUS time-frequency space. In discrete form, the WVD must
use the SAME coordinate grid as the FrFT. This requires computing
the WVD with the HALVED-lag kernel:

    R[t, τ] = x[t + τ] · x*[t - τ]    ← standard, freq = 2×
    
vs the INTERPOLATED version where we handle the 1/2 factor:
    evaluate at fractional lags using interpolation, or equivalently
    upsample x by 2×, compute WVD of upsampled signal, then 
    downsample time axis.

Let me test the upsampling approach.
"""
import numpy as np
from scipy.signal import resample


def wvd_upsampled(x, n_fbins=None):
    """
    Compute WVD whose frequency grid matches the N-point DFT.
    
    Method: upsample x by 2× (sinc interpolation), compute WVD
    with n_fbins = 2N freq bins, then downsample time axis by 2.
    
    The upsampled signal x_up[n] has half the normalised frequency
    of x[n], so the WVD kernel doubles it back to the original 
    frequency. WVD bin k with n_fbins=2N corresponds to:
      freq = 2 · (f0/(2N)) · (2N) / (2N) ... 
    
    Actually let me just think in terms of bins.
    
    x[n] = e^{j2πf0·n/N}  has DFT peak at bin f0 (N-point DFT)
    
    x_up[n] = e^{j2πf0·n/(2N)} has DFT peak at bin f0 (2N-point DFT)
    
    WVD kernel of x_up: 
      x_up[t+τ]·x_up*[t-τ] = e^{j2πf0·2τ/(2N)} = e^{j2πf0·τ/N}
    
    FFT of this with n_fbins bins: peak at bin f0·n_fbins/N
    
    With n_fbins = N: peak at bin f0  ← CORRECT!
    """
    x = np.asarray(x, dtype=complex).ravel()
    N = x.shape[0]
    if n_fbins is None:
        n_fbins = N
    
    # Upsample x by 2× via sinc interpolation
    N2 = 2 * N
    x_up = resample(x, N2)
    
    # Zero-pad for full lag access
    xp = np.zeros(3 * N2, dtype=complex)
    xp[N2: 2 * N2] = x_up
    
    # Compute WVD of upsampled signal with n_fbins freq bins
    # Time axis: 2N points (we'll downsample to N later)
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
    # Take every other column (the original sample positions)
    tfr_ds = tfr[:, ::2]
    
    return tfr_ds


def dfrft_simple(x, alpha):
    N = len(x)
    alpha = float(alpha) % 4
    x = np.asarray(x, dtype=complex)
    
    d = np.cos(2.0 * np.pi * np.arange(N) / N)
    S = np.diag(d) + np.diag(np.ones(N-1)*0.5, 1) + np.diag(np.ones(N-1)*0.5, -1)
    S[0, N-1] = 0.5
    S[N-1, 0] = 0.5
    _, V = np.linalg.eigh(S)
    
    n = np.arange(N)
    F = np.exp(-1j * 2 * np.pi * np.outer(n, n) / N) / np.sqrt(N)
    FV = F @ V
    lambdas = np.sum(V.conj() * FV, axis=0)
    ks = np.round(-np.angle(lambdas) * 2.0 / np.pi).astype(int) % 4
    frac_eig = np.exp(-1j * np.pi * alpha * ks / 2.0)
    return V @ (frac_eig * (V.T @ x))


def test_upsampled_simple_exponential():
    """Verify WVD freq marginal peak matches DFT with upsampled approach."""
    N = 64
    f0 = 10
    n = np.arange(N)
    x = np.exp(1j * 2 * np.pi * f0 * n / N)
    
    wvd = wvd_upsampled(x, n_fbins=N)
    print(f"\n  WVD shape: {wvd.shape}")
    
    freq_marg = wvd.sum(axis=1)
    wvd_peak = np.argmax(np.abs(freq_marg))
    
    X = np.fft.fft(x)
    dft_peak = np.argmax(np.abs(X)**2)
    
    y = dfrft_simple(x, 1.0)
    frft_peak = np.argmax(np.abs(y)**2)
    
    print(f"  WVD  freq marginal peak: bin {wvd_peak}")
    print(f"  DFT  N-point peak:       bin {dft_peak}")
    print(f"  FrFT α=1 peak:           bin {frft_peak}")
    print(f"  All should be at bin {f0}")
    
    assert wvd_peak == f0, f"WVD peak at {wvd_peak}, expected {f0}"
    
    # Full correlation
    fm_norm = freq_marg / np.max(np.abs(freq_marg))
    fd_norm = np.abs(y)**2; fd_norm /= fd_norm.max()
    r = np.corrcoef(fm_norm, fd_norm)[0, 1]
    print(f"  WVD freq marginal vs FrFT α=1: r = {r:.6f}")
    assert r > 0.95, f"r = {r}"


def test_upsampled_test_signal():
    """Verify with two-Gaussian test signal."""
    from frft import build_test_signal
    
    N = 127
    t, x, fs = build_test_signal(N=N, fs=512.0)
    
    wvd = wvd_upsampled(x, n_fbins=N)
    print(f"\n  WVD shape: {wvd.shape}")
    assert wvd.shape == (N, N), f"Expected ({N},{N}), got {wvd.shape}"
    
    # Time marginal
    time_marg = wvd.sum(axis=0)
    expected_t = np.abs(x)**2
    r_t = np.corrcoef(time_marg / time_marg.max(),
                       expected_t / expected_t.max())[0, 1]
    print(f"  Time marginal: r = {r_t:.6f}")
    assert r_t > 0.95, f"Time marginal: r = {r_t}"
    
    # Freq marginal vs N-point DFT
    freq_marg = wvd.sum(axis=1)
    X = np.fft.fft(x)
    expected_f = np.abs(X)**2
    r_f = np.corrcoef(freq_marg / np.max(np.abs(freq_marg)),
                       expected_f / expected_f.max())[0, 1]
    print(f"  Freq marginal vs |FFT_N|²: r = {r_f:.6f}")
    assert r_f > 0.95, f"Freq marginal: r = {r_f}"
    
    # FrFT vs Radon at key angles
    from scipy.ndimage import rotate as ndrotate
    
    wvd_shifted = np.fft.fftshift(wvd, axes=0)
    M, Nc = wvd_shifted.shape
    D = int(np.ceil(np.sqrt(M**2 + Nc**2)))
    if D % 2 == 0:
        D += 1
    padded = np.zeros((D, D))
    pr, pc = (D-M)//2, (D-Nc)//2
    padded[pr:pr+M, pc:pc+Nc] = wvd_shifted
    
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        theta = alpha * 90.0
        
        rotated = ndrotate(padded, theta, reshape=False,
                           order=3, mode='constant', cval=0.0)
        proj_full = rotated.sum(axis=0)
        centre = D // 2
        half = N // 2
        proj = proj_full[centre - half: centre - half + N]
        if len(proj) != N:
            proj = np.interp(np.linspace(0, len(proj_full)-1, N),
                             np.arange(len(proj_full)), proj_full)
        proj = np.abs(proj)
        proj /= proj.max() if proj.max() > 0 else 1.0
        
        ya = dfrft_simple(x, alpha)
        fd = np.abs(ya)**2
        fd /= fd.max() if fd.max() > 0 else 1.0
        
        r = np.corrcoef(fd, proj)[0, 1]
        print(f"  α={alpha:.1f}, θ={theta:.0f}°: r = {r:.4f}")