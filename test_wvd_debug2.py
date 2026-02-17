#!/usr/bin/env python3
"""
Definitive WVD debug — isolate the frequency marginal issue.
Run with:  pytest test_wvd_debug2.py -v -s
"""
import numpy as np


def wvd_reference(x, n_fbins):
    """
    Reference WVD with signal zero-padded so every time index
    gets the full lag range.  n_fbins = N (original signal length).
    """
    x = np.asarray(x, dtype=complex).ravel()
    N = x.shape[0]

    # Zero-pad to length 3N: place signal at centre
    xp = np.zeros(3 * N, dtype=complex)
    xp[N: 2 * N] = x

    ts = np.arange(N)
    tfr = np.zeros((n_fbins, N), dtype=complex)

    for icol in range(N):
        t_pad = icol + N
        tau_max = n_fbins // 2 - 1
        tau = np.arange(-tau_max, tau_max + 1).astype(int)
        freq_idx = np.remainder(n_fbins + tau, n_fbins).astype(int)
        tfr[freq_idx, icol] = xp[t_pad + tau] * np.conj(xp[t_pad - tau])

    tfr = np.fft.fft(tfr, axis=0)
    tfr = np.real(tfr)
    return tfr


def test_freq_marginal_nbins_equals_N():
    """With n_fbins=N, freq bin k maps to normalised freq k/N."""
    N = 64
    n = np.arange(N)
    f0_bin = 10
    x = np.exp(1j * 2 * np.pi * f0_bin * n / N)

    wvd = wvd_reference(x, n_fbins=N)
    freq_marginal = wvd.sum(axis=1)

    fm_peak = np.argmax(np.abs(freq_marginal))
    print(f"\n  n_fbins=N={N}, freq marginal peak at bin {fm_peak}, "
          f"expected {2 * f0_bin % N}")
    # With the bilinear product, peak should be at 2*f0_bin mod N
    # because x[t+τ]*x*[t-τ] = e^{j2π(2f0)τ/N}
    # FFT bin = 2*f0_bin when n_fbins=N

    X = np.fft.fft(x)
    expected = np.abs(X)**2
    ex_peak = np.argmax(expected)
    print(f"  |FFT(x)|² peak at bin {ex_peak}")

    # The WVD frequency marginal with n_fbins=N should give |X(f)|²
    # The frequency mapping: WVD bin k corresponds to normalised freq k/N
    # But the bilinear product doubles the frequency → bin 2*f0
    # The DFT |X|² has its peak at bin f0
    # So: WVD freq marginal bin k corresponds to DFT bin k/2
    # This means we should compare WVD marginal against |X|² evaluated
    # at HALF the frequency resolution... OR we need to undo the doubling.

    # Actually: the standard WVD convention is
    # W(t,f) = Σ_τ x(t+τ)x*(t-τ) e^{-j4πfτ}
    # Note the 4π, not 2π! This is because the standard definition uses
    # continuous frequency f where the Nyquist is fs/2, and the bilinear
    # product spans [-fs/2, fs/2].
    #
    # In discrete form with N frequency bins covering [0, fs):
    # The e^{-j2πkτ/N} FFT kernel maps the bilinear product
    # x[t+τ]x*[t-τ] (which oscillates at 2*f0 cycles per N samples)
    # to bin 2*f0.
    #
    # To get the marginal to equal |X[k]|², we need the FFT of the
    # kernel evaluated at k, where the kernel frequency is 2*f0.
    # So bin k of the WVD = physical frequency k*fs/(2*N) when n_fbins=N.
    #
    # The solution: n_fbins should be 2*N, and then we only keep
    # the first N bins (or equivalently, the frequency grid is
    # [0, fs/2) with N points).

    # Let's verify: with n_fbins=2*N, does the peak land at f0_bin?
    wvd2 = wvd_reference(x, n_fbins=2*N)
    freq_marginal2 = wvd2.sum(axis=1)
    fm_peak2 = np.argmax(np.abs(freq_marginal2))
    print(f"\n  n_fbins=2N={2*N}, freq marginal peak at bin {fm_peak2}, "
          f"expected {2 * f0_bin}")
    # With n_fbins=2N: bilinear product at 2*f0 cycles/N = 2*f0 cycles/(N)
    # FFT with 2N bins: bin = 2*f0 * (2N)/N = 4*f0 ... no that's wrong
    # Actually: the lag τ runs over the signal, the FFT is along τ
    # kernel = e^{j2π(2f0)τ/N}, FFT bin = 2f0 * n_fbins/N
    # With n_fbins=N:  bin = 2*f0
    # With n_fbins=2N: bin = 4*f0 ... that's worse
    
    # CONCLUSION: The mapping is always bin = 2*f0 * n_fbins/N
    # To make bin = f0 (matching the DFT), we need n_fbins/N = 1/2
    # i.e., n_fbins = N/2 ... that loses resolution.
    #
    # OR: we fix the kernel to remove the factor-of-2.
    # The standard discrete WVD uses:
    # R[t,τ] = x[t + τ] * conj(x[t - τ])
    # and the FFT index k maps to normalised frequency k/(2*n_fbins)
    # (not k/n_fbins) because of the doubling.
    #
    # So the correct comparison is:
    # WVD freq marginal at bin k == |X(k * N / (2*n_fbins))|²
    # With n_fbins=N: WVD bin k == |X(k/2)|² → need to resample X

    print(f"\n  Testing with resampled comparison (n_fbins=N)...")
    freq_marg_N = wvd_reference(x, n_fbins=N).sum(axis=1)
    # WVD bin k corresponds to DFT bin k/2
    # So WVD bin 2*f0 should have the peak, and that maps to DFT bin f0
    # Resample: for each WVD bin k, the physical frequency is k*fs/(2*N)
    # For the DFT with N bins, bin j has frequency j*fs/N
    # Matching: k*fs/(2N) = j*fs/N → j = k/2
    
    # Create expected by evaluating |X|² at double frequency resolution
    X_2N = np.fft.fft(x, n=2*N)
    expected_2N = np.abs(X_2N)**2
    # Take first N bins (which cover [0, fs/2))
    expected_resampled = expected_2N[:N]
    
    fm_norm = freq_marg_N / np.max(np.abs(freq_marg_N))
    ex_norm = expected_resampled / expected_resampled.max()
    r = np.corrcoef(fm_norm, ex_norm)[0, 1]
    print(f"  WVD(n_fbins=N) vs |FFT(x, 2N)|²[:N]:  r = {r:.6f}")
    assert r > 0.95, f"r = {r}"