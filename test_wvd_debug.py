#!/usr/bin/env python3
"""
Diagnostic test to understand the WVD frequency marginal failure.
Run with:  pytest /Users/jacobassayag/Projects/inr_project/FrFTvsRadTEST/test_wvd_debug.py -v -s
"""

import numpy as np
from frft import wigner_ville, build_test_signal


def test_wvd_freq_marginal_simple_signal():
    """
    Use a simple signal where we know the exact spectrum:
    a single complex exponential x[n] = exp(j·2π·f0·n/N).
    
    |X(f)|² should be a single peak at f0.
    The WVD freq marginal should match.
    """
    N = 64
    n = np.arange(N)
    f0_bin = 10  # frequency bin
    x = np.exp(1j * 2 * np.pi * f0_bin * n / N)

    n_fbins = 2 * N
    wvd, _, _ = wigner_ville(x, n_fbins=n_fbins)

    freq_marginal = wvd.sum(axis=1)   # sum over time

    X = np.fft.fft(x, n=n_fbins)
    expected = np.abs(X)**2

    # Print peak locations
    fm_peak = np.argmax(np.abs(freq_marginal))
    ex_peak = np.argmax(expected)
    print(f"\n  Freq marginal peak at bin {fm_peak}, expected at bin {ex_peak}")
    print(f"  Freq marginal[0:10] = {freq_marginal[:10]}")
    print(f"  Expected[0:10]      = {expected[:10]}")

    # Check: are they even the same shape?
    print(f"  Freq marginal shape: {freq_marginal.shape}")
    print(f"  Expected shape:      {expected.shape}")

    # Normalise and correlate
    fm_norm = freq_marginal / np.max(np.abs(freq_marginal))
    ex_norm = expected / expected.max()
    r = np.corrcoef(fm_norm, ex_norm)[0, 1]
    print(f"  Pearson r = {r:.6f}")


def test_wvd_tau_coverage():
    """
    Check how many lag samples each time index actually uses.
    If edge samples use very few lags, the WVD is truncated.
    """
    N = 64
    n_fbins = 2 * N
    tau_counts = []
    for icol in range(N):
        tau_max = min(icol, N - icol - 1, n_fbins // 2 - 1)
        n_taus = 2 * tau_max + 1
        tau_counts.append(n_taus)

    tau_counts = np.array(tau_counts)
    print(f"\n  Min tau count: {tau_counts.min()} at edges")
    print(f"  Max tau count: {tau_counts.max()} at centre")
    print(f"  First 5: {tau_counts[:5]}")
    print(f"  Last 5:  {tau_counts[-5:]}")
    # Edge samples use only 1-9 lags vs 63 at centre → massive truncation


def test_wvd_with_zero_padded_signal():
    """
    Zero-pad the signal to 2N before computing WVD.
    This gives every time sample access to the full lag range.
    """
    N = 64
    n = np.arange(N)
    f0_bin = 10
    x = np.exp(1j * 2 * np.pi * f0_bin * n / N)

    # Zero-pad signal
    x_padded = np.zeros(2 * N, dtype=complex)
    x_padded[:N] = x

    n_fbins = 2 * N  # match padded length
    wvd, _, _ = wigner_ville(x_padded, n_fbins=n_fbins)

    # Only look at the time range where the signal lives
    freq_marginal = wvd[:, :N].sum(axis=1)

    X = np.fft.fft(x, n=n_fbins)
    expected = np.abs(X)**2

    fm_norm = freq_marginal / np.max(np.abs(freq_marginal))
    ex_norm = expected / expected.max()
    r = np.corrcoef(fm_norm, ex_norm)[0, 1]
    print(f"\n  With zero-padded signal: Pearson r = {r:.6f}")
    assert r > 0.95, f"Still failing with padded signal: r = {r}"