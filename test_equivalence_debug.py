#!/usr/bin/env python3
"""
Debug the FrFT ↔ Radon mismatch.
Run with:  pytest test_equivalence_debug.py -v -s
"""
import numpy as np
from frft import (wigner_ville, pad_wvd_for_rotation, radon_projection,
                  dfrft, build_test_signal)


def test_dimension_mismatch_diagnosis():
    """
    The WVD is N×N but freq axis covers [0, fs/2) due to doubling.
    The FrFT output is N samples covering [0, fs).
    At α=1, FrFT gives the N-point DFT, but the WVD freq marginal
    gives the 2N-point DFT restricted to first N bins.
    These live on DIFFERENT grids → projection can't match FrFT.
    """
    N = 64
    n = np.arange(N)
    f0_bin = 10
    x = np.exp(1j * 2 * np.pi * f0_bin * n / N)

    # FrFT at α=1 → DFT
    y = dfrft(x, 1.0)
    frft_density = np.abs(y)**2
    frft_peak = np.argmax(frft_density)
    print(f"\n  FrFT α=1 peak at bin {frft_peak}")

    # WVD freq marginal with n_fbins=N
    wvd, _, _ = wigner_ville(x, n_fbins=N)
    freq_marg = wvd.sum(axis=1)
    wvd_peak = np.argmax(np.abs(freq_marg))
    print(f"  WVD freq marginal peak at bin {wvd_peak}")

    # The FrFT peak is at bin 10 (= f0_bin)
    # The WVD peak is at bin 20 (= 2*f0_bin)
    # → They're on different frequency grids!
    print(f"\n  FrFT bin {frft_peak} maps to freq {frft_peak}/N = {frft_peak/N:.4f}")
    print(f"  WVD  bin {wvd_peak} maps to freq {wvd_peak}/(2N) = {wvd_peak/(2*N):.4f}")
    print(f"  Both = normalised freq {f0_bin/N:.4f} ✓ — but different bin indices!")


def test_square_wvd_with_folding():
    """
    Solution: compute WVD with n_fbins=2N, then fold (alias)
    back to N bins by summing bins k and k+N. This maps the
    WVD onto the same N-point frequency grid as the DFT/FrFT.
    """
    N = 64
    n = np.arange(N)
    f0_bin = 10
    x = np.exp(1j * 2 * np.pi * f0_bin * n / N)

    # Compute WVD with 2N freq bins
    wvd_2N, _, _ = wigner_ville(x, n_fbins=2*N)

    # Fold: W_folded[k, t] = W[k, t] + W[k+N, t]  for k = 0..N-1
    wvd_folded = wvd_2N[:N, :] + wvd_2N[N:, :]

    freq_marg = wvd_folded.sum(axis=1)
    fm_peak = np.argmax(np.abs(freq_marg))
    print(f"\n  Folded WVD freq marginal peak at bin {fm_peak}")

    # Compare against N-point DFT (= FrFT at α=1)
    X = np.fft.fft(x)
    expected = np.abs(X)**2
    ex_peak = np.argmax(expected)
    print(f"  |FFT_N|² peak at bin {ex_peak}")

    freq_marg_norm = freq_marg / np.max(np.abs(freq_marg))
    expected_norm = expected / expected.max()
    r = np.corrcoef(freq_marg_norm, expected_norm)[0, 1]
    print(f"  Folded WVD freq marginal vs |FFT_N|²: r = {r:.6f}")
    assert r > 0.99, f"r = {r}"

    # Time marginal should still work
    time_marg = wvd_folded.sum(axis=0)
    expected_t = np.abs(x)**2
    time_marg_norm = time_marg / time_marg.max()
    expected_t_norm = expected_t / expected_t.max()
    r_t = np.corrcoef(time_marg_norm, expected_t_norm)[0, 1]
    print(f"  Folded WVD time marginal vs |x|²:     r = {r_t:.6f}")
    assert r_t > 0.99, f"r_t = {r_t}"


def test_folded_wvd_radon_equivalence():
    """
    With the folded N×N WVD, test Radon projection vs FrFT.
    """
    N = 127
    t, x, fs = build_test_signal(N=N, fs=512.0)

    # Compute folded WVD
    wvd_2N, _, _ = wigner_ville(x, n_fbins=2*N)
    wvd_folded = wvd_2N[:N, :] + wvd_2N[N:, :]

    wvd_shifted = np.fft.fftshift(wvd_folded, axes=0)
    padded, _, _ = pad_wvd_for_rotation(wvd_shifted)

    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        X_a = dfrft(x, alpha)
        fd = np.abs(X_a)**2
        fd /= fd.max() if fd.max() > 0 else 1.0

        theta = alpha * 90.0
        rd = radon_projection(padded, theta, N)
        rd = np.abs(rd)
        rd /= rd.max() if rd.max() > 0 else 1.0

        r = np.corrcoef(fd, rd)[0, 1]
        print(f"  α={alpha:.1f}, θ={theta:.0f}°: r = {r:.4f}")