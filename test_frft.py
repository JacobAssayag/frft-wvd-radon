#!/usr/bin/env python3
"""
Unit tests for FrFT ↔ WVD Radon equivalence.
Run with:  pytest test_frft.py -v -s
"""

import numpy as np
import pytest
from frft import (
    wigner_ville, pad_wvd_for_rotation, radon_projection,
    dfrft, build_test_signal, _dft_eigenvectors
)


# =====================================================================
# TEST 1:  dfrft sanity
# =====================================================================
class TestDFrFT:
    def test_alpha0_is_identity(self):
        np.random.seed(42)
        x = np.random.randn(64) + 1j * np.random.randn(64)
        y = dfrft(x, 0.0)
        np.testing.assert_allclose(np.abs(y)**2, np.abs(x)**2, atol=1e-10)

    def test_alpha1_is_dft(self):
        np.random.seed(42)
        x = np.random.randn(64) + 1j * np.random.randn(64)
        y = dfrft(x, 1.0)
        X = np.fft.fft(x) / np.sqrt(len(x))
        np.testing.assert_allclose(
            np.sort(np.abs(y)**2), np.sort(np.abs(X)**2), atol=1e-8)

    def test_alpha2_is_reversal(self):
        np.random.seed(42)
        x = np.random.randn(64) + 1j * np.random.randn(64)
        y = dfrft(x, 2.0)
        x_rev = np.abs(x[::-1])**2
        y_dens = np.abs(y)**2
        corr = np.max(np.correlate(y_dens / np.max(y_dens),
                                    x_rev / np.max(x_rev), mode='full'))
        assert corr > 0.95

    def test_alpha4_is_identity(self):
        np.random.seed(42)
        x = np.random.randn(32) + 1j * np.random.randn(32)
        y = dfrft(x, 4.0)
        np.testing.assert_allclose(np.abs(y)**2, np.abs(x)**2, atol=1e-10)

    def test_unitarity(self):
        np.random.seed(42)
        x = np.random.randn(64) + 1j * np.random.randn(64)
        for alpha in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]:
            y = dfrft(x, alpha)
            np.testing.assert_allclose(
                np.sum(np.abs(y)**2), np.sum(np.abs(x)**2), atol=1e-8)


# =====================================================================
# TEST 2:  Eigenvectors
# =====================================================================
class TestEigenvectors:
    def test_eigenvectors_diagonalise_dft(self):
        N = 64
        V, ks = _dft_eigenvectors(N)
        n = np.arange(N)
        F = np.exp(-1j * 2 * np.pi * np.outer(n, n) / N) / np.sqrt(N)
        D = V.T @ F @ V
        off_diag = D - np.diag(np.diag(D))
        assert np.max(np.abs(off_diag)) < 1e-8

    def test_eigenvalues_are_fourth_roots(self):
        N = 64
        V, ks = _dft_eigenvectors(N)
        n = np.arange(N)
        F = np.exp(-1j * 2 * np.pi * np.outer(n, n) / N) / np.sqrt(N)
        D = V.T @ F @ V
        for lam in np.diag(D):
            dists = [abs(lam - t) for t in [1, -1j, -1, 1j]]
            assert min(dists) < 1e-8


# =====================================================================
# TEST 3:  WVD marginals (folded N×N)
# =====================================================================
class TestWVD:
    def test_time_marginal(self):
        """Sum over frequency → |x(t)|²."""
        N = 127
        t, x, fs = build_test_signal(N=N, fs=512.0)
        wvd, _, _ = wigner_ville(x, n_fbins=N)
        time_marginal = wvd.sum(axis=0)
        expected = np.abs(x)**2
        time_marginal /= time_marginal.max()
        expected /= expected.max()
        r = np.corrcoef(time_marginal, expected)[0, 1]
        assert r > 0.99, f"Time marginal: r = {r}"

    def test_freq_marginal(self):
        """Sum over time → |X[k]|² (N-point DFT)."""
        N = 127
        t, x, fs = build_test_signal(N=N, fs=512.0)
        wvd, _, _ = wigner_ville(x, n_fbins=N)
        freq_marginal = wvd.sum(axis=1)

        X = np.fft.fft(x)
        expected = np.abs(X)**2

        freq_marginal /= np.max(np.abs(freq_marginal))
        expected /= expected.max()
        r = np.corrcoef(freq_marginal, expected)[0, 1]
        print(f"  Freq marginal: r = {r:.6f}")
        assert r > 0.99, f"Freq marginal: r = {r}"

    def test_freq_marginal_simple(self):
        """Single complex exponential: peak at f0 bin."""
        N = 64
        n = np.arange(N)
        f0_bin = 10
        x = np.exp(1j * 2 * np.pi * f0_bin * n / N)
        wvd, _, _ = wigner_ville(x, n_fbins=N)
        freq_marginal = wvd.sum(axis=1)

        actual_peak = np.argmax(np.abs(freq_marginal))
        print(f"  Peak at bin {actual_peak}, expected {f0_bin}")
        assert actual_peak == f0_bin, \
            f"Peak at {actual_peak}, expected {f0_bin}"

    def test_wvd_is_square(self):
        """The folded WVD must be N×N (square)."""
        N = 127
        t, x, fs = build_test_signal(N=N, fs=512.0)
        wvd, _, _ = wigner_ville(x, n_fbins=N)
        assert wvd.shape == (N, N), f"Shape is {wvd.shape}, expected ({N},{N})"


# =====================================================================
# TEST 4:  Radon
# =====================================================================
class TestRadon:
    def test_radon_0deg_matches_time_marginal(self):
        N = 127
        t, x, fs = build_test_signal(N=N, fs=512.0)
        wvd, _, _ = wigner_ville(x, n_fbins=N)
        # No fftshift - WVD and FrFT use same frequency grid
        padded, _, _ = pad_wvd_for_rotation(wvd)

        proj = radon_projection(padded, 0.0, N)
        proj = np.abs(proj); proj /= proj.max()

        expected = np.abs(x)**2; expected /= expected.max()
        r = np.corrcoef(proj, expected)[0, 1]
        assert r > 0.95, f"Radon 0°: r = {r}"

    def test_radon_90deg_matches_freq_marginal(self):
        """At 90°, projection should match |X[k]|²."""
        N = 127
        t, x, fs = build_test_signal(N=N, fs=512.0)
        wvd, _, _ = wigner_ville(x, n_fbins=N)
        # No fftshift - WVD and FrFT use same frequency grid
        padded, _, _ = pad_wvd_for_rotation(wvd)

        proj = radon_projection(padded, 90.0, N)
        proj = np.abs(proj); proj /= proj.max()

        # Freq marginal of the WVD (not shifted)
        freq_marg = np.abs(wvd.sum(axis=1))
        freq_marg /= freq_marg.max()

        r = np.corrcoef(proj, freq_marg)[0, 1]
        print(f"  Radon 90° vs freq marginal: r = {r:.4f}")
        assert r > 0.90, f"Radon 90°: r = {r}"


# =====================================================================
# TEST 5:  Equivalence
# =====================================================================
class TestEquivalence:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
    def test_frft_matches_radon(self, alpha):
        """Test FrFT-Radon equivalence at key angles (0°, 90°, 180°)."""
        N = 127
        t, x, fs = build_test_signal(N=N, fs=512.0)
        wvd, _, _ = wigner_ville(x, n_fbins=N)
        # No fftshift - WVD and FrFT use same frequency grid
        padded, _, _ = pad_wvd_for_rotation(wvd)

        X_a = dfrft(x, alpha)
        fd = np.abs(X_a)**2
        fd /= fd.max() if fd.max() > 0 else 1.0

        theta = alpha * 90.0
        rd = radon_projection(padded, theta, N)
        rd = np.abs(rd)
        rd /= rd.max() if rd.max() > 0 else 1.0

        r = np.corrcoef(fd, rd)[0, 1]
        print(f"  α={alpha}, θ={theta}°: r = {r:.4f}")
        assert r > 0.90, f"α={alpha}, θ={theta}°: r = {r}"