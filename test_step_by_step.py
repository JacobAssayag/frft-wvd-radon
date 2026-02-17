#!/usr/bin/env python3
"""
Step-by-step debug of WVD frequency mapping.
Run with:  pytest test_step_by_step.py -v -s
"""
import numpy as np


def wvd_raw_standalone(x, n_fbins):
    """
    Standalone raw WVD — no internal padding, no folding.
    Just the basic WVD kernel with the signal as given.
    """
    x = np.asarray(x, dtype=complex).ravel()
    N = x.shape[0]

    tfr = np.zeros((n_fbins, N), dtype=complex)

    for icol in range(N):
        tau_max = min(icol, N - icol - 1, n_fbins // 2 - 1)
        tau = np.arange(-tau_max, tau_max + 1).astype(int)
        freq_idx = np.remainder(n_fbins + tau, n_fbins).astype(int)
        tfr[freq_idx, icol] = x[icol + tau] * np.conj(x[icol - tau])

    tfr = np.fft.fft(tfr, axis=0)
    tfr = np.real(tfr)
    return tfr


def wvd_raw_padded(x, n_fbins):
    """
    Raw WVD with the signal zero-padded to 3N for full lag access.
    Output is (n_fbins, N) — only original time indices.
    """
    x = np.asarray(x, dtype=complex).ravel()
    N = x.shape[0]

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


def test_step1_raw_no_padding():
    """
    Step 1: Raw WVD (no padding) of e^{j2πf0n/N} with various n_fbins.
    
    The kernel is:
        x[t+τ] · x*[t-τ] = e^{j2πf0(t+τ)/N} · e^{-j2πf0(t-τ)/N}
                           = e^{j2πf0·2τ/N}
    
    This is a complex exponential in τ with frequency 2f0/N cycles per sample.
    
    FFT with n_fbins points maps this to bin:
        k = 2·f0 · n_fbins / N
    """
    N = 64
    f0 = 10
    n = np.arange(N)
    x = np.exp(1j * 2 * np.pi * f0 * n / N)

    print("\n  === Step 1: Raw WVD (no padding) ===")
    for nf in [N, 2*N]:
        wvd = wvd_raw_standalone(x, n_fbins=nf)
        freq_marg = wvd.sum(axis=1)
        peak = np.argmax(np.abs(freq_marg))
        expected_bin = (2 * f0 * nf // N) % nf
        print(f"  n_fbins={nf:4d}: peak at bin {peak:3d}, "
              f"theory says bin {expected_bin:3d}, "
              f"match={peak == expected_bin}")


def test_step2_raw_with_padding():
    """
    Step 2: Same but with zero-padded signal.
    Padding shouldn't change the frequency, only fix edge truncation.
    """
    N = 64
    f0 = 10
    n = np.arange(N)
    x = np.exp(1j * 2 * np.pi * f0 * n / N)

    print("\n  === Step 2: Raw WVD (with 3N padding) ===")
    for nf in [N, 2*N]:
        wvd = wvd_raw_padded(x, n_fbins=nf)
        freq_marg = wvd.sum(axis=1)
        peak = np.argmax(np.abs(freq_marg))
        expected_bin = (2 * f0 * nf // N) % nf
        print(f"  n_fbins={nf:4d}: peak at bin {peak:3d}, "
              f"theory says bin {expected_bin:3d}, "
              f"match={peak == expected_bin}")


def test_step3_what_folding_actually_does():
    """
    Step 3: See what happens when we fold 2N → N.
    
    With n_fbins=2N: peak is at bin 2·f0·2N/N = 4·f0 = 40
    Folding: bin 40 mod 64 = 40 (since 40 < 64)
    
    But we WANT the peak at bin f0 = 10!
    
    The issue: folding k → k mod N only works if the peak is at 
    bin 2·f0 (with n_fbins=N), not 4·f0 (with n_fbins=2N).
    """
    N = 64
    f0 = 10
    n = np.arange(N)
    x = np.exp(1j * 2 * np.pi * f0 * n / N)

    print("\n  === Step 3: Folding analysis ===")
    
    # With n_fbins=N (no padding): peak at 2*f0 = 20
    wvd_N = wvd_raw_standalone(x, n_fbins=N)
    fm_N = wvd_N.sum(axis=1)
    print(f"  Raw n_fbins=N:   peak at bin {np.argmax(np.abs(fm_N))}")
    
    # With n_fbins=2N (no padding): peak at 4*f0 = 40
    wvd_2N = wvd_raw_standalone(x, n_fbins=2*N)
    fm_2N = wvd_2N.sum(axis=1)
    print(f"  Raw n_fbins=2N:  peak at bin {np.argmax(np.abs(fm_2N))}")
    
    # Folding 2N → N: bin k' = bin k + bin k+N
    folded = wvd_2N[:N, :] + wvd_2N[N:, :]
    fm_folded = folded.sum(axis=1)
    print(f"  Folded 2N→N:     peak at bin {np.argmax(np.abs(fm_folded))}")
    
    # The peak at bin 40 in the 2N array → remains at bin 40 in folded
    # because 40 < N=64, so it's not folded at all!
    
    # CORRECT APPROACH: use n_fbins=N (which gives peak at 2*f0=20)
    # and then DOWNSAMPLE by 2: take every other bin
    # Or equivalently: the WVD with n_fbins=N already has the right
    # frequency mapping, we just need to understand it:
    #   WVD bin k (n_fbins=N) → physical freq = k·fs/(2·N)
    #   DFT bin j (N-point)   → physical freq = j·fs/N
    #   So WVD bin k = DFT bin k/2
    #   Meaning: WVD frequency grid has HALF the spacing of the DFT grid
    
    print("\n  === The real mapping with n_fbins=N ===")
    print(f"  WVD bin 20 → phys freq = 20/(2·64) = {20/(2*64):.4f}")
    print(f"  DFT bin 10 → phys freq = 10/64     = {10/64:.4f}")
    print(f"  They're the same! WVD bin 2k = DFT bin k")
    
    # So the correct approach: resample WVD by taking bins 0, 2, 4, ..., 2(N-1)
    # But that only works if n_fbins >= 2*N to have enough resolution
    # With n_fbins = N, bin 2k wraps around...
    
    # Actually the CORRECT approach for the Radon equivalence is:
    # DON'T try to make the WVD match the DFT grid.
    # Instead, keep the WVD on its natural grid (n_fbins=N, freq=k/(2N))
    # and adjust the FrFT comparison to use the same grid.
    # OR: compute the WVD with n_fbins=2N and DON'T fold.
    # The WVD is then (2N × N) — rectangular.
    # But Radon needs a SQUARE matrix for proper rotation.
    
    # THE ACTUAL SOLUTION: make the WVD N×N by using n_fbins=N
    # and accept that both axes use the SAME scale.
    # Time axis: N samples, spacing Δt = T/N
    # Freq axis: N samples, spacing Δf = fs/(2N)  [because of doubling]
    # For the rotation to be a proper Radon transform,
    # we need Δt = Δf (isotropic pixels).
    # Currently Δt = 1/N (sample spacing) and Δf = 1/(2N) → not equal!
    # This means the WVD pixels are NOT square → rotation ≠ Radon.
    
    print("\n  === Pixel aspect ratio ===")
    print(f"  Δt = 1 sample")
    print(f"  Δf = 1/(2) = 0.5 in normalised units")
    print(f"  Aspect ratio Δt/Δf = 2.0 → pixels are NOT square!")
    print(f"  Rotation mixes time and freq with wrong scaling!")


def test_step4_verify_n_fbins_2N_square():
    """
    Step 4: With n_fbins=2N (and padding), the WVD is (2N × N).
    Both axes have the same scale if we interpret:
      Time:  N points at spacing Δt
      Freq: 2N points at spacing Δf = 1/(2N) → total range = 1
      But time range = N·Δt = N
    
    For isotropy we need: Δt = Δf in the same units.
    Normalised: Δt = 1 (sample), Δf = 1/(2N)
    These are NOT equal.
    
    The solution: pad TIME to 2N samples as well, making a 2N×2N square.
    """
    N = 64
    f0 = 10
    n = np.arange(N)
    x = np.exp(1j * 2 * np.pi * f0 * n / N)

    # Zero-pad signal to 2N (pad with zeros in time)
    x2 = np.zeros(2 * N, dtype=complex)
    x2[:N] = x

    # WVD of the 2N-length signal with n_fbins=2N → (2N × 2N) square
    wvd = wvd_raw_padded(x2, n_fbins=2*N)
    print(f"\n  WVD shape: {wvd.shape}")
    
    # Freq marginal (sum over time columns)
    freq_marg = wvd.sum(axis=1)
    peak = np.argmax(np.abs(freq_marg))
    print(f"  Freq marginal peak at bin {peak}")
    # Theory: kernel freq = 2·f0, n_fbins=2N, so bin = 2·f0·(2N)/(2N) = 2·f0 = 20
    print(f"  Expected: 2·f0 = {2*f0}")
    
    # Compare with 2N-point DFT of x2
    X2N = np.fft.fft(x2)
    expected = np.abs(X2N)**2
    ex_peak = np.argmax(expected)
    print(f"  |FFT_{2*N}(x_padded)|² peak at bin {ex_peak}")
    # DFT peak: x2 has frequency f0 over first N samples
    # fft(x2, 2N) peak at bin f0·(2N)/N = 2·f0 = 20 ... wait
    # Actually x2 has period N, so fft bin = f0·(2N)/N = 2·f0
    # Hmm no. x = e^{j2πf0n/N} for n=0..N-1, zero-padded to 2N
    # fft(x2)[k] = Σ_{n=0}^{N-1} e^{j2π(f0/N - k/(2N))n}
    # Peak at k where f0/N = k/(2N) → k = 2·f0 = 20
    
    # So WVD peak = 20, DFT peak = 20 → THEY MATCH with 2N×2N!
    
    freq_marg_norm = freq_marg / np.max(np.abs(freq_marg))
    expected_norm = expected / expected.max()
    r = np.corrcoef(freq_marg_norm, expected_norm)[0, 1]
    print(f"  Correlation: r = {r:.6f}")
    
    # Now check time marginal too
    time_marg = wvd.sum(axis=0)
    expected_t = np.abs(x2)**2
    time_marg_norm = time_marg / time_marg.max()
    expected_t_norm = expected_t / expected_t.max()
    r_t = np.corrcoef(time_marg_norm, expected_t_norm)[0, 1]
    print(f"  Time marginal correlation: r_t = {r_t:.6f}")