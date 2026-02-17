#!/usr/bin/env python3
"""
Debug the Radon projection step by step.
Run with:  pytest test_radon_debug.py -v -s
"""
import numpy as np
from scipy.signal import resample
from scipy.ndimage import rotate as ndrotate


def wvd_upsampled(x, n_fbins=None):
    """WVD with 2× upsampling to fix frequency grid."""
    x = np.asarray(x, dtype=complex).ravel()
    N = x.shape[0]
    if n_fbins is None:
        n_fbins = N
    N2 = 2 * N
    x_up = resample(x, N2)
    xp = np.zeros(3 * N2, dtype=complex)
    xp[N2: 2 * N2] = x_up
    tfr = np.zeros((n_fbins, N2), dtype=complex)
    for icol in range(N2):
        t_pad = icol + N2
        tau_max = min(N2 - 1, n_fbins // 2 - 1)
        tau = np.arange(-tau_max, tau_max + 1).astype(int)
        freq_idx = np.remainder(n_fbins + tau, n_fbins).astype(int)
        tfr[freq_idx, icol] = xp[t_pad + tau] * np.conj(xp[t_pad - tau])
    tfr = np.fft.fft(tfr, axis=0)
    tfr = np.real(tfr)
    return tfr[:, ::2]


def test_radon_without_padding():
    """
    Test: skip the diagonal padding entirely. Just rotate the N×N
    WVD directly and sum. At 0° and 90° this should work perfectly
    since no corners are clipped for axis-aligned rotations.
    """
    N = 64
    f0 = 10
    n = np.arange(N)
    x = np.exp(1j * 2 * np.pi * f0 * n / N)
    
    wvd = wvd_upsampled(x, n_fbins=N)
    wvd_shifted = np.fft.fftshift(wvd, axes=0)
    
    print(f"\n  WVD shape: {wvd_shifted.shape}")
    
    # θ=0°: no rotation, sum along axis=0 → time profile
    proj_0 = wvd_shifted.sum(axis=0)
    peak_0 = np.argmax(np.abs(proj_0))
    print(f"  θ=0° (no rotation, sum axis=0): peak at col {peak_0}")
    
    # θ=90°: rotate 90° CCW, sum along axis=0
    rot90 = ndrotate(wvd_shifted, 90.0, reshape=False, order=3,
                     mode='constant', cval=0.0)
    proj_90 = rot90.sum(axis=0)
    peak_90 = np.argmax(np.abs(proj_90))
    print(f"  θ=90° (rotate 90, sum axis=0): peak at col {peak_90}")
    
    # Compare: at 90° the projection should be the freq marginal
    freq_marg = wvd_shifted.sum(axis=1)
    peak_fm = np.argmax(np.abs(freq_marg))
    print(f"  Direct freq marginal (sum axis=1): peak at row {peak_fm}")
    
    # Check if rot90 projection matches freq marginal
    proj_90_norm = np.abs(proj_90) / np.max(np.abs(proj_90))
    freq_marg_norm = np.abs(freq_marg) / np.max(np.abs(freq_marg))
    r = np.corrcoef(proj_90_norm, freq_marg_norm)[0, 1]
    print(f"  rot90 projection vs freq marginal: r = {r:.4f}")
    
    # Also check: does rot90 lose energy due to reshape=False?
    print(f"\n  Energy before rotation: {np.sum(np.abs(wvd_shifted)):.2f}")
    print(f"  Energy after  rotation: {np.sum(np.abs(rot90)):.2f}")
    ratio = np.sum(np.abs(rot90)) / np.sum(np.abs(wvd_shifted))
    print(f"  Energy ratio: {ratio:.4f}")


def test_radon_90_with_padding():
    """
    Test the full pipeline at 90° with padding.
    Compare what the padded Radon gives vs the direct freq marginal.
    """
    N = 64
    f0 = 10
    n = np.arange(N)
    x = np.exp(1j * 2 * np.pi * f0 * n / N)
    
    wvd = wvd_upsampled(x, n_fbins=N)
    wvd_shifted = np.fft.fftshift(wvd, axes=0)
    
    # Pad to diagonal
    M, Nc = wvd_shifted.shape
    D = int(np.ceil(np.sqrt(M**2 + Nc**2)))
    if D % 2 == 0:
        D += 1
    padded = np.zeros((D, D))
    pr, pc = (D-M)//2, (D-Nc)//2
    padded[pr:pr+M, pc:pc+Nc] = wvd_shifted
    
    print(f"\n  WVD: {M}×{Nc}, Padded: {D}×{D}")
    print(f"  Pad offsets: pr={pr}, pc={pc}")
    
    # Rotate 90° and project
    rot90 = ndrotate(padded, 90.0, reshape=False, order=3,
                     mode='constant', cval=0.0)
    proj_full = rot90.sum(axis=0)
    
    print(f"  proj_full length: {len(proj_full)}")
    print(f"  proj_full nonzero range: "
          f"{np.argmax(np.abs(proj_full) > 1e-10)} to "
          f"{len(proj_full) - np.argmax(np.abs(proj_full[::-1]) > 1e-10) - 1}")
    
    # Current extraction: centre crop N samples
    centre = D // 2
    half = N // 2
    start = centre - half
    proj_crop = proj_full[start: start + N]
    
    print(f"  Extracting [{start}:{start+N}] from proj_full")
    
    # The WVD data in the padded canvas occupies rows [pr..pr+M-1]
    # After 90° CCW rotation, those rows become columns [D-1-pr-M+1..D-1-pr]
    # = columns [D-pr-M..D-pr-1]
    # The projection (sum along axis=0) of the rotated image
    # will have non-zero values in those column positions.
    
    data_col_start = D - pr - M
    data_col_end = D - pr
    print(f"  After 90° rotation, data should be in cols [{data_col_start}:{data_col_end}]")
    print(f"  That's {data_col_end - data_col_start} columns = M = {M}")
    
    # The direct freq marginal (for comparison)
    freq_marg = wvd_shifted.sum(axis=1)
    
    # Try extracting from the correct column range
    proj_correct = proj_full[data_col_start:data_col_end]
    
    # Compare
    proj_crop_norm = np.abs(proj_crop) / np.max(np.abs(proj_crop))
    proj_correct_norm = np.abs(proj_correct) / np.max(np.abs(proj_correct))
    freq_marg_norm = np.abs(freq_marg) / np.max(np.abs(freq_marg))
    
    r_crop = np.corrcoef(proj_crop_norm, freq_marg_norm)[0, 1]
    r_correct = np.corrcoef(proj_correct_norm, freq_marg_norm)[0, 1]
    
    print(f"\n  Centre-crop vs freq marginal: r = {r_crop:.4f}")
    print(f"  Correct-range vs freq marginal: r = {r_correct:.4f}")
    
    # Also: is the projection reversed?
    r_correct_flip = np.corrcoef(proj_correct_norm[::-1], freq_marg_norm)[0, 1]
    print(f"  Correct-range FLIPPED vs freq marginal: r = {r_correct_flip:.4f}")


def test_rotation_convention():
    """
    Understand exactly what ndrotate does.
    Create a simple test image and see how it rotates.
    """
    # Create a 10×10 image with a single bright pixel at (row=2, col=7)
    img = np.zeros((10, 10))
    img[2, 7] = 1.0
    
    print("\n  Original image: bright pixel at (row=2, col=7)")
    
    # Rotate 90° CCW
    rot = ndrotate(img, 90.0, reshape=False, order=0, mode='constant', cval=0)
    bright = np.unravel_index(np.argmax(rot), rot.shape)
    print(f"  After 90° CCW: bright pixel at (row={bright[0]}, col={bright[1]})")
    # 90° CCW: (r,c) → (N-1-c, r) ... so (2,7) → (2, 2)? or (7,2)?
    
    # Rotate 90° CW (negative angle)
    rot_cw = ndrotate(img, -90.0, reshape=False, order=0, mode='constant', cval=0)
    bright_cw = np.unravel_index(np.argmax(rot_cw), rot_cw.shape)
    print(f"  After 90° CW:  bright pixel at (row={bright_cw[0]}, col={bright_cw[1]})")
    
    # What we need: after rotation, summing axis=0 should give 
    # the projection along the rotated direction
    
    # For a (freq × time) matrix with shape (M, N):
    # θ=0:  sum axis=0 → time marginal (project onto time axis)
    # θ=90: sum axis=0 of rotated → should give freq marginal
    #        which means freq axis (rows) must become columns after rotation
    
    # If we rotate CCW by 90°: rows become columns (going left)
    # Original col 7 → after CCW → row 7... 
    # Let me just check what sum(axis=0) gives
    
    print(f"\n  sum(axis=0) of rotated: non-zero at col {np.argmax(rot.sum(axis=0))}")
    print(f"  This should correspond to original row 2")