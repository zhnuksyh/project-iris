"""
src/models/gabor_baseline.py

Traditional Gabor-filter iris recognition baseline.

Pipeline:
  1. Build a bank of 2D Gabor filter pairs (real + imaginary quadrature) at
     multiple scales and orientations.
  2. Convolve a normalised iris image with each filter pair.
  3. Binarise the sign of each response → IrisCode (1-D binary array).
  4. Compare two IrisCodes with fractional Hamming Distance.

This module is the classical baseline against which IrisNet (Phase 3) is
evaluated in Phase 6.

Filter bank configuration (documented here for reproducibility):
  SCALES       = 4   (kernel sizes 9, 13, 17, 23 px; sigmas 2, 3, 4, 6)
  ORIENTATIONS = 8   (angles 0, π/8, 2π/8, …, 7π/8 radians)
  Total filter pairs: 32  (each pair = real cos + imaginary sin Gabor)
  IrisCode length: 32 pairs × 2 phases × 64 × 64 subsampled grid = 262,144 bits
"""

import cv2
import numpy as np
from typing import List, Tuple


# ── Filter-bank hyper-parameters ─────────────────────────────────────────────
SCALES       = 4
ORIENTATIONS = 8

_KERNEL_SIZES = [9,   13,   17,   23]
_SIGMAS       = [2.0,  3.0,  4.0,  6.0]
_LAMBDAS      = [5.0,  8.0, 11.0, 15.0]   # spatial wavelength (px)

# Responses are sub-sampled to this spatial resolution before binarisation.
# Reduces code length and removes highly correlated adjacent pixels.
_SUBSAMPLE = 2   # keep every 2nd row and column → 64×64 from 128×128


def build_gabor_filters() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build a bank of 32 quadrature Gabor kernel pairs.

    Each pair is (kernel_real, kernel_imag):
      - kernel_real : psi = 0        → cosine response (even-symmetric)
      - kernel_imag : psi = π/2      → sine response (odd-symmetric)

    Kernels are L2-normalised so response magnitudes are comparable.

    Returns:
        List of 32 (kernel_real, kernel_imag) tuples, ordered
        [scale0_orient0, scale0_orient1, …, scale3_orient7].
    """
    filter_pairs = []
    for si in range(SCALES):
        ksize = _KERNEL_SIZES[si]
        sigma = _SIGMAS[si]
        lam   = _LAMBDAS[si]
        for oi in range(ORIENTATIONS):
            theta = oi * np.pi / ORIENTATIONS

            k_real = cv2.getGaborKernel(
                ksize=(ksize, ksize), sigma=sigma, theta=theta,
                lambd=lam, gamma=0.5, psi=0.0, ktype=cv2.CV_32F,
            )
            k_imag = cv2.getGaborKernel(
                ksize=(ksize, ksize), sigma=sigma, theta=theta,
                lambd=lam, gamma=0.5, psi=np.pi / 2, ktype=cv2.CV_32F,
            )

            # L2 normalise so response amplitudes are scale-independent
            k_real /= (np.sqrt((k_real ** 2).sum()) + 1e-8)
            k_imag /= (np.sqrt((k_imag ** 2).sum()) + 1e-8)

            filter_pairs.append((k_real, k_imag))
    return filter_pairs


# Build once at import time
_FILTER_BANK: List[Tuple[np.ndarray, np.ndarray]] = build_gabor_filters()


def extract_iris_code(normalized_iris_array: np.ndarray) -> np.ndarray:
    """Extract a binary IrisCode from a normalised iris image.

    For each quadrature filter pair the image is convolved with both the real
    (cos) and imaginary (sin) kernel.  The sign of each response is binarised
    after subsampling by a factor of 2 to remove correlated adjacent bits.

    Args:
        normalized_iris_array: shape (128, 128) or (128, 128, 1), float32
                               values in [0.0, 1.0].

    Returns:
        1-D bool numpy array of length
        SCALES * ORIENTATIONS * 2 * (128//_SUBSAMPLE)**2
        = 32 * 2 * 64 * 64 = 262,144 bits.
    """
    if normalized_iris_array.ndim == 3:
        img = normalized_iris_array[:, :, 0]
    else:
        img = normalized_iris_array.copy()
    img = img.astype(np.float32)

    # Centre the image so it has zero mean — required for unbiased Gabor responses.
    # Without this, the [0,1] pixel range creates a positive DC offset in every
    # filter response, pushing ~75% of bits to 1 and destroying discriminability.
    img = img - img.mean()

    code_parts = []
    for k_real, k_imag in _FILTER_BANK:
        resp_real = cv2.filter2D(img, cv2.CV_32F, k_real)
        resp_imag = cv2.filter2D(img, cv2.CV_32F, k_imag)

        # Subsample to decorrelate adjacent pixels
        resp_real = resp_real[::_SUBSAMPLE, ::_SUBSAMPLE]
        resp_imag = resp_imag[::_SUBSAMPLE, ::_SUBSAMPLE]

        # Binarise: True where response > 0
        code_parts.append(resp_real > 0)
        code_parts.append(resp_imag > 0)

    return np.stack(code_parts, axis=0).ravel()


def calculate_hamming_distance(code1: np.ndarray, code2: np.ndarray) -> float:
    """Compute the fractional Hamming Distance between two binary IrisCodes.

    HD = (number of mismatching bits) / (total bits compared)

    Interpretation:
      HD ≈ 0.00–0.32 : likely same iris (genuine match)
      HD ≈ 0.32–0.50 : likely different irises (impostor)
      HD ≈ 0.50      : statistically independent (random chance)

    Args:
        code1: 1-D bool/int numpy array.
        code2: 1-D bool/int numpy array of the same length.

    Returns:
        Fractional Hamming Distance in [0.0, 1.0].

    Raises:
        ValueError: if code lengths do not match.
    """
    if code1.shape != code2.shape:
        raise ValueError(
            f"IrisCode length mismatch: {code1.shape} vs {code2.shape}"
        )
    mismatches = np.count_nonzero(code1.astype(bool) ^ code2.astype(bool))
    return float(mismatches) / code1.size
