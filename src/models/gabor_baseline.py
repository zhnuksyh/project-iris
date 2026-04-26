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

Two encoders are exposed:
  - extract_iris_code(img128)        — operates on the (128, 128) resized
    tensor (Phase 6 default; carries an aspect-ratio penalty noted in
    Section 10.2 of the evaluation report)
  - extract_iris_code_strip(strip)   — operates on the native (64, 512)
    Daugman rubber-sheet strip (Phase 7+; the fairer classical baseline)

Filter bank configuration (documented here for reproducibility):
  SCALES       = 4   (kernel sizes 9, 13, 17, 23 px; sigmas 2, 3, 4, 6)
  ORIENTATIONS = 8   (angles 0, π/8, 2π/8, …, 7π/8 radians)
  Total filter pairs: 32  (each pair = real cos + imaginary sin Gabor)
  IrisCode length (128² input):  32×2 × 64×64    = 262,144 bits
  IrisCode length (64×512 input): 32×2 × 32×256  = 524,288 bits
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


def extract_iris_code_strip(strip: np.ndarray) -> np.ndarray:
    """Extract a binary IrisCode from a (64, 512) Daugman rubber-sheet strip.

    Naive implementation that applies the 128x128 Gabor bank directly to
    the strip with no geometric adaptation. Retained for ablation against
    the engineered v2 below; v2 (`extract_iris_code_strip_v2`) is the one
    used in evaluation.

    Returns 524,288 bits with no occlusion mask.
    """
    if strip.ndim == 3:
        strip = strip[:, :, 0]
    img = strip.astype(np.float32)
    if img.max() > 1.5:  # uint8 path
        img = img / 255.0
    img = img - img.mean()

    code_parts = []
    for k_real, k_imag in _FILTER_BANK:
        resp_real = cv2.filter2D(img, cv2.CV_32F, k_real)
        resp_imag = cv2.filter2D(img, cv2.CV_32F, k_imag)
        resp_real = resp_real[::_SUBSAMPLE, ::_SUBSAMPLE]
        resp_imag = resp_imag[::_SUBSAMPLE, ::_SUBSAMPLE]
        code_parts.append(resp_real > 0)
        code_parts.append(resp_imag > 0)
    return np.stack(code_parts, axis=0).ravel()


# ── Strip encoder v2: cyclic wrap + occlusion mask ────────────────────────────
#
# Re-uses the existing 32-pair filter bank (4 scales x 8 orientations) but
# applies it on the strip with two corrections that the naive variant lacks:
#
#   1. Cyclic convolution on the angular axis. The strip is an unwrapped
#      polar map: theta=0 and theta=2pi are the same iris column, so
#      filter responses near the seam should wrap, not reflect.
#   2. Eyelid occlusion mask. The upper-eyelid sector (~25% of angles
#      centered at 12 o'clock) routinely contains lashes/skin rather
#      than iris texture; bits in that sector are excluded from the
#      Hamming distance.
#
# An anisotropic-kernel variant was also tried (4 orientations, gamma=2);
# it hurt the genuine/impostor gap on a single-pair sanity check, so we
# keep the original isotropic bank and rely on geometry corrections.
_STRIP_SUBSAMPLE   = 2

# In normalize_iris(), theta=0 is at 3 o'clock and theta increases toward
# the bottom (sin > 0 -> +y in cv2 image coords). Hence theta = 3pi/2 is
# the top of the iris (12 o'clock, column 384 of 512). Mask +/- 60 deg
# around that = columns 320..448 (128 cols, 25%).
_EYELID_MASK_START = 320
_EYELID_MASK_END   = 448


def _filter_strip_cyclic(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve img with kernel using cyclic padding on the angular (column)
    axis and reflect-padding on the radial (row) axis."""
    pad_x = kernel.shape[1] // 2
    pad_y = kernel.shape[0] // 2
    padded = np.concatenate([img[:, -pad_x:], img, img[:, :pad_x]], axis=1)
    padded = cv2.copyMakeBorder(padded, pad_y, pad_y, 0, 0,
                                borderType=cv2.BORDER_REFLECT_101)
    full = cv2.filter2D(padded, cv2.CV_32F, kernel,
                        borderType=cv2.BORDER_CONSTANT)
    return full[pad_y:pad_y + img.shape[0], pad_x:pad_x + img.shape[1]]


def extract_iris_code_strip_v2(strip: np.ndarray) -> tuple:
    """Engineered strip-Gabor encoder.

    Re-uses the 32-filter Gabor bank from the (128, 128) baseline but
    corrects two strip-specific issues the naive `extract_iris_code_strip`
    lacks:
      * Cyclic convolution on the angular axis (theta=0 == theta=2pi).
      * Eyelid occlusion mask (the upper-eyelid sector is dropped from
        the Hamming distance via `calculate_hamming_distance_masked`).

    Args:
        strip: shape (64, 512), uint8 in [0,255] or float in [0,1].

    Returns:
        (code, mask) where
          code: 1-D bool array, length 32 * 32 * 256 = 524,288
          mask: 1-D bool array of same length, True = valid bit, False =
                occluded (skip in Hamming distance).
    """
    if strip.ndim == 3:
        strip = strip[:, :, 0]
    img = strip.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    img = img - img.mean()

    code_parts = []
    for k_real, k_imag in _FILTER_BANK:
        resp_real = _filter_strip_cyclic(img, k_real)
        resp_imag = _filter_strip_cyclic(img, k_imag)
        resp_real = resp_real[::_STRIP_SUBSAMPLE, ::_STRIP_SUBSAMPLE]
        resp_imag = resp_imag[::_STRIP_SUBSAMPLE, ::_STRIP_SUBSAMPLE]
        code_parts.append(resp_real > 0)
        code_parts.append(resp_imag > 0)
    code = np.stack(code_parts, axis=0)            # (P, H', W')
    P, Hs, Ws = code.shape

    # Eyelid mask: column-level mask broadcast across (P, H', W')
    col_mask = np.ones(Ws, dtype=bool)
    start_sub = _EYELID_MASK_START // _STRIP_SUBSAMPLE
    end_sub   = _EYELID_MASK_END   // _STRIP_SUBSAMPLE
    col_mask[start_sub:end_sub] = False
    mask = np.broadcast_to(col_mask[None, None, :], (P, Hs, Ws)).copy()

    return code.ravel(), mask.ravel()


def calculate_hamming_distance_masked(code1: np.ndarray, mask1: np.ndarray,
                                      code2: np.ndarray, mask2: np.ndarray) -> float:
    """Fractional Hamming Distance over bits valid in BOTH masks.

    HD = #(c1 != c2 within m1 & m2) / #(m1 & m2)

    If the intersection mask is empty, returns 1.0 (treat as max distance).
    """
    valid = mask1.astype(bool) & mask2.astype(bool)
    n = int(valid.sum())
    if n == 0:
        return 1.0
    diff = np.count_nonzero((code1.astype(bool) ^ code2.astype(bool)) & valid)
    return float(diff) / n


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
