"""
src/utils/metrics.py

Evaluation metrics for iris biometric authentication.

Metrics implemented:
  - FAR  (False Acceptance Rate): Fraction of impostor pairs incorrectly accepted.
  - FRR  (False Rejection Rate):  Fraction of genuine pairs incorrectly rejected.
  - TAR  (True Acceptance Rate):  1 - FRR; fraction of genuine pairs correctly accepted.
  - EER  (Equal Error Rate):      Threshold where FAR == FRR; lower is better.

All functions accept **similarity scores** (higher = more similar).
Gabor Hamming Distances must be converted to similarity (1 - HD) before use.
"""

import numpy as np


def compute_far_frr_curve(genuine_scores: np.ndarray,
                          impostor_scores: np.ndarray,
                          num_thresholds: int = 1000):
    """Sweep thresholds and compute FAR/FRR at each.

    Args:
        genuine_scores:  1-D array of similarity scores for genuine pairs.
        impostor_scores: 1-D array of similarity scores for impostor pairs.
        num_thresholds:  number of evenly spaced thresholds to evaluate.

    Returns:
        (thresholds, far, frr) — each a 1-D numpy array of length num_thresholds.
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    far = np.array([np.mean(impostor_scores >= t) for t in thresholds])
    frr = np.array([np.mean(genuine_scores < t) for t in thresholds])

    return thresholds, far, frr


def compute_eer(genuine_scores: np.ndarray,
                impostor_scores: np.ndarray,
                num_thresholds: int = 10000):
    """Find the Equal Error Rate — the threshold where FAR ≈ FRR.

    Uses linear interpolation on the FAR-FRR difference curve for sub-step
    precision.

    Returns:
        (eer, threshold) — EER as a fraction in [0, 1] and the corresponding
        decision threshold.
    """
    thresholds, far, frr = compute_far_frr_curve(
        genuine_scores, impostor_scores, num_thresholds
    )
    diff = far - frr
    # Find the crossing point where diff changes sign (FAR goes below FRR)
    idx = np.argmin(np.abs(diff))

    # Linear interpolation between the two nearest points
    if idx > 0 and diff[idx - 1] * diff[idx] < 0:
        # Interpolate between idx-1 and idx
        w = abs(diff[idx - 1]) / (abs(diff[idx - 1]) + abs(diff[idx]))
        eer = far[idx - 1] + w * (far[idx] - far[idx - 1])
        threshold = thresholds[idx - 1] + w * (thresholds[idx] - thresholds[idx - 1])
    else:
        eer = (far[idx] + frr[idx]) / 2.0
        threshold = thresholds[idx]

    return float(eer), float(threshold)


def compute_tar_at_far(genuine_scores: np.ndarray,
                       impostor_scores: np.ndarray,
                       target_far: float = 0.01,
                       num_thresholds: int = 10000):
    """Find the True Acceptance Rate at a given FAR operating point.

    Args:
        target_far: desired FAR operating point (e.g. 0.01 for FAR=1%).

    Returns:
        (tar, threshold) — TAR (= 1 - FRR) at the threshold closest to target_far.
    """
    thresholds, far, frr = compute_far_frr_curve(
        genuine_scores, impostor_scores, num_thresholds
    )
    # Find highest threshold where FAR <= target_far
    valid = far <= target_far
    if not np.any(valid):
        # All thresholds have FAR > target_far
        idx = np.argmin(far)
    else:
        # Among valid thresholds, pick the one with lowest FRR (= highest TAR)
        candidates = np.where(valid)[0]
        idx = candidates[np.argmin(frr[candidates])]

    tar = 1.0 - frr[idx]
    return float(tar), float(thresholds[idx])


def build_roc_curve(genuine_scores: np.ndarray,
                    impostor_scores: np.ndarray,
                    num_thresholds: int = 1000):
    """Build ROC curve data (FPR vs TPR).

    Returns:
        (fpr, tpr) — 1-D arrays. fpr = FAR, tpr = 1 - FRR.
    """
    thresholds, far, frr = compute_far_frr_curve(
        genuine_scores, impostor_scores, num_thresholds
    )
    return far, 1.0 - frr


def build_det_curve(genuine_scores: np.ndarray,
                    impostor_scores: np.ndarray,
                    num_thresholds: int = 1000):
    """Build DET curve data (FPR vs FNR).

    Returns:
        (fpr, fnr) — 1-D arrays. fpr = FAR, fnr = FRR.
    """
    thresholds, far, frr = compute_far_frr_curve(
        genuine_scores, impostor_scores, num_thresholds
    )
    return far, frr
