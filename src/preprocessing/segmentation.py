"""
src/preprocessing/segmentation.py

Iris segmentation and normalization pipeline for IrisNet.

Pipeline (in order):
  1. denoise_image    — load + median blur + gaussian blur
  2. segment_iris     — two-phase HoughCircles (pupil then iris) with fallback
  3. normalize_iris   — Daugman Rubber-Sheet Model via cv2.remap
  4. scale_pixels     — resize to (128,128), float32 [0,1], expand to (128,128,1)
"""

import cv2
import numpy as np
from typing import Optional


def denoise_image(image_path: str) -> np.ndarray:
    """Load a grayscale iris image and apply noise reduction.

    Applies median blur first to suppress salt-and-pepper noise and specular
    reflections (common in CASIA-Iris-Lamp), then Gaussian blur to smooth
    remaining high-frequency noise.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the .jpg iris image.

    Returns
    -------
    np.ndarray
        Denoised grayscale image as a uint8 numpy array.

    Raises
    ------
    FileNotFoundError
        If the image cannot be loaded from image_path.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.medianBlur(image, ksize=5)
    image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1.5)
    return image


def segment_iris(blurred_image: np.ndarray) -> Optional[dict]:
    """Detect pupil and iris boundaries using two-phase HoughCircles.

    Performs two independent Hough circle detections:
      - Phase 1: pupil (inner boundary) — small, dark circle
      - Phase 2: iris  (outer boundary) — larger circle surrounding the pupil

    Each phase uses a fallback loop that progressively relaxes param2
    (circle accumulator threshold) until circles are found.

    A sanity check validates that r_iris > r_pupil and that the two centres
    are within 25 pixels of each other.

    Parameters
    ----------
    blurred_image : np.ndarray
        Denoised grayscale image (output of denoise_image).

    Returns
    -------
    dict or None
        On success: {"center": (cx, cy), "r_pupil": float, "r_iris": float}
        On failure (no valid circles found): None
    """
    def _detect(image, dp, min_dist, p1, p2_start, p2_min, p2_step, min_r, max_r):
        for p2 in range(p2_start, p2_min - 1, p2_step):
            circles = cv2.HoughCircles(
                image,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=min_dist,
                param1=p1,
                param2=p2,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None:
                return np.round(circles[0, :]).astype(int)
        return None

    # --- Phase 1: Pupil ---
    pupil_circles = _detect(
        blurred_image,
        dp=1.5, min_dist=50,
        p1=200, p2_start=50, p2_min=15, p2_step=-5,
        min_r=10, max_r=80,
    )
    if pupil_circles is None:
        return None

    px, py, r_pupil = pupil_circles[0]

    # --- Phase 2: Iris ---
    iris_circles = _detect(
        blurred_image,
        dp=1.5, min_dist=50,
        p1=100, p2_start=30, p2_min=10, p2_step=-5,
        min_r=80, max_r=150,
    )
    if iris_circles is None:
        return None

    ix, iy, r_iris = iris_circles[0]

    # --- Sanity check ---
    center_dist = np.sqrt((px - ix) ** 2 + (py - iy) ** 2)
    if r_iris <= r_pupil or center_dist > 25:
        return None

    # Use pupil centre as the canonical centre (more stable)
    return {"center": (int(px), int(py)), "r_pupil": float(r_pupil), "r_iris": float(r_iris)}


def normalize_iris(
    image: np.ndarray,
    pupil_circle: dict,
    iris_circle: dict,
    width: int = 512,
    height: int = 64,
) -> np.ndarray:
    """Unwrap the annular iris region into a rectangular strip.

    Implements Daugman's Rubber-Sheet Model: maps each point in the annular
    region between the pupil and iris boundaries to a point on a 2-D polar
    coordinate grid (rho, theta), producing a fixed-size rectangular image.

    Mapping formula:
        x(rho, theta) = x_pupil(theta) + rho * (x_iris(theta) - x_pupil(theta))
        y(rho, theta) = y_pupil(theta) + rho * (y_iris(theta) - y_pupil(theta))

    where rho ∈ [0, 1] indexes rows (radial direction) and
    theta ∈ [0, 2π) indexes columns (angular direction).

    Parameters
    ----------
    image : np.ndarray
        Denoised grayscale image.
    pupil_circle : dict
        Dict with keys "center" (cx, cy) and "r_pupil".
    iris_circle : dict
        Dict with keys "center" (cx, cy) and "r_iris".
        In practice both dicts come from segment_iris and share the same centre.
    width : int
        Number of angular samples (columns). Default 512.
    height : int
        Number of radial samples (rows). Default 64.

    Returns
    -------
    np.ndarray
        Normalised iris strip of shape (height, width), dtype uint8.
    """
    cx, cy = pupil_circle["center"]
    r_pupil = pupil_circle["r_pupil"]
    r_iris = iris_circle["r_iris"]

    # Build polar coordinate meshgrid
    theta = np.linspace(0, 2 * np.pi, width, endpoint=False)   # (width,)
    rho   = np.linspace(0, 1, height, endpoint=False)           # (height,)
    theta_grid, rho_grid = np.meshgrid(theta, rho)              # (height, width)

    # Boundary points on pupil and iris circumferences
    x_pupil = cx + r_pupil * np.cos(theta_grid)
    y_pupil = cy + r_pupil * np.sin(theta_grid)
    x_iris  = cx + r_iris  * np.cos(theta_grid)
    y_iris  = cy + r_iris  * np.sin(theta_grid)

    # Interpolated Cartesian coordinates for each (rho, theta)
    map_x = (x_pupil + rho_grid * (x_iris - x_pupil)).astype(np.float32)
    map_y = (y_pupil + rho_grid * (y_iris - y_pupil)).astype(np.float32)

    normalized = cv2.remap(
        image,
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return normalized  # (height, width) uint8


def scale_pixels(
    normalized_image: np.ndarray,
    target_shape: tuple = (128, 128),
) -> np.ndarray:
    """Resize the normalised iris strip and convert to a CNN-ready tensor.

    Steps:
      1. Resize to target_shape using bilinear interpolation.
      2. Cast to float32 and divide by 255 → [0.0, 1.0].
      3. Expand the channel dimension → (H, W, 1).

    Parameters
    ----------
    normalized_image : np.ndarray
        Output of normalize_iris, shape (64, 512) uint8.
    target_shape : tuple of int
        (width, height) for cv2.resize. Default (128, 128).

    Returns
    -------
    np.ndarray
        Float32 array of shape (128, 128, 1) with values in [0.0, 1.0].
    """
    resized = cv2.resize(normalized_image, target_shape, interpolation=cv2.INTER_LINEAR)
    scaled  = resized.astype(np.float32) / 255.0
    return np.expand_dims(scaled, axis=-1)
