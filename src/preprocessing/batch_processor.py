"""
src/preprocessing/batch_processor.py

Batch preprocessing script for all CASIA-IrisV4 dataset subsets.

Walks each dataset directory, applies the full preprocessing pipeline
(denoise -> segment -> normalize -> scale), and saves the resulting
(128, 128, 1) float32 tensors as .npy files under data/processed/,
mirroring the source directory tree.

Usage (from project root, with venv active):
    python -m src.preprocessing.batch_processor

Output layout mirrors the source tree:
    data/raw/CASIA-Iris-Interval/CASIA-Iris-Interval/001/L/S1001L01.jpg
    -> data/processed/CASIA-Iris-Interval/001/L/S1001L01.npy
"""

import os
import sys
import numpy as np

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.segmentation import (
    denoise_image,
    normalize_iris,
    scale_pixels,
    segment_iris,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUBSETS = {
    "CASIA-Iris-Interval": os.path.join("data", "raw", "CASIA-Iris-Interval", "CASIA-Iris-Interval"),
    "CASIA-Iris-Lamp":     os.path.join("data", "raw", "CASIA-Iris-Lamp",     "CASIA-Iris-Lamp"),
    "CASIA-Iris-Thousand": os.path.join("data", "raw", "CASIA-Iris-Thousand", "CASIA-Iris-Thousand"),
    "CASIA-Iris-Syn":      os.path.join("data", "raw", "CASIA-Iris-Syn"),
}

OUTPUT_ROOT = os.path.join("data", "processed")


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

def process_subset(subset_name: str, subset_root: str) -> tuple:
    """Process all .jpg images in one dataset subset.

    Parameters
    ----------
    subset_name : str
        Human-readable name (used as the output subdirectory).
    subset_root : str
        Path to the root of the subset's image directory.

    Returns
    -------
    (processed_count, skipped_count) : tuple of int
    """
    if not os.path.isdir(subset_root):
        print(f"  [WARN] Directory not found, skipping: {subset_root}")
        return 0, 0

    output_subset_root = os.path.join(OUTPUT_ROOT, subset_name)
    processed = 0
    skipped = 0

    for dirpath, _, filenames in os.walk(subset_root):
        jpg_files = [f for f in filenames if f.lower().endswith(".jpg")]
        for filename in jpg_files:
            img_path = os.path.join(dirpath, filename)

            # Mirror relative path from subset root into output tree
            rel_path = os.path.relpath(img_path, subset_root)
            rel_npy  = os.path.splitext(rel_path)[0] + ".npy"
            out_path = os.path.join(output_subset_root, rel_npy)
            out_dir  = os.path.dirname(out_path)

            try:
                os.makedirs(out_dir, exist_ok=True)

                denoised = denoise_image(img_path)
                circles  = segment_iris(denoised)

                if circles is None:
                    print(f"  [SKIP] Segmentation failed: {img_path}")
                    skipped += 1
                    continue

                normalized = normalize_iris(denoised, circles, circles)
                tensor     = scale_pixels(normalized)

                np.save(out_path, tensor)
                processed += 1

            except Exception as exc:
                print(f"  [ERROR] {img_path}: {exc}")
                skipped += 1

    return processed, skipped


def run_all():
    """Process every subset and print a summary."""
    print("=" * 60)
    print("Iris Preprocessing — Batch Processor")
    print("=" * 60)

    total_processed = 0
    total_skipped   = 0

    for subset_name, subset_root in SUBSETS.items():
        print(f"\n[{subset_name}]")
        print(f"  Source : {subset_root}")
        p, s = process_subset(subset_name, subset_root)
        print(f"  Result : Processed={p}  Skipped={s}")
        total_processed += p
        total_skipped   += s

    print("\n" + "=" * 60)
    print(f"GRAND TOTAL  Processed={total_processed}  Skipped={total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
