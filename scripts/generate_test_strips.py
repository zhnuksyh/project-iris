"""
scripts/generate_test_strips.py

Regenerates the (64, 512) Daugman rubber-sheet strip for each test image
referenced in the closed-set and open-set splits, saving to
`data/processed_strip/<same-subpath>.npy`.

The standard preprocessing pipeline downsamples the strip to (128, 128),
which destroys the 8:1 angular/radial aspect ratio that Gabor filters
rely on. This script preserves the original strip so the Gabor baseline
can be evaluated on its native input domain — a fairer comparison than
the resized version reported in Phase 6.

Usage
-----
    python -m scripts.generate_test_strips
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

from src.preprocessing.segmentation import (
    denoise_image, segment_iris, normalize_iris,
)

PROCESSED_ROOT = 'data/processed'
RAW_ROOT       = 'data/raw'
STRIP_ROOT     = 'data/processed_strip'

SUBSET_RAW_PREFIX = {
    'CASIA-Iris-Interval':  'CASIA-Iris-Interval',
    'CASIA-Iris-Lamp':      'CASIA-Iris-Lamp',
    'CASIA-Iris-Syn':       '',
    'CASIA-Iris-Thousand':  'CASIA-Iris-Thousand/CASIA-Iris-Thousand',
}


def _resolve_raw_path(processed_path: str) -> Optional[str]:
    """Map a processed .npy path back to its source .jpg in data/raw.

    Layouts vary per subset (some are nested twice, some once, Syn is flat),
    so we strip the processed root and prepend the subset-specific prefix.
    """
    rel = os.path.relpath(processed_path, PROCESSED_ROOT)
    parts = rel.split(os.sep)
    subset = parts[0]
    rest_parts = parts[1:]

    if subset not in SUBSET_RAW_PREFIX:
        return None

    # Drop any embedded duplicate of the subset name in the processed path
    # (Thousand's processed path is "CASIA-Iris-Thousand/CASIA-Iris-Thousand/...").
    while rest_parts and rest_parts[0] == subset:
        rest_parts = rest_parts[1:]

    rest = os.sep.join(rest_parts).replace('.npy', '.jpg')
    extra = SUBSET_RAW_PREFIX[subset]
    raw = os.path.join(RAW_ROOT, subset, extra, rest) if extra \
          else os.path.join(RAW_ROOT, subset, rest)
    return raw if os.path.isfile(raw) else None


def _strip_output_path(processed_path: str) -> str:
    rel = os.path.relpath(processed_path, PROCESSED_ROOT)
    return os.path.join(STRIP_ROOT, rel)


def _process_one(processed_path: str) -> tuple:
    """Returns (status, processed_path). status in {'ok','exists','no_raw','seg_fail','error'}."""
    out_path = _strip_output_path(processed_path)
    if os.path.isfile(out_path):
        return ('exists', processed_path)

    raw_path = _resolve_raw_path(processed_path)
    if raw_path is None:
        return ('no_raw', processed_path)

    try:
        denoised = denoise_image(raw_path)
        circles  = segment_iris(denoised)
        if circles is None:
            return ('seg_fail', processed_path)
        strip = normalize_iris(denoised, circles, circles, width=512, height=64)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, strip)
        return ('ok', processed_path)
    except Exception as exc:
        return (f'error:{exc}', processed_path)


def _collect_test_paths() -> list:
    paths = set()
    for split_file in ('data/test_split.json', 'data/test_split_openset.json'):
        if not os.path.isfile(split_file):
            continue
        with open(split_file) as f:
            d = json.load(f)
        for s in d['samples']:
            paths.add(s['path'])
    return sorted(paths)


def main():
    paths = _collect_test_paths()
    print(f'Generating (64, 512) strips for {len(paths)} test images')
    print(f'Output: {STRIP_ROOT}/<subset>/...')

    counts = {'ok': 0, 'exists': 0, 'no_raw': 0, 'seg_fail': 0, 'error': 0}
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(_process_one, p): p for p in paths}
        done = 0
        for fut in as_completed(futures):
            status, p = fut.result()
            key = status if status in counts else 'error'
            counts[key] += 1
            done += 1
            if done % 500 == 0:
                print(f'  [{done}/{len(paths)}] ok={counts["ok"]} '
                      f'cached={counts["exists"]} fail={counts["seg_fail"]+counts["error"]+counts["no_raw"]}')

    print('\nSummary:')
    for k, v in counts.items():
        print(f'  {k:10s} {v}')


if __name__ == '__main__':
    main()
