"""
src/utils/data_loader.py

Dataset discovery, stratified splitting, and tf.data pipeline for IrisNet.

Usage
-----
    from src.utils.data_loader import build_datasets, NUM_CLASSES

    train_ds, val_ds, test_ds = build_datasets(
        processed_root='data/processed',
        batch_size=32,
        arcface=False,   # True → yields (image, label_onehot) for ArcFace
    )

Design decisions
----------------
* Class label = one unique (subset, subject, eye) folder path  →  up to 4 115
  classes from the 30 626 preprocessed tensors.
* Identities with only 1 file are added to train only (cannot validate/test).
* Identities with exactly 2 files get train + test (skip val); val gets
  duplicated from train to avoid an empty dataset for that identity.
* Identities with >= 3 files receive a proper stratified 70/20/10 split.
* Files are loaded lazily via tf.data.Dataset.map so nothing is pre-loaded
  into RAM.
* The exact test split is serialised to data/test_split.json for Phase 6.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.20
# TEST_FRAC  = 0.10  (remainder)

SEED = 42
IMG_SHAPE = (128, 128, 1)

TEST_SPLIT_PATH = 'data/test_split.json'


# ── 1. Discover all .npy files and assign integer class labels ────────────────

def _discover(processed_root: str):
    """Walk processed_root and return (paths, int_labels, label_to_idx).

    A 'class' is the subdirectory directly under processed_root that
    contains .npy files (e.g. CASIA-Iris-Interval/001/L).
    """
    root = Path(processed_root)
    # Map identity folder → sorted list of .npy paths
    identity_files: dict = {}
    for path in sorted(root.rglob('*.npy')):
        identity = path.parent.relative_to(root).as_posix()
        identity_files.setdefault(identity, []).append(str(path))

    # Stable, sorted label assignment
    sorted_identities = sorted(identity_files.keys())
    label_to_idx = {ident: idx for idx, ident in enumerate(sorted_identities)}

    all_paths: List[str] = []
    all_labels: List[int] = []
    for ident in sorted_identities:
        files = sorted(identity_files[ident])
        lbl = label_to_idx[ident]
        all_paths.extend(files)
        all_labels.extend([lbl] * len(files))

    return all_paths, all_labels, label_to_idx, identity_files


def _stratified_split(identity_files: dict, label_to_idx: dict, rng: random.Random):
    """Return train/val/test lists of (path, label) tuples with stratified split.

    Rules:
      >=  3 samples: 70/20/10 per-identity (at least 1 per split)
      == 2 samples: 1 train, 1 test  (val borrows the train sample)
      == 1 sample:  train only
    """
    train, val, test = [], [], []
    for ident, files in identity_files.items():
        files = sorted(files)
        lbl = label_to_idx[ident]
        n = len(files)
        shuffled = files[:]
        rng.shuffle(shuffled)

        if n == 1:
            train.append((shuffled[0], lbl))
        elif n == 2:
            train.append((shuffled[0], lbl))
            test.append((shuffled[1], lbl))
        else:
            n_test  = max(1, round(n * (1 - TRAIN_FRAC - VAL_FRAC)))
            n_val   = max(1, round(n * VAL_FRAC))
            n_train = n - n_val - n_test

            train_files = shuffled[:n_train]
            val_files   = shuffled[n_train:n_train + n_val]
            test_files  = shuffled[n_train + n_val:]

            train.extend([(f, lbl) for f in train_files])
            val.extend(  [(f, lbl) for f in val_files])
            test.extend( [(f, lbl) for f in test_files])

    return train, val, test


# ── 2. tf.data.Dataset factory ────────────────────────────────────────────────

def _make_tf_dataset(
    samples: List[Tuple[str, int]],
    num_classes: int,
    batch_size: int,
    augment: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    """Build a batched tf.data.Dataset from a list of (path, label) pairs.

    Labels are one-hot float32 in all cases — both the softmax head and the
    ArcFace head consume CategoricalCrossentropy, so the format is identical.

    Args:
        samples:     list of (npy_path_str, int_label)
        num_classes: total number of identity classes
        batch_size:  samples per batch
        augment:     apply RandomRotation augmentation
        shuffle:     shuffle before each epoch
    """
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    def load_npy(path):
        """Load one (128,128,1) float32 tensor from an .npy file."""
        img = tf.numpy_function(
            func=lambda p: np.load(p.decode()).astype(np.float32),
            inp=[path],
            Tout=tf.float32,
        )
        img.set_shape(IMG_SHAPE)
        return img

    img_ds = path_ds.map(load_npy, num_parallel_calls=tf.data.AUTOTUNE)

    # One-hot labels (both softmax and arcface use categorical cross-entropy)
    def to_onehot(lbl):
        return tf.one_hot(lbl, depth=num_classes, dtype=tf.float32)

    label_oh_ds = label_ds.map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)

    ds = tf.data.Dataset.zip((img_ds, label_oh_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(samples), 4096), seed=SEED,
                        reshuffle_each_iteration=True)

    if augment:
        augmentor = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(factor=0.05, fill_mode='reflect'),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05),
                fill_mode='reflect',
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.03,
                width_factor=0.03,
                fill_mode='reflect',
            ),
            tf.keras.layers.GaussianNoise(0.01),
        ])
        ds = ds.map(
            lambda x, y: (augmentor(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── 3. Public API ─────────────────────────────────────────────────────────────

# Global — set after first build_datasets() call
NUM_CLASSES: int = 0


def build_datasets(
    processed_root: str = 'data/processed',
    batch_size: int = 32,
    test_split_path: str = TEST_SPLIT_PATH,
    min_samples: int = 1,
):
    """Discover data, split, and return three tf.data.Dataset objects.

    Labels are always one-hot float32; both the softmax head and the ArcFace
    head consume CategoricalCrossentropy so no format difference is needed.

    Args:
        processed_root:  path to data/processed/
        batch_size:      batch size for all three splits
        test_split_path: where to write/read the test split JSON file
        min_samples:     minimum images per identity to include (default 1 =
                         keep all; set to 2 for ArcFace to exclude singletons)

    Returns:
        (train_ds, val_ds, test_ds, num_classes)
    """
    global NUM_CLASSES

    _, _, label_to_idx, identity_files = _discover(processed_root)

    # Filter out identities with fewer than min_samples images
    if min_samples > 1:
        identity_files = {k: v for k, v in identity_files.items()
                          if len(v) >= min_samples}
        sorted_identities = sorted(identity_files.keys())
        label_to_idx = {ident: idx for idx, ident in enumerate(sorted_identities)}
        print(f'[data_loader] Filtered to identities with >= {min_samples} samples')

    num_classes = len(label_to_idx)
    NUM_CLASSES = num_classes

    rng = random.Random(SEED)
    train_samples, val_samples, test_samples = _stratified_split(
        identity_files, label_to_idx, rng
    )

    # Persist test split for Phase 6 evaluation
    _save_test_split(test_samples, label_to_idx, test_split_path)

    print(f'[data_loader] Classes      : {num_classes}')
    print(f'[data_loader] Train samples: {len(train_samples)}')
    print(f'[data_loader] Val   samples: {len(val_samples)}')
    print(f'[data_loader] Test  samples: {len(test_samples)}')

    train_ds = _make_tf_dataset(train_samples, num_classes, batch_size,
                                augment=True,  shuffle=True)
    val_ds   = _make_tf_dataset(val_samples,   num_classes, batch_size,
                                augment=False, shuffle=False)
    test_ds  = _make_tf_dataset(test_samples,  num_classes, batch_size,
                                augment=False, shuffle=False)

    return train_ds, val_ds, test_ds, num_classes


def _save_test_split(test_samples, label_to_idx, path):
    """Serialise the test split to JSON for reproducible Phase 6 evaluation."""
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    records = [
        {'path': p, 'label_idx': lbl, 'identity': idx_to_label[lbl]}
        for p, lbl in test_samples
    ]
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'num_classes': len(label_to_idx), 'samples': records}, f, indent=2)
    print(f'[data_loader] Test split saved -> {path}  ({len(records)} samples)')


def load_test_split(test_split_path: str = TEST_SPLIT_PATH):
    """Load the persisted test split JSON and return (paths, int_labels, num_classes)."""
    with open(test_split_path) as f:
        data = json.load(f)
    paths  = [s['path']      for s in data['samples']]
    labels = [s['label_idx'] for s in data['samples']]
    return paths, labels, data['num_classes']
