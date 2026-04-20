"""
src/models/train_softmax.py

Softmax baseline training script for IrisNet.

This trains IrisNet with a standard Dense + softmax classification head and
categorical cross-entropy loss.  It serves as the closed-set accuracy baseline
before the ArcFace variant is trained.

Saved artefacts
---------------
  models/softmax_best.h5      Best checkpoint (lowest val_loss)
  models/softmax_history.json Training history (loss + accuracy per epoch)

Usage
-----
    python -m src.models.train_softmax
    python -m src.models.train_softmax --epochs 5 --batch_size 16
    python -m src.models.train_softmax --cpu        # force CPU (disables DirectML)
"""

import argparse
import json
import os

import tensorflow as tf

from src.models.architecture import build_irisnet
from src.utils.data_loader import build_datasets

# ── Hyper-parameters (can be overridden via CLI) ──────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 32
LR_INITIAL  = 1e-3
EMBEDDING_DIM = 512

CHECKPOINT_PATH         = 'models/softmax_best.h5'
HISTORY_PATH            = 'models/softmax_history.json'
OPENSET_CHECKPOINT_PATH = 'models/softmax_openset_best.h5'
OPENSET_HISTORY_PATH    = 'models/softmax_openset_history.json'


def build_softmax_model(num_classes: int, embedding_dim: int = EMBEDDING_DIM):
    """Attach a softmax classification head to the IrisNet backbone.

    The backbone output is the L2-normalised 512-D embedding.
    A Dense(num_classes, activation='softmax') head is added on top.

    Args:
        num_classes:   number of identity classes in the training set
        embedding_dim: embedding dimension of the IrisNet backbone

    Returns:
        Compiled tf.keras.Model ready for model.fit()
    """
    backbone = build_irisnet(input_shape=(128, 128, 1), embedding_dim=embedding_dim)

    # Freeze nothing — train end-to-end
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='softmax_head',
    )(backbone.output)

    model = tf.keras.Model(
        inputs=backbone.input,
        outputs=outputs,
        name='IrisNet_softmax',
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INITIAL),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        jit_compile=False,
    )
    return model


def get_callbacks(checkpoint_path: str = CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, cpu: bool = False,
          openset: bool = False):
    if cpu:
        tf.config.set_visible_devices([], 'GPU')
        print('[train_softmax] GPU disabled — running on CPU')

    checkpoint_path = OPENSET_CHECKPOINT_PATH if openset else CHECKPOINT_PATH
    history_path    = OPENSET_HISTORY_PATH    if openset else HISTORY_PATH
    split_mode      = 'identity_disjoint'     if openset else 'stratified'
    min_samples     = 1  # Softmax handles singleton identities fine

    print('=' * 60)
    print(f'IrisNet — Softmax Training  ({"open-set" if openset else "closed-set"})')
    print('=' * 60)

    train_ds, val_ds, _, num_classes = build_datasets(
        batch_size=batch_size,
        split_mode=split_mode,
        min_samples=min_samples,
    )
    print(f'Classes: {num_classes}  |  Batch size: {batch_size}  |  Epochs: {epochs}')

    model = build_softmax_model(num_classes)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(checkpoint_path),
        verbose=1,
    )

    # Persist history for the notebook
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f'History saved -> {history_path}')
    print(f'Best model saved -> {checkpoint_path}')
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,            default=EPOCHS)
    parser.add_argument('--batch_size', type=int,            default=BATCH_SIZE)
    parser.add_argument('--cpu',        action='store_true', default=False)
    parser.add_argument('--openset',    action='store_true', default=False,
                        help='Train with identity-disjoint open-set split')
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, cpu=args.cpu,
          openset=args.openset)
