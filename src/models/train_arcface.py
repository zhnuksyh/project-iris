"""
src/models/train_arcface.py

ArcFace training script for IrisNet.

Combines the IrisNet backbone (which outputs L2-normalised 512-D embeddings)
with the ArcFaceLayer classification head.  At inference time only the
backbone is needed — embeddings are compared via cosine similarity.

Training detail
---------------
ArcFace requires the one-hot labels during the forward pass (so it can apply
the angular margin only to the true-class angle).  This means the model has
TWO inputs: [image, label_onehot].  A thin wrapper model is built to fuse
them for Keras's model.fit() API.

Saved artefacts
---------------
  models/arcface_best.h5       Best full-model checkpoint (backbone + head)
  models/arcface_backbone.h5   Backbone-only weights (used for inference / Phase 6)
  models/arcface_history.json  Training history

Usage
-----
    python -m src.models.train_arcface
    python -m src.models.train_arcface --epochs 5 --batch_size 16
    python -m src.models.train_arcface --cpu        # force CPU (disables DirectML)
"""

import argparse
import json
import os

import tensorflow as tf

from src.models.architecture import build_irisnet
from src.models.arcface_loss import ArcFaceLayer
from src.utils.data_loader import build_datasets

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS        = 50
BATCH_SIZE    = 32
LR_INITIAL    = 1e-3
EMBEDDING_DIM = 512
ARCFACE_MARGIN = 0.5
ARCFACE_SCALE  = 64.0

CHECKPOINT_PATH = 'models/arcface_best.h5'
BACKBONE_PATH   = 'models/arcface_backbone.weights.h5'
HISTORY_PATH    = 'models/arcface_history.json'


def build_arcface_model(num_classes: int, embedding_dim: int = EMBEDDING_DIM,
                        margin: float = ARCFACE_MARGIN, scale: float = ARCFACE_SCALE):
    """Build and compile the full ArcFace training model.

    Architecture (training):
        image (128,128,1)  ─┐
                             ├─ IrisNet backbone ─ ArcFaceLayer ─ scaled logits
        label_onehot        ─┘

    Args:
        num_classes:   number of identity classes
        embedding_dim: IrisNet embedding dimension
        margin:        ArcFace angular margin m (default 0.5)
        scale:         ArcFace feature scale s (default 64.0)

    Returns:
        (training_model, backbone)
          training_model: compiled model with two inputs [image, label_onehot]
          backbone:       IrisNet base model (single image → embedding)
    """
    backbone = build_irisnet(input_shape=(128, 128, 1), embedding_dim=embedding_dim)

    # Two inputs for the training wrapper
    img_input   = tf.keras.Input(shape=(128, 128, 1),    name='image')
    label_input = tf.keras.Input(shape=(num_classes,),   name='label_onehot',
                                 dtype=tf.float32)

    embeddings = backbone(img_input, training=True)

    arcface_layer = ArcFaceLayer(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=margin,
        scale=scale,
        name='arcface',
    )
    logits = arcface_layer([embeddings, label_input])

    training_model = tf.keras.Model(
        inputs=[img_input, label_input],
        outputs=logits,
        name='IrisNet_ArcFace',
    )

    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INITIAL),
        # ArcFace outputs raw scaled logits → from_logits=True
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        jit_compile=False,
    )
    return training_model, backbone


def _adapt_dataset_for_arcface(ds: tf.data.Dataset):
    """Re-map a (image, label_onehot) dataset to ([image, label_onehot], label_onehot).

    Keras model.fit() expects (inputs, targets).  Because the ArcFace training
    model has two inputs, the input must be [image, label_onehot] while the
    target is the same label_onehot (for cross-entropy).
    """
    return ds.map(
        lambda x, y: ({'image': x, 'label_onehot': y}, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def get_callbacks():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
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


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, cpu: bool = False):
    if cpu:
        tf.config.set_visible_devices([], 'GPU')
        print('[train_arcface] GPU disabled — running on CPU')

    print('=' * 60)
    print('IrisNet — ArcFace Training')
    print('=' * 60)

    train_ds, val_ds, _, num_classes = build_datasets(batch_size=batch_size)
    print(f'Classes: {num_classes}  |  Batch size: {batch_size}  |  Epochs: {epochs}')
    print(f'ArcFace: margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE}')

    # Wrap datasets so both inputs and targets are provided
    train_ds_af = _adapt_dataset_for_arcface(train_ds)
    val_ds_af   = _adapt_dataset_for_arcface(val_ds)

    training_model, backbone = build_arcface_model(num_classes)
    training_model.summary()

    history = training_model.fit(
        train_ds_af,
        validation_data=val_ds_af,
        epochs=epochs,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Save backbone separately for Phase 6 inference
    backbone.save_weights(BACKBONE_PATH)
    print(f'Backbone weights saved -> {BACKBONE_PATH}')

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f'History saved -> {HISTORY_PATH}')
    print(f'Best full model saved -> {CHECKPOINT_PATH}')
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,            default=EPOCHS)
    parser.add_argument('--batch_size', type=int,            default=BATCH_SIZE)
    parser.add_argument('--cpu',        action='store_true', default=False)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, cpu=args.cpu)
