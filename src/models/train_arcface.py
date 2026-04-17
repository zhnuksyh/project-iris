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

The training uses:
  - SGD with momentum (0.9) and weight decay (5e-4) instead of Adam, to
    prevent the backbone from collapsing while the W-matrix absorbs all
    discriminative capacity.
  - Margin/scale annealing: warmup with m=0 s=16, then linear ramp to
    m=0.5 s=64 over RAMPUP_EPOCHS.
  - Deterministic step LR decay at epochs 40/60/80 (standard ArcFace schedule).

Saved artefacts
---------------
  models/arcface_best.h5             Best full-model checkpoint (backbone + head)
  models/arcface_backbone.weights.h5 Backbone-only weights (used for inference)
  models/arcface_history.json        Training history

Usage
-----
    python -m src.models.train_arcface
    python -m src.models.train_arcface --epochs 100 --batch_size 64
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
EPOCHS         = 50
BATCH_SIZE     = 64
LR_INITIAL     = 0.01      # SGD base LR
EMBEDDING_DIM  = 512
ARCFACE_MARGIN = 0.5       # target margin (annealed from 0.0)
ARCFACE_SCALE  = 64.0      # target scale  (annealed from 16.0)
WARMUP_EPOCHS  = 5         # softmax-only warmup (m=0, s=16)
RAMPUP_EPOCHS  = 15        # linear ramp m: 0→0.5, s: 16→64
MIN_SAMPLES    = 2         # exclude single-sample classes

CHECKPOINT_PATH = 'models/arcface_best.h5'
BACKBONE_PATH   = 'models/arcface_backbone.weights.h5'
HISTORY_PATH    = 'models/arcface_history.json'


# ── Margin / Scale Annealing ─────────────────────────────────────────────────

class MarginScaleAnnealingCallback(tf.keras.callbacks.Callback):
    """Anneal ArcFace margin and scale during training.

    Schedule:
      - Warmup  (epoch 0 .. warmup-1):  m = 0.0,           s = initial_scale
      - Ramp-up (warmup .. warmup+ramp): m linearly → target_m, s linearly → target_s
      - Full    (after ramp):            m = target_m,      s = target_s
    """

    def __init__(self, warmup_epochs, rampup_epochs,
                 target_margin, target_scale, initial_scale=16.0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.target_margin = target_margin
        self.target_scale = target_scale
        self.initial_scale = initial_scale

    def on_epoch_begin(self, epoch, logs=None):
        arcface = self.model.get_layer('arcface')
        if epoch < self.warmup_epochs:
            m, s = 0.0, self.initial_scale
        elif epoch < self.warmup_epochs + self.rampup_epochs:
            progress = (epoch - self.warmup_epochs) / self.rampup_epochs
            m = self.target_margin * progress
            s = self.initial_scale + (self.target_scale - self.initial_scale) * progress
        else:
            m, s = self.target_margin, self.target_scale

        arcface.margin_var.assign(m)
        arcface.scale_var.assign(s)
        if epoch < self.warmup_epochs + self.rampup_epochs + 1 or epoch % 10 == 0:
            print(f'  [anneal] epoch {epoch}: margin={m:.3f}, scale={s:.1f}')


def build_arcface_model(num_classes: int, num_train_samples: int,
                        embedding_dim: int = EMBEDDING_DIM):
    """Build and compile the full ArcFace training model.

    Architecture (training):
        image (128,128,1)  ─┐
                             ├─ IrisNet backbone ─ ArcFaceLayer ─ scaled logits
        label_onehot        ─┘

    Args:
        num_classes:      number of identity classes
        num_train_samples: number of training samples (for LR schedule)
        embedding_dim:    IrisNet embedding dimension

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

    # Start with m=0, s=16 — annealed by MarginScaleAnnealingCallback
    arcface_layer = ArcFaceLayer(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=0.0,
        scale=16.0,
        name='arcface',
    )
    logits = arcface_layer([embeddings, label_input])

    training_model = tf.keras.Model(
        inputs=[img_input, label_input],
        outputs=logits,
        name='IrisNet_ArcFace',
    )

    # Step LR decay at epochs 25/35/45 (scaled for 50-epoch schedule)
    steps_per_epoch = num_train_samples // BATCH_SIZE + 1
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[
            steps_per_epoch * 25,
            steps_per_epoch * 35,
            steps_per_epoch * 45,
        ],
        values=[
            LR_INITIAL,
            LR_INITIAL * 0.1,
            LR_INITIAL * 0.01,
            LR_INITIAL * 0.001,
        ],
    )

    training_model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=0.9,
            weight_decay=5e-4,
        ),
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
        MarginScaleAnnealingCallback(
            warmup_epochs=WARMUP_EPOCHS,
            rampup_epochs=RAMPUP_EPOCHS,
            target_margin=ARCFACE_MARGIN,
            target_scale=ARCFACE_SCALE,
            initial_scale=16.0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # No EarlyStopping — the margin/scale ramp-up causes val_accuracy
        # to drop temporarily, which misleads patience-based stopping.
        # Instead, rely on the full 100-epoch schedule with LR step decay
        # at epochs 40/60/80 (standard ArcFace protocol).
    ]


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, cpu: bool = False):
    if cpu:
        tf.config.set_visible_devices([], 'GPU')
        print('[train_arcface] GPU disabled — running on CPU')

    print('=' * 60)
    print('IrisNet — ArcFace Training (v2: warmup + SGD)')
    print('=' * 60)

    train_ds, val_ds, _, num_classes = build_datasets(
        batch_size=batch_size, min_samples=MIN_SAMPLES,
    )
    # Count training samples for LR schedule boundaries
    num_train_samples = sum(1 for _ in train_ds.unbatch())
    print(f'Classes: {num_classes}  |  Batch size: {batch_size}  |  Epochs: {epochs}')
    print(f'Train samples: {num_train_samples}  |  Min samples/class: {MIN_SAMPLES}')
    print(f'ArcFace target: margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE}')
    print(f'Warmup: {WARMUP_EPOCHS} epochs (m=0, s=16)  |  Ramp-up: {RAMPUP_EPOCHS} epochs')
    print(f'Optimizer: SGD(lr={LR_INITIAL}, momentum=0.9, wd=5e-4)')

    # Wrap datasets so both inputs and targets are provided
    train_ds_af = _adapt_dataset_for_arcface(train_ds)
    val_ds_af   = _adapt_dataset_for_arcface(val_ds)

    training_model, backbone = build_arcface_model(
        num_classes, num_train_samples=num_train_samples,
    )
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
