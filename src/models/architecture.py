"""
src/models/architecture.py

IrisNet (MiniIrisXception) model architecture.

Architecture summary:
  - Input: Preprocessed iris image (128 x 128 x 1 grayscale, values in [0, 1]).
  - Initial Block: Conv2D -> BN -> ReLU -> MaxPool to downsample to 64x64.
  - Entry Flow: 2 residual blocks with SeparableConv2D, halving spatial dims each time.
  - Middle Flow: 3 identical residual blocks at constant spatial resolution (16x16).
  - Exit Flow: 1 residual block widening channels to 256, halving to 8x8.
  - Head: GlobalAveragePooling2D -> Dense(embedding_dim, no bias) -> L2 normalisation.

The model is trained end-to-end with ArcFace loss (see arcface_loss.py).
At inference time, cosine similarity between embeddings is used for verification.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def _entry_block(x, filters, name):
    """Residual block that doubles channels and halves spatial resolution."""
    # Shortcut path: 1x1 conv to match channel count, stride 2 to match spatial
    shortcut = layers.Conv2D(filters, 1, strides=2, padding='same', use_bias=False,
                             name=f'{name}_shortcut_conv')(x)
    shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    # Main path
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False,
                                name=f'{name}_sep1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False,
                                name=f'{name}_sep2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name=f'{name}_pool')(x)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu_out')(x)
    return x


def _middle_block(x, filters, dropout_rate, name):
    """Residual block with identity shortcut — no change in dims or channels."""
    shortcut = x

    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False,
                                name=f'{name}_sep1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False,
                                name=f'{name}_sep2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.Dropout(dropout_rate, name=f'{name}_drop')(x)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu_out')(x)
    return x


def build_irisnet(input_shape=(128, 128, 1), embedding_dim=512):
    """Build the IrisNet (MiniIrisXception) feature extractor.

    Args:
        input_shape: Shape of one input tensor — (height, width, channels).
                     Default (128, 128, 1) matches the preprocessed .npy tensors.
        embedding_dim: Dimension of the output L2-normalised embedding vector.

    Returns:
        A tf.keras.Model that maps (batch, 128, 128, 1) -> (batch, embedding_dim).
        All output vectors are guaranteed to have L2 norm == 1.0.
    """
    inputs = tf.keras.Input(shape=input_shape, name='iris_input')

    # ── Initial Block ─────────────────────────────────────────────────────────
    # (128, 128, 1) -> (64, 64, 32)
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='init_conv')(inputs)
    x = layers.BatchNormalization(name='init_bn')(x)
    x = layers.ReLU(name='init_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name='init_pool')(x)

    # ── Entry Flow ────────────────────────────────────────────────────────────
    # Block 1: (64, 64, 32) -> (32, 32, 64)
    x = _entry_block(x, filters=64, name='entry1')
    # Block 2: (32, 32, 64) -> (16, 16, 128)
    x = _entry_block(x, filters=128, name='entry2')

    # ── Middle Flow ───────────────────────────────────────────────────────────
    # 3 identical residual blocks, (16, 16, 128) throughout
    for i in range(1, 4):
        x = _middle_block(x, filters=128, dropout_rate=0.2, name=f'middle{i}')

    # ── Exit Flow ─────────────────────────────────────────────────────────────
    # (16, 16, 128) -> (8, 8, 256)
    x = _entry_block(x, filters=256, name='exit1')

    # ── Head ─────────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name='gap')(x)          # (256,)
    x = layers.Dense(embedding_dim, use_bias=False,
                     name='embedding')(x)                      # (embedding_dim,)
    embeddings = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name='l2_norm'
    )(x)                                                        # unit-norm embedding

    return Model(inputs=inputs, outputs=embeddings, name='IrisNet')
