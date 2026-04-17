"""
src/evaluation/embeddings.py

Load trained models and extract embeddings/codes from the test set.

Three systems:
  - ArcFace backbone  → 512-D L2-normalised embeddings
  - Softmax backbone  → 512-D L2-normalised embeddings (up to l2_norm layer)
  - Gabor IrisCode    → 262,144-bit binary codes
"""

import numpy as np
import tensorflow as tf

from src.models.train_arcface import build_arcface_model
from src.models.train_softmax import build_softmax_model
from src.models.gabor_baseline import extract_iris_code

ARCFACE_MODEL_PATH = 'models/arcface_best.h5'
SOFTMAX_MODEL_PATH = 'models/softmax_best.h5'
INFERENCE_BATCH = 64


def load_test_images(paths: list) -> np.ndarray:
    """Load preprocessed .npy iris images into a single array.

    Args:
        paths: list of file paths to .npy files, each containing a
               (128, 128, 1) float32 array.

    Returns:
        np.ndarray of shape (N, 128, 128, 1), float32.
    """
    images = []
    for p in paths:
        img = np.load(p)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        images.append(img)
    return np.stack(images, axis=0).astype(np.float32)


def extract_arcface_embeddings(images: np.ndarray,
                               num_classes: int = 3960) -> np.ndarray:
    """Extract embeddings using the ArcFace-trained backbone.

    Rebuilds the full ArcFace training model and loads the checkpoint
    weights, then extracts the backbone sub-model for embedding inference.
    This avoids Lambda layer deserialization issues and layer-name mismatches
    that occur with standalone backbone weight files.

    Args:
        images: (N, 128, 128, 1) float32 array.
        num_classes: number of classes the ArcFace model was trained with.

    Returns:
        (N, 512) L2-normalised embeddings.
    """
    training_model, backbone = build_arcface_model(
        num_classes=num_classes, num_train_samples=1000,  # dummy, not used for inference
    )
    training_model.load_weights(ARCFACE_MODEL_PATH)
    print(f'[embeddings] ArcFace model loaded from {ARCFACE_MODEL_PATH}')

    return _batch_inference(backbone, images)


def extract_softmax_embeddings(images: np.ndarray,
                               num_classes: int = 4115) -> np.ndarray:
    """Extract embeddings from the softmax-trained model.

    Rebuilds the softmax model architecture (to avoid Lambda layer
    deserialization issues in Keras 3), loads saved weights, then uses
    the IrisNet backbone sub-model for embedding extraction.

    Args:
        images: (N, 128, 128, 1) float32 array.
        num_classes: number of classes the softmax model was trained with.

    Returns:
        (N, 512) L2-normalised embeddings.
    """
    # Rebuild architecture to avoid Lambda deserialization error
    full_model = build_softmax_model(num_classes)
    full_model.load_weights(SOFTMAX_MODEL_PATH)
    print(f'[embeddings] Softmax weights loaded from {SOFTMAX_MODEL_PATH}')

    # Extract backbone sub-model up to the l2_norm layer
    embedding_layer = full_model.get_layer('l2_norm')
    backbone = tf.keras.Model(
        inputs=full_model.input,
        outputs=embedding_layer.output,
        name='softmax_backbone',
    )

    return _batch_inference(backbone, images)


def extract_gabor_codes(images: np.ndarray) -> np.ndarray:
    """Extract Gabor IrisCodes for all test images.

    Args:
        images: (N, 128, 128, 1) float32 array.

    Returns:
        (N, 262144) bool array.
    """
    n = images.shape[0]
    codes = []
    for i in range(n):
        code = extract_iris_code(images[i])
        codes.append(code)
        if (i + 1) % 500 == 0:
            print(f'[embeddings] Gabor codes: {i + 1}/{n}')
    print(f'[embeddings] Gabor codes: {n}/{n} done')
    return np.stack(codes, axis=0)


def _batch_inference(model: tf.keras.Model, images: np.ndarray) -> np.ndarray:
    """Run inference in chunks to avoid OOM / XLA issues.

    Uses model(x, training=False) instead of model.predict() to avoid
    XLA autotuner crashes on RTX 5090 Blackwell.
    """
    results = []
    n = images.shape[0]
    for start in range(0, n, INFERENCE_BATCH):
        end = min(start + INFERENCE_BATCH, n)
        batch = images[start:end]
        out = model(batch, training=False)
        results.append(out.numpy())
        if (start + INFERENCE_BATCH) % (INFERENCE_BATCH * 10) == 0:
            print(f'[embeddings] Inference: {min(end, n)}/{n}')
    print(f'[embeddings] Inference: {n}/{n} done')
    return np.concatenate(results, axis=0)
