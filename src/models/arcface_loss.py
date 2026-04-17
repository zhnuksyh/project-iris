"""
src/models/arcface_loss.py

ArcFace (Additive Angular Margin Loss) implementation.

ArcFace improves upon standard softmax loss by adding an angular margin penalty
to the angle between the feature embedding and the class weight vector.
This enforces intra-class compactness and inter-class separability in the
hyperspherical embedding space.

Key hyperparameters:
  - num_classes (int): Number of identity classes in the dataset.
  - embedding_dim (int): Dimensionality of the L2-normalized feature embedding.
  - margin (float): Additive angular margin 'm' (typically 0.5 radians).
  - scale (float): Feature scale factor 's' (typically 64.0).

Margin and scale are stored as tf.Variable so they can be annealed during
training via the MarginScaleAnnealingCallback (see train_arcface.py).

Reference:
  Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
  CVPR 2019. https://arxiv.org/abs/1801.07698
"""

import numpy as np
import tensorflow as tf


class ArcFaceLayer(tf.keras.layers.Layer):
    """ArcFace classification head.

    Holds trainable class prototype weights W of shape (embedding_dim, num_classes).
    During the forward pass it:
      1. L2-normalises both the incoming embeddings and W.
      2. Computes cosine similarity -> angles (theta).
      3. Adds the angular margin m to the true-class angle.
      4. Scales the resulting logits by s.

    Margin (m) and scale (s) are non-trainable tf.Variables that can be
    updated externally (e.g. by a callback) for warmup / annealing.

    The layer is used only during training; for inference the IrisNet base model
    (which outputs L2-normalised embeddings) is used directly with cosine similarity.

    Args:
        inputs: tuple of (embeddings, labels_onehot)
            embeddings   -- (batch, embedding_dim) float32, already L2-normalised
            labels_onehot -- (batch, num_classes) float32 one-hot encoded labels

    Returns:
        Scaled logits of shape (batch, num_classes); feed into
        tf.keras.losses.CategoricalCrossentropy(from_logits=True).
    """

    def __init__(self, num_classes, embedding_dim, margin=0.5, scale=64.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self._initial_margin = float(margin)
        self._initial_scale = float(scale)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='arcface_weights',
            shape=(self.embedding_dim, self.num_classes),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.margin_var = tf.Variable(
            self._initial_margin, trainable=False,
            dtype=tf.float32, name='arcface_margin',
        )
        self.scale_var = tf.Variable(
            self._initial_scale, trainable=False,
            dtype=tf.float32, name='arcface_scale',
        )
        super().build(input_shape)

    def call(self, inputs):
        embeddings, labels_onehot = inputs

        # Normalise embeddings (belt-and-suspenders; IrisNet already does this)
        x = tf.math.l2_normalize(embeddings, axis=1)       # (batch, embedding_dim)
        # Normalise class prototype weights column-wise
        w = tf.math.l2_normalize(self.W, axis=0)           # (embedding_dim, num_classes)

        # Cosine similarity = dot product of unit vectors
        cos_theta = tf.matmul(x, w)                        # (batch, num_classes)
        # Clamp for numerical stability before acos
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Add angular margin to the true-class angle only
        theta = tf.acos(cos_theta)                         # angle in [0, pi]
        cos_theta_m = tf.cos(theta + self.margin_var)      # cos(theta + m)

        # Easy-margin boundary guard (ArcFace paper Eq. 5-7):
        # When theta + m > pi, cos(theta+m) is non-monotonic — fall back to
        # a linear penalty: cos(theta) - sin(pi-m)*m to keep gradient stable.
        threshold = tf.cos(tf.constant(np.pi, dtype=tf.float32) - self.margin_var)
        sin_m = tf.sin(self.margin_var)
        safe_logit = cos_theta - sin_m * self.margin_var
        final_target = tf.where(cos_theta > threshold, cos_theta_m, safe_logit)

        mask = tf.cast(labels_onehot, dtype=tf.bool)
        logits = tf.where(mask, final_target, cos_theta)   # margined for true class only

        return logits * self.scale_var                      # scale before softmax

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
            'margin': self._initial_margin,
            'scale': self._initial_scale,
        })
        return cfg
