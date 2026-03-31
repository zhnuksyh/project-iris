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

Reference:
  Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
  CVPR 2019. https://arxiv.org/abs/1801.07698
"""

import tensorflow as tf


class ArcFaceLayer(tf.keras.layers.Layer):
    """ArcFace classification head.

    Holds trainable class prototype weights W of shape (embedding_dim, num_classes).
    During the forward pass it:
      1. L2-normalises both the incoming embeddings and W.
      2. Computes cosine similarity -> angles (theta).
      3. Adds the angular margin m to the true-class angle.
      4. Scales the resulting logits by s.

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
        self.margin = margin
        self.scale = scale

    def build(self, input_shape):
        self.W = self.add_weight(
            name='arcface_weights',
            shape=(self.embedding_dim, self.num_classes),
            initializer='glorot_uniform',
            trainable=True,
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
        margined = tf.cos(theta + self.margin)             # cos(theta + m) for all classes

        mask = tf.cast(labels_onehot, dtype=tf.bool)
        logits = tf.where(mask, margined, cos_theta)       # swap in margined for true class

        return logits * self.scale                         # scale before softmax

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
            'margin': self.margin,
            'scale': self.scale,
        })
        return cfg
