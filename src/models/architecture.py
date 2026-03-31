"""
src/models/architecture.py

IrisNet (MiniIrisXception) model architecture.

Architecture summary:
  - Input: Normalized iris strip (fixed H x W x 1 grayscale).
  - Backbone: Stack of Depthwise Separable Convolution blocks (XCeption-inspired,
    miniaturized for efficiency on the target hardware).
  - Pooling: GlobalAveragePooling2D to collapse spatial dimensions.
  - Embedding: Dense layer with L2 normalization and no bias term.
    Output is a unit-norm embedding vector in R^d (d = embedding_dim).

The model is trained end-to-end with ArcFace loss (see arcface_loss.py).
At inference time, cosine similarity between embeddings is used for verification.

# TODO: Implement in Phase 3
"""
