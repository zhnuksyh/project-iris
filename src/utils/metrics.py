"""
src/utils/metrics.py

Evaluation metrics for iris biometric authentication.

Metrics implemented:
  - FAR  (False Acceptance Rate): Fraction of impostor pairs incorrectly accepted.
  - FRR  (False Rejection Rate):  Fraction of genuine pairs incorrectly rejected.
  - TAR  (True Acceptance Rate):  1 - FRR; fraction of genuine pairs correctly accepted.
  - EER  (Equal Error Rate):      Threshold where FAR == FRR; lower is better.
  - Accuracy: Overall correct classification rate.

All metrics operate on cosine similarity scores between embedding pairs and
a configurable decision threshold.

# TODO: Implement in Phase 6
"""
