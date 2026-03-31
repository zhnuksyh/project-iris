"""
src/preprocessing/segmentation.py

Iris segmentation module for the IrisNet pipeline.

Responsibilities:
  - Detect the pupil boundary (inner circle) using Hough Circle Transform or similar.
  - Detect the iris boundary (outer circle).
  - Apply Daugman's Rubber-Sheet Model to normalize the annular iris region
    into a fixed-size rectangular strip.
  - Apply noise reduction (Gaussian or Median filtering) prior to segmentation.

Input:  Raw grayscale iris image (numpy array or file path).
Output: Normalized iris strip of fixed dimensions (height x width), pixel values in [0, 1].

# TODO: Implement in Phase 2
"""
