# Claude Project Memory — Iris Biometric Authentication System

## Project Overview
**Title:** Improving Image Detection for Biometric (Iris) Authentication Systems
**Type:** University Final Year Project (FYP2)
**Goal:** Build a highly accurate, deep learning-based iris recognition system capable of handling real-world degradations (poor lighting, motion blur, partial occlusions). Implement a CNN feature extractor (MiniIrisXception) trained with ArcFace loss. Quantitatively compare against a traditional Gabor filter baseline.

---

## Dataset: CASIA-IrisV4

### Subsets in Use
| Subset | Purpose |
|---|---|
| CASIA-Iris-Interval | Primary clean iris images |
| CASIA-Iris-Lamp | Illumination variation robustness |
| CASIA-Iris-Thousand | Large-scale identity coverage |
| CASIA-Iris-Syn | Synthetic deformations (replaces Twins subset) |

### Split Ratio
- **70%** Train / **20%** Validation / **10%** Test

### Storage Location
All raw data lives under `data/raw/<subset>/` and is excluded from git.

---

## Preprocessing Pipeline
1. **Noise Reduction** — Gaussian or Median filtering
2. **Segmentation** — Pupil and iris boundary detection (Hough Circle Transform or similar)
3. **Normalization** — Daugman's Rubber-Sheet Model (maps annular iris region to rectangular strip)
4. **Pixel Scaling** — Normalize to [0, 1]

---

## Model Architecture: IrisNet (MiniIrisXception)

- **Backbone:** Depthwise Separable Convolutions (XCeption-style, miniaturized)
- **Pooling:** GlobalAveragePooling2D
- **Output:** L2-Normalized Dense embedding layer (no bias)
- **Embedding dimension:** 512

### Loss Function: ArcFace (Additive Angular Margin Loss)
- Enforces intra-class compactness and inter-class separation
- Margin parameter `m` and scale parameter `s` to be tuned during training

### Baseline for Comparison: Gabor Filter
- Classical texture-based feature extraction
- Compared against IrisNet on the same metrics

---

## Evaluation Metrics
| Metric | Description |
|---|---|
| FAR | False Acceptance Rate — impostor accepted as genuine |
| FRR | False Rejection Rate — genuine rejected as impostor |
| TAR | True Acceptance Rate = 1 - FRR |
| EER | Equal Error Rate — point where FAR == FRR |
| Accuracy | Overall classification accuracy |

---

## Tech Stack
| Component | Version / Note |
|---|---|
| Python | 3.9 |
| TensorFlow | 2.10 (CPU build) |
| tensorflow-directml-plugin | AMD GPU acceleration via DirectML |
| OpenCV | opencv-python (latest compatible) |
| NumPy | Latest compatible |
| Pandas | Latest compatible |
| Matplotlib | Latest compatible |
| Scikit-learn | Latest compatible |
| Jupyter | For exploratory notebooks |
| Package Manager | uv |

> **IMPORTANT:** Do NOT use `tensorflow-gpu`. The target hardware is an MSI Bravo 15 with an AMD Radeon GPU. GPU support is provided exclusively via `tensorflow-directml-plugin`.

---

## Target Hardware
- **Machine:** MSI Bravo 15
- **OS:** Windows 11
- **CPU:** AMD Ryzen 5 5600H
- **GPU:** AMD Radeon (DirectML acceleration)

---

## Project Phase Roadmap
| Phase | Description | Status |
|---|---|---|
| Phase 1 | Environment & Scaffold | Complete |
| Phase 2 | Preprocessing Implementation | Complete |
| Phase 3 | Model Architecture (IrisNet) | Complete |
| Phase 4 | Training with ArcFace | Pending |
| Phase 5 | Baseline (Gabor Filter) | Pending |
| Phase 6 | Evaluation & Comparison | Pending |

---

## Phase 2 Notes — Preprocessing

### HoughCircles Parameters (final tuned values)
| Stage | dp | minDist | param1 | param2 range | minRadius | maxRadius |
|---|---|---|---|---|---|---|
| Pupil | 1.0 | 50 | 100 | 50 → 5 (step -5) | 10 | 80 |
| Iris  | 1.0 | 50 | 100 | 30 → 5 (step -5) | 80 | 200 |

- Pupil candidate selected by **minimum distance to image centre** (handles off-centre pupils).
- Iris candidate selected by **minimum distance to detected pupil centre**.
- Sanity check: `r_iris > r_pupil` AND `centre_distance ≤ max(0.60 × r_iris, 60 px)`.

### Batch Processing Results
| Subset | Processed | Skipped | Detection Rate |
|---|---|---|---|
| CASIA-Iris-Interval | 2 490 | 149 | 94.3% |
| CASIA-Iris-Lamp     | 10 703 | 5 509 | 66.0% |
| CASIA-Iris-Thousand | 10 359 | 9 641 | 51.8% |
| CASIA-Iris-Syn      | 7 074 | 2 926 | 70.7% |
| **Grand Total**     | **30 626** | **18 225** | **62.7%** |

Lamp/Thousand lower rates are expected — ring illuminator creates circular artefacts that confuse HoughCircles; Thousand has heavy occlusion variation. 30 626 tensors of shape `(128, 128, 1)` float32 saved to `data/processed/`.

---

## Phase 3 Notes — Model Architecture

### IrisNet (MiniIrisXception) — Final Architecture
| Component | Detail |
|---|---|
| Total parameters | 423,232 |
| Embedding dimension | 512 |
| Input shape | (128, 128, 1) |
| Output | L2-normalised embedding (unit norm) |

**Layer flow:**
- Initial Block: Conv2D(32) → BN → ReLU → MaxPool → (64, 64, 32)
- Entry Block 1: SepConv residual, 32→64 ch, stride 2 → (32, 32, 64)
- Entry Block 2: SepConv residual, 64→128 ch, stride 2 → (16, 16, 128)
- Middle Flow ×3: Identity residual, 128 ch, Dropout(0.2) → (16, 16, 128)
- Exit Block: SepConv residual, 128→256 ch, stride 2 → (8, 8, 256)
- Head: GlobalAveragePooling2D → Dense(512, no bias) → Lambda(L2-normalise)

**ArcFace Layer (training head):**
- Default margin `m = 0.5`, scale `s = 64.0`
- Trainable class prototype weights W of shape (512, num_classes)
- Input: `(embeddings, labels_onehot)` → output: scaled logits for cross-entropy

**NumPy constraint:** TF 2.10 requires NumPy < 2.0. Pin `numpy<2.0` in any new environment.

---

## Strict Git Workflow Rules
1. **Never work on `main` or `master` directly.** Always create a descriptive branch (e.g., `feature/phase2-preprocessing`).
2. **Frequent, atomic commits.** Each logical unit of work gets its own commit.
3. **Never use `--author` or co-author tags in commits.** Commit normally using the local configured Git user.
4. **Branch naming convention:** `setup/`, `feature/`, `fix/`, `experiment/` prefixes.
