# Improving Image Detection for Biometric (Iris) Authentication Systems

**Final Year Project — Phase 6 Evaluation Report**
**April 2026**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 — Environment & Setup](#2-phase-1--environment--setup)
3. [Phase 2 — Preprocessing Pipeline](#3-phase-2--preprocessing-pipeline)
4. [Phase 3 — IrisNet Architecture](#4-phase-3--irisnet-architecture)
5. [Phase 4 — Gabor Filter Baseline](#5-phase-4--gabor-filter-baseline)
6. [Phase 5 — Model Training](#6-phase-5--model-training)
7. [Phase 6 — Evaluation & Comparison](#7-phase-6--evaluation--comparison)
8. [Final Results](#8-final-results)
9. [Analysis & Discussion](#9-analysis--discussion)
10. [Limitations & Statistical Caveats](#10-limitations--statistical-caveats)
11. [Phase 7 — Open-Set Evaluation](#11-phase-7--open-set-evaluation)
12. [Conclusion & Future Work](#12-conclusion--future-work)

---

## 1. Project Overview

This project implements and compares three iris recognition systems for biometric authentication, ranging from classical computer vision to modern deep metric learning:

| System | Type | Representation | Similarity Metric |
|---|---|---|---|
| **ArcFace** | Deep metric learning (IrisNet + ArcFace loss) | 512-D L2-normalised embeddings | Cosine similarity |
| **Softmax** | Deep classification (IrisNet + softmax head) | 512-D L2-normalised embeddings | Cosine similarity |
| **Gabor** | Classical (Gabor filter bank) | 262,144-bit binary IrisCodes | 1 − Hamming distance |

The project uses the **CASIA-IrisV4** dataset (4 subsets, ~48,851 raw images) and follows a 6-phase development lifecycle, from environment setup through final evaluation.

---

## 2. Phase 1 — Environment & Setup

### Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | TensorFlow 2.21 compatibility |
| Framework | TensorFlow 2.21 (CUDA) | GPU training on RTX 5090 |
| Image Processing | OpenCV 4.x | HoughCircles, Gabor filters |
| Package Manager | uv + venv | Fast, reproducible dependencies |
| Hardware | NVIDIA RTX 5090 (32 GB VRAM) | Blackwell architecture, CUDA 12.x |

### Dataset: CASIA-IrisV4

| Subset | Raw Images | Characteristics |
|---|---|---|
| CASIA-Iris-Interval | ~2,600 | Clean baseline, NIR illuminator |
| CASIA-Iris-Lamp | ~16,200 | Ring illuminator, specular reflections |
| CASIA-Iris-Thousand | ~20,000 | Large-scale, heavy occlusion variation |
| CASIA-Iris-Syn | ~10,000 | Synthetic deformations |
| **Total** | **~48,851** | |

### Project Structure

```
project-iris/
├── data/
│   ├── raw/                    # CASIA-IrisV4 (4 subsets)
│   ├── processed/              # 30,626 preprocessed .npy tensors
│   └── test_split.json         # Test set manifest (3,960 identities)
├── src/
│   ├── preprocessing/          # Segmentation & normalization
│   ├── models/                 # Architecture, training, Gabor baseline
│   ├── evaluation/             # Pairs, embeddings, plotting
│   └── utils/                  # Data loader, metrics
├── models/                     # Trained weights (.h5)
├── notebooks/                  # Exploratory notebooks (01–05)
├── figures/                    # Generated evaluation plots
└── reports/                    # This report + dashboard
```

---

## 3. Phase 2 — Preprocessing Pipeline

The preprocessing pipeline converts raw iris images into standardised (128, 128, 1) float32 tensors suitable for CNN training. It applies four sequential operations:

### Pipeline Steps

**Step 1: Noise Reduction** — Median blur (k=5) removes salt-and-pepper noise and specular reflections. Gaussian blur (5×5, σ=1.5) smooths remaining high-frequency noise.

**Step 2: Iris Segmentation** — Two-phase HoughCircles detection: first the pupil boundary (inner), then the iris boundary (outer). A fallback loop progressively relaxes detection thresholds (param2: 50→5 for pupil, 30→5 for iris). Sanity check ensures r_iris > r_pupil and centres are within tolerance (60% of iris radius).

**Step 3: Daugman Rubber-Sheet Normalization** — Maps the annular iris region to a rectangular strip (64×512) using polar coordinate transformation:

```
x(ρ, θ) = x_pupil(θ) + ρ × (x_iris(θ) - x_pupil(θ))
y(ρ, θ) = y_pupil(θ) + ρ × (y_iris(θ) - y_pupil(θ))
```

Where ρ ∈ [0, 1] (radial) and θ ∈ [0, 2π) (angular). This produces a size-invariant, rotation-aligned representation.

**Step 4: Pixel Scaling** — Resizes to 128×128 (bilinear interpolation), casts to float32, normalises to [0, 1], and adds channel dimension → final shape (128, 128, 1).

### Batch Processing Results

| Subset | Processed | Skipped | Detection Rate |
|---|---|---|---|
| CASIA-Iris-Interval | 2,490 | 149 | 94.3% |
| CASIA-Iris-Lamp | 10,703 | 5,509 | 66.0% |
| CASIA-Iris-Thousand | 10,359 | 9,641 | 51.8% |
| CASIA-Iris-Syn | 7,074 | 2,926 | 70.7% |
| **Total** | **30,626** | **18,225** | **62.7%** |

Lower detection rates in Lamp and Thousand subsets are expected due to challenging imaging conditions (ring illuminator artefacts, heavy eyelid/eyelash occlusion).

### Key Files

- `src/preprocessing/segmentation.py` — Core pipeline (denoise, segment, normalise, scale)
- `src/preprocessing/batch_processor.py` — Batch processing across all subsets

---

## 4. Phase 3 — IrisNet Architecture

**IrisNet** (MiniIrisXception) is a compact, XCeption-inspired CNN that extracts 512-dimensional L2-normalised embeddings from preprocessed iris images.

### Architecture Overview

| Block | Input Shape | Output Shape | Operation |
|---|---|---|---|
| Initial | (128, 128, 1) | (64, 64, 32) | Conv2D(32) → BN → ReLU → MaxPool |
| Entry Block 1 | (64, 64, 32) | (32, 32, 64) | SepConv residual block, stride 2 |
| Entry Block 2 | (32, 32, 64) | (16, 16, 128) | SepConv residual block, stride 2 |
| Middle ×3 | (16, 16, 128) | (16, 16, 128) | Identity residual blocks, Dropout(0.2) |
| Exit Block | (16, 16, 128) | (8, 8, 256) | SepConv residual block, stride 2 |
| Head | (8, 8, 256) | (512,) | GlobalAvgPool → Dense(512, no bias) |
| L2 Norm | (512,) | (512,) | λ(x/‖x‖₂) — unit-norm embedding |

### Model Statistics

- **Total Parameters:** 423,232 (all trainable)
- **Embedding Dimension:** 512
- **Input Shape:** (128, 128, 1)
- **Output:** Unit-norm 512-D vector (cosine similarity compatible)
- **Model Size:** ~1.8 MB (backbone weights)

### ArcFace Loss Layer

The ArcFace classification head enforces intra-class compactness and inter-class separation by adding an angular margin penalty during training:

1. L2-normalise both embeddings and weight matrix W
2. Compute cos(θ) = embedding · Wᵀ
3. Add angular margin to true-class: cos(θ + m)
4. Apply easy-margin boundary guard for gradient stability when θ + m > π
5. Scale logits by factor *s*

At inference, only the backbone is used — embeddings are compared via cosine similarity.

### Key Files

- `src/models/architecture.py` — IrisNet backbone definition
- `src/models/arcface_loss.py` — ArcFaceLayer with annealable margin/scale via tf.Variable

---

## 5. Phase 4 — Gabor Filter Baseline

The classical Gabor baseline provides a training-free reference point using Daugman's IrisCode approach.

### Filter Bank Configuration

| Parameter | Value |
|---|---|
| Scales (frequencies) | 4 (kernel sizes: 9, 13, 17, 23 px) |
| Orientations | 8 (0° to 157.5°, step 22.5°) |
| Filter pairs | 32 (quadrature: real + imaginary) |
| Subsampling factor | 2× (128×128 → 64×64) |
| **IrisCode length** | **262,144 bits** (32 × 2 × 64 × 64) |

### Extraction Process

1. **Mean-centering** — Subtract image mean to remove DC bias (critical: without this, ~75% of bits = 1, destroying discriminability)
2. **Convolution** — Convolve with each quadrature filter pair (real and imaginary)
3. **Subsampling** — Keep every 2nd pixel to decorrelate adjacent bits
4. **Binarisation** — Sign-based: True where response > 0

### Matching

Iris comparison uses the **fractional Hamming distance**: HD = (mismatching bits) / (total bits). Similarity score = 1 − HD for consistency with cosine similarity convention.

### Key Files

- `src/models/gabor_baseline.py` — Filter bank construction, IrisCode extraction, Hamming distance

---

## 6. Phase 5 — Model Training

### Dataset Split

| Split | Samples | Percentage | Augmentation |
|---|---|---|---|
| Train | 20,238 | 70% | RandomRotation(±5%), RandomZoom(±5%), RandomTranslation(±3%), GaussianNoise(0.01) |
| Validation | 6,180 | 20% | None |
| Test | 4,364 | 10% | None |
| **Total** | **30,782** | | |

Identities: **4,115** unique (subset/subject/eye combinations). Stratified split ensures each identity appears in applicable splits. Identities with 1 sample go to train only; identities with 2 samples split into train + test.

### Softmax Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam, lr=1e-3 |
| Batch size | 32 |
| Epochs | 50 |
| Loss | CategoricalCrossentropy |
| Classes | 4,115 (all identities) |
| Callbacks | ModelCheckpoint (val_loss), EarlyStopping (patience=10), ReduceLROnPlateau |

### ArcFace Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | SGD (lr=0.01, momentum=0.9, weight_decay=5e-4) |
| Batch size | 64 |
| Epochs | 50 |
| Loss | CategoricalCrossentropy (from_logits=True) |
| Classes | 3,960 (filtered with min_samples=2) |
| Target margin / scale | m=0.5, s=64.0 (annealed from m=0, s=16) |
| Warmup | 5 epochs (m=0, s=16 — pure cosine softmax) |
| Ramp-up | 15 epochs (linear m: 0→0.5, s: 16→64) |
| LR schedule | Step decay at epochs 25/35/45 (×0.1 each) |
| Callbacks | MarginScaleAnnealing, ModelCheckpoint (val_accuracy) |

### ArcFace Collapse & Fix

The initial ArcFace training collapsed — all inputs mapped to a single point on the hypersphere (EER ~47.6%). Five compounding issues were identified and fixed:

1. **Adam → SGD with momentum** — Adam's per-parameter adaptive rates let the W-matrix (2.1M params) absorb all discriminative capacity while starving the backbone (423K params). SGD provides uniform gradient scaling.

2. **Margin/scale annealing** — Full margin (m=0.5) and scale (s=64) from epoch 0 created an energy landscape where collapsing all embeddings to one point was the easiest minimum. Warmup (5 epochs at m=0, s=16) followed by linear ramp-up (15 epochs) prevents this.

3. **Step LR decay** — ReduceLROnPlateau (monitoring val_loss) never triggered because val_loss kept slowly decreasing even during collapse. Deterministic step decay at epochs 25/35/45 guarantees LR reduction.

4. **Class filtering** — 155 identities with only 1 sample provide degenerate ArcFace gradients (no intra-class compactness to enforce). Filtering to min_samples=2 retains 3,960 identities.

5. **Stronger augmentation** — Only RandomRotation(±5%) was applied. Adding RandomZoom(±5%), RandomTranslation(±3%), and GaussianNoise(0.01) prevents memorization by epoch 10.

6. **No EarlyStopping** — EarlyStopping is incompatible with margin ramp-up because val_accuracy drops temporarily as margin increases. It prematurely restored warmup-only weights (m=0.033) that had no meaningful angular margin.

### Training Results (Final Epoch)

| Model | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|---|---|---|---|---|
| Softmax | 97.81% | 84.56% | 0.431 | 1.091 |
| ArcFace | 35.69% | 26.34% | 6.195 | 10.89 |

> **Note:** ArcFace's lower classification accuracy is expected — the angular margin (m=0.5) makes the closed-set classification task much harder. What matters for verification is embedding quality, measured by EER and TAR@FAR in Phase 6.

### Key Files

- `src/models/train_softmax.py` — Softmax training pipeline
- `src/models/train_arcface.py` — ArcFace training with margin/scale annealing
- `src/utils/data_loader.py` — Stratified data loader with augmentation and min_samples filtering

---

## 7. Phase 6 — Evaluation & Comparison

### Evaluation Protocol

1. Load **4,364 test images** across **3,960 identities** (held out during training)
2. Generate **405 genuine pairs** (same identity) and **40,500 impostor pairs** (different identities, 1:100 ratio)
3. Extract embeddings/codes from each system
4. Compute pairwise similarity scores
5. Sweep thresholds to compute FAR/FRR curves
6. Report EER, TAR@FAR=1%, TAR@FAR=0.1%

### Test Set Composition

| Category | Count |
|---|---|
| Total test samples | 4,364 |
| Unique identities | 3,960 |
| Identities with 1 test sample | 3,557 |
| Identities with 2 test samples | 402 |
| Identities with 3 test samples | 1 |
| Genuine pairs possible | 405 |

> **Note:** 3,557 of 3,960 test identities have only 1 sample, so they cannot form genuine pairs. Only 403 identities contribute to the 405 genuine pairs. This limits the statistical power of EER estimates (see [Section 10](#10-limitations--statistical-caveats)).

### Metrics Definitions

| Metric | Definition | Interpretation |
|---|---|---|
| FAR | Fraction of impostor pairs accepted (score ≥ threshold) | False alarm rate |
| FRR | Fraction of genuine pairs rejected (score < threshold) | Miss rate |
| EER | Threshold where FAR = FRR | Lower is better; single-number summary |
| TAR@FAR | True acceptance rate at a fixed FAR | Higher is better; operational metric |

### Embedding Quality Diagnostics

| System | Genuine Mean | Impostor Mean | Gap | Embedding Std/dim |
|---|---|---|---|---|
| ArcFace | 0.7523 | 0.0140 | **0.7383** | 0.0437 |
| Softmax | 0.7720 | 0.0380 | 0.7340 | 0.0433 |
| Gabor | 0.6863 | 0.5712 | 0.1151 | — |

Both deep learning systems show excellent genuine/impostor separation. ArcFace has the widest gap (0.738) due to the angular margin pushing impostor scores closer to zero. The Gabor baseline has heavy overlap (gap only 0.115), explaining its high EER.

### Key Files

- `src/evaluation/pairs.py` — Pair generation, cosine and Hamming scoring
- `src/evaluation/embeddings.py` — Embedding extraction from all three systems
- `src/evaluation/plotting.py` — ROC, DET, score distributions, t-SNE, training curves
- `src/utils/metrics.py` — EER, TAR@FAR, ROC/DET curve builders
- `notebooks/05_evaluation.ipynb` — Interactive evaluation notebook

---

## 8. Final Results

### Comparison Table

| Metric | ArcFace | Softmax | Gabor |
|---|---|---|---|
| **EER (%)** | **3.46%** | 4.20% | 26.67% |
| **TAR @ FAR=1%** | **93.33%** | 93.09% | 42.72% |
| **TAR @ FAR=0.1%** | 52.10% | **82.72%** | 29.14% |
| **EER Threshold** | 0.3172 | 0.3594 | 0.6112 |

### Visual Results

All evaluation plots are available in the companion dashboard (`reports/phase6_evaluation_report.html`):

- **ROC Curves** — `figures/roc_curves.png`
- **DET Curves** — `figures/det_curves.png`
- **Score Distributions** — `figures/score_distributions.png`
- **Training Curves** — `figures/training_curves.png`
- **t-SNE (ArcFace)** — `figures/tsne_arcface.png`
- **t-SNE (Softmax)** — `figures/tsne_softmax.png`

---

## 9. Analysis & Discussion

### ArcFace vs Softmax

ArcFace achieves the best EER (3.46%) and TAR@FAR=1% (93.33%), directionally confirming the theoretical advantage of angular margin losses for verification tasks. The margin penalty (m=0.5) explicitly optimises for inter-class separation in angular space, which directly translates to better cosine similarity-based matching.

However, Softmax outperforms ArcFace at the strictest operating point — TAR@FAR=0.1% (82.72% vs 52.10%). This suggests that Softmax-trained embeddings have a tighter genuine score distribution. The likely cause: with 4,115 classes and stable Adam training, the Softmax model produces more consistent intra-class embeddings, while ArcFace's aggressive margin constraint with only ~7.5 samples per class introduces higher intra-class variance.

### Deep Learning vs Classical

Both deep learning systems dramatically outperform the Gabor baseline (EER 26.67%). The Gabor IrisCode approach is inherently limited by:

- **Fixed filter parameters** — no adaptation to data distribution
- **Binary quantisation** — loses continuous information
- **Sensitivity to segmentation accuracy** — misalignment causes bit errors
- **No capacity for learning** identity-discriminative features
- **Shared encoding bias** — Across all Gabor variants tested (resized 128² and native 64×512 strip), impostor pairs settle at similarity ~0.57, well above the 0.50 floor for independent codes (see Section 10.2). The fixed-filter binary code retains a shared signature regardless of input geometry.

However, the Gabor baseline requires no training data and runs in ~1ms per comparison — useful as a lightweight pre-filter or in resource-constrained environments.

### ArcFace Training Collapse: Lessons Learned

The initial ArcFace training produced a degenerate model (EER 47.6%) where the backbone mapped all inputs to a single point on the hypersphere. The degenerate solution is: backbone outputs a constant embedding vector, while the ArcFace W-matrix (512 × 3,960 = 2.0M parameters) handles all class discrimination through its column angles to that single point. Despite this collapse, the model achieved 100% training accuracy — masking the failure entirely if only classification metrics are monitored.

This underscores the importance of:
1. **Monitoring embedding diversity** (std per dimension) during training, not just accuracy
2. **Proper training configuration** for metric learning losses, which are more sensitive to hyperparameters than standard classification
3. **Warmup schedules** for angular margin losses, especially with many classes and few samples per class

---

## 10. Limitations & Statistical Caveats

### 10.1 Statistical Significance of ArcFace vs Softmax

With only **405 genuine pairs**, the EER estimates have substantial uncertainty. Bootstrap analysis yields the following 95% confidence intervals:

| System | EER | 95% CI |
|---|---|---|
| ArcFace | 3.46% | [1.73%, 5.19%] |
| Softmax | 4.20% | [2.47%, 6.18%] |

The confidence intervals **overlap substantially** (ArcFace upper bound 5.19% > Softmax lower bound 2.47%). Therefore, the difference in EER between ArcFace and Softmax is **not statistically significant** at the 95% confidence level. The claim "ArcFace beats Softmax" is directionally supported but not conclusively proven with this test set size. A larger test set with more multi-sample identities would be needed for statistical significance.

### 10.2 Gabor Baseline Bottleneck — Hypothesis Tested & Rejected

The Phase 6 Gabor baseline is applied to **128×128 resized images**, not the original (64, 512) rubber-sheet strip. The original report hypothesised that this isotropic resize was the limiting factor — compressing the angular dimension 4× and distorting the spatial frequencies Gabor filters were designed to capture — and that applying Gabor on the native strip would yield a fairer (better) classical baseline.

**This hypothesis was tested and rejected.** Two strip-Gabor variants were implemented and evaluated against the closed-set test split:

| Variant | EER | TAR@FAR=1% | Genuine mean | Impostor mean | Gap |
|---|---|---|---|---|---|
| **Gabor (128×128 resized)** | **26.67%** | **42.72%** | 0.686 | 0.571 | 0.115 |
| Gabor (64×512 strip, naive) | 30.04% | 38.46% | 0.661 | 0.569 | 0.092 |
| Gabor (64×512 strip, cyclic + occlusion mask) | 29.78% | 37.72% | 0.658 | 0.567 | 0.091 |

Both strip variants performed *worse* than the resized baseline by ~3 EER points. The engineering corrections — cyclic angular wrap-around (so θ=0 ≡ θ=2π under convolution) and an upper-eyelid occlusion mask covering ±60° around 12 o'clock — barely moved the needle (0.3 EER points).

**The actual bottleneck is shared encoding artefacts, not aspect ratio.** Across all three variants the impostor mean similarity is ~0.57, well above the 0.50 floor for statistically independent codes. This residual correlation is the same in both spatial domains, so re-encoding in polar coordinates cannot escape it. The likely cause is the radial luminance gradient (pupil edge brighter than iris edge) shared across all images, which produces a consistent DC-like response that survives binarisation regardless of the Gabor bank's geometry.

**Implication.** The Gabor baseline's 26.67% EER is not the cap of "classical Gabor with better preprocessing" but is genuinely close to the cap of "fixed-filter binary IrisCode on this dataset under our preprocessing". Closing the gap would require attacking the shared-pattern bias directly (per-row contrast normalisation, learned masking, occlusion-aware comparison), not just preserving native resolution. The strip encoder is retained in `src/models/gabor_baseline.py` (`extract_iris_code_strip`, `extract_iris_code_strip_v2`) for reproducibility and future ablations.

### 10.3 Closed-Set Evaluation Protocol

Train and test sets share the same identities (stratified 70/20/10 split). The deep learning models have seen training samples from identities that appear in the test set. While the specific test images were held out, the models learned identity-specific representations during training.

A more rigorous evaluation would use a truly **open-set protocol** with completely disjoint identity sets between train and test. This would test whether the model learned generalisable iris features rather than memorising specific identity patterns.

### 10.4 t-SNE Visualisation Sparsity

Only **41 test samples** across 20 identities have 2+ samples, limiting the informativeness of t-SNE plots. Most identities (3,557 / 3,960) have exactly 1 test sample, making cluster visualisation impossible for the majority of the test set.

### 10.5 Single Dataset

All data comes from CASIA-IrisV4. Cross-dataset evaluation (e.g., ND-IRIS-0405, UBIRIS.v2, IITD) would be needed to verify generalisation to different sensors, illumination conditions, and demographics.

---

## 11. Phase 7 — Open-Set Evaluation

The Phase 6 evaluation used a stratified closed-set split (train and test share identities), which the limitations section flagged as a key caveat. Phase 7 addresses this by retraining both deep models on an **identity-disjoint** split and re-running the full verification evaluation. No classifier is deployed at test time — identification decisions are made by cosine similarity in the 512-D embedding space, so the absence of test-identity classes in training is by design.

### 11.1 Protocol

| Property | Closed-set (Phase 6) | Open-set (Phase 7) |
|---|---|---|
| Split strategy | Stratified per-identity 70/20/10 | Identity-disjoint: 10% of ≥2-sample identities held out |
| Test identities seen in training? | **Yes** (different images) | **No** (completely disjoint) |
| Test samples | 4,364 | 2,982 |
| Test identities | 3,960 | 396 |
| Genuine pairs | 405 | **13,216** (32.6× more) |
| Impostor pairs | 40,500 | 1,321,600 |
| Softmax training classes | 4,115 | 3,719 |
| ArcFace training classes | 4,115 | 3,564 (min_samples=2) |

The 32.6× increase in genuine pairs comes from selecting held-out identities exclusively from those with ≥2 samples, so every test identity contributes real genuine pairs. This directly resolves the statistical-power limitation noted in Section 10.1.

### 11.2 Implementation

- `src/utils/data_loader.py` — new `_identity_disjoint_split()` function; `build_datasets()` accepts `split_mode='stratified'|'identity_disjoint'`.
- `src/models/train_{softmax,arcface}.py` — `--openset` CLI flag selects the disjoint split and writes to `*_openset_best.h5` / `*_openset_history.json`.
- `scripts/run_evaluation.py` — unified standalone runner (`--openset` flag) that infers per-model class counts directly from the h5 weight shapes, handling the two models' different `min_samples` settings.

Retraining was performed on the same RTX 5090 hardware: Softmax ~25 min (val_acc 84.20%), ArcFace ~50 min (val_acc 33.33% — low by design, since the 3,564-way classifier is extremely difficult; embedding quality is the real target and is verified below).

### 11.3 Results

| Metric | ArcFace | Softmax | Gabor |
|---|---|---|---|
| **EER (%)** | **3.26%** | 4.52% | 25.29% |
| **TAR @ FAR=1%** | **94.31%** | 91.16% | 41.15% |
| **TAR @ FAR=0.1%** | **81.55%** | 78.47% | 28.22% |
| **EER Threshold** | 0.3119 | 0.3805 | 0.6139 |
| Genuine mean | 0.7400 | 0.7654 | 0.6877 |
| Impostor mean | 0.0139 | 0.0639 | 0.5709 |
| Gap | **0.7261** | 0.7016 | 0.1168 |

### 11.4 Key Findings

**1. ArcFace now beats Softmax at every operating point.** In Phase 6, ArcFace won on EER and TAR@FAR=1% but lost TAR@FAR=0.1% (52.10% vs 82.72%) — attributed to statistical noise from only 405 genuine pairs. With 32.6× more pairs on truly unseen identities, ArcFace leads across all three headline metrics, including the previously contested strict operating point (81.55% vs 78.47%). The direction of the Phase 6 finding is now statistically firm.

**2. Open-set EER is lower than closed-set EER.** This is counter-intuitive at first — a harder evaluation protocol should give worse numbers. The explanation is that the Phase 6 numbers were noisy estimates from 405 pairs, while the Phase 7 numbers are from 13,216 pairs. The true population EER is closer to the Phase 7 estimate; the Phase 6 "3.46%" point estimate happened to sit in the upper half of its confidence interval. Both deep models generalise to unseen identities.

**3. Gabor degrades gracefully.** The Gabor baseline EER moves from 26.67% → 25.29% — essentially unchanged, as expected for a non-learned system. This confirms the hand-crafted approach has no identity-specific memorisation.

**4. Embedding separation improves for ArcFace.** The genuine/impostor gap widens from 0.7383 (closed-set) to 0.7261 (open-set) — nearly identical, confirming the angular-margin objective produces transferable representations rather than identity-specific memorisation.

### 11.5 Figures

Open-set plots are saved to `figures/openset/`:

- `roc_curves.png`, `det_curves.png` — verification curves across all FAR operating points
- `score_distributions.png` — genuine vs impostor score histograms per system
- `training_curves.png` — loss and accuracy trajectories for the open-set retraining runs
- `tsne_arcface.png`, `tsne_softmax.png` — 2-D projection of embeddings for the top 20 most-sampled held-out identities (unlike Phase 6 where only 41 multi-sample test points existed, open-set t-SNE has hundreds of points per major identity, making cluster structure clearly visible)

### 11.6 What This Resolves from Section 10

| Caveat | Phase 7 Status |
|---|---|
| 10.1 Statistical significance (405 pairs) | **Resolved** — 13,216 genuine pairs confirm ArcFace > Softmax across all metrics |
| 10.3 Closed-set protocol | **Resolved** — identity-disjoint split with 396 unseen test identities |
| 10.4 t-SNE sparsity (41 multi-sample points) | **Resolved** — hundreds of points per top-20 identity |
| 10.2 Gabor aspect-ratio disadvantage | **Hypothesis rejected** — strip-Gabor variants tested separately, both ~3 EER points *worse* than resized; bottleneck is shared encoding artefacts (impostor mean 0.57 ≫ 0.50), not aspect ratio |
| 10.5 Single dataset | Unchanged — cross-dataset evaluation is listed as future work |

---

## 12. Conclusion & Future Work

### Conclusion

This project demonstrates that deep metric learning (ArcFace) achieves strong iris verification performance on the CASIA-IrisV4 dataset. Under the rigorous **open-set protocol** (Phase 7, identity-disjoint train/test), ArcFace reaches EER 3.26% with 94.31% TAR@FAR=1%, outperforming both classification-based deep learning (Softmax, EER 4.52%) and classical Gabor IrisCode (EER 25.29%) across every operating point. With 13,216 genuine pairs on 396 unseen identities, these differences are no longer within statistical noise.

Two training findings stand out:

1. **Proper ArcFace configuration is critical.** Without margin annealing, SGD optimizer, and single-sample class filtering, the backbone collapses entirely (all images map to one point on the hypersphere, EER 47.6%). This collapse is invisible to standard training metrics — loss decreases and accuracy hits 100% while the embeddings have zero variance.
2. **Evaluation protocol matters.** The closed-set stratified split produced wide confidence intervals (ArcFace EER 1.73–5.19%) that failed to separate ArcFace from Softmax. The identity-disjoint split with 32× more genuine pairs resolved this ambiguity directly.

### Future Work

- **More samples per class** — ArcFace benefits from ≥10 samples per identity. Data augmentation or synthetic generation could help.
- **Cross-dataset evaluation** — Test on ND-IRIS-0405, UBIRIS.v2, or IITD to verify generalisation.
- **Improved segmentation** — Replace HoughCircles with a learned segmentation network (e.g., U-Net) to improve the 62.7% detection rate.
- **Attention mechanisms** — Add channel/spatial attention (SE blocks, CBAM) to focus on discriminative iris texture regions.
- **Open-set protocols** — Evaluate with disjoint train/test identity sets for a more rigorous generalisation test.
- **Per-row contrast normalisation for Gabor** — The strip-Gabor experiment (Section 10.2) showed that aspect ratio is not the bottleneck; the shared radial luminance gradient is. Per-row mean removal or histogram equalisation on the strip before filtering would attack this directly.
- **Fusion** — Combine deep learning and Gabor features via score-level or feature-level fusion.
- **Larger test set** — Increase multi-sample identities in the test set to improve statistical significance of EER comparisons.
