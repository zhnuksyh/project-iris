# Project Iris — Process Journal

**Improving Image Detection for Biometric (Iris) Authentication Systems**
End-to-end record of every step from raw images to final evaluation.

This document is a procedural log, not a project report. The companion file [phase6_evaluation_report.md](phase6_evaluation_report.md) (and its dashboard [phase6_evaluation_report.html](phase6_evaluation_report.html)) presents the *findings*; this file walks through *how each finding was produced* — design choices, code references, what worked, what failed, what was rejected.

---

## Table of Contents

1. [Preface](#1-preface)
2. [Phase 1 — Environment & Data Acquisition](#2-phase-1--environment--data-acquisition)
3. [Phase 2 — Preprocessing Pipeline](#3-phase-2--preprocessing-pipeline)
4. [Phase 3 — IrisNet Architecture](#4-phase-3--irisnet-architecture)
5. [Phase 4 — Gabor Filter Baseline](#5-phase-4--gabor-filter-baseline)
6. [Phase 5 — Model Training](#6-phase-5--model-training)
7. [Phase 6 — Closed-Set Evaluation](#7-phase-6--closed-set-evaluation)
8. [Phase 7 — Open-Set Evaluation](#8-phase-7--open-set-evaluation)
9. [Strip-Gabor Experiment (Section 10.2)](#9-strip-gabor-experiment-section-102)
10. [Final State of the Repository](#10-final-state-of-the-repository)

---

## 1. Preface

### Goal

Build, train, and evaluate three iris-recognition systems on CASIA-IrisV4 and report verification metrics (EER, TAR@FAR) on a held-out test set:

| System | Type | Representation | Metric |
|---|---|---|---|
| **ArcFace** | Deep metric learning | 512-D L2-normalised embedding | Cosine similarity |
| **Softmax** | Deep classification | 512-D L2-normalised embedding | Cosine similarity |
| **Gabor** | Classical hand-crafted | 262,144-bit binary IrisCode | 1 − Hamming distance |

### Pipeline at a glance

```
data/raw/ (CASIA-IrisV4 JPEGs, ~48k)
   │
   ▼  Phase 2: src/preprocessing/segmentation.py + batch_processor.py
data/processed/ (30,626 .npy tensors, shape (128, 128, 1))
   │
   ├── Phase 4: src/models/gabor_baseline.py        ─→ 262,144-bit codes
   │
   └── Phase 5: src/models/architecture.py + ...    ─→ IrisNet backbones
                src/models/train_softmax.py           ─→ models/softmax_best.h5
                src/models/train_arcface.py           ─→ models/arcface_best.h5
   │
   ▼  Phase 6/7: scripts/run_evaluation.py
reports/phase6_closedset_results.json
reports/phase7_openset_results.json
figures/*.png  +  figures/openset/*.png
```

### Repository layout

[src/preprocessing/](../src/preprocessing/) — denoise, segment, normalise, batch driver
[src/models/](../src/models/) — IrisNet, ArcFaceLayer, Gabor encoder, training scripts
[src/utils/](../src/utils/) — data loader (splits, augmentation), verification metrics
[src/evaluation/](../src/evaluation/) — pair generation, scoring, plots
[scripts/](../scripts/) — `run_evaluation.py`, `generate_test_strips.py`
[notebooks/](../notebooks/) — exploratory notebooks (preprocessing, Gabor, training, evaluation)
[data/](../data/) — `raw/`, `processed/`, `processed_strip/`, split JSONs (all but split JSONs gitignored)
[models/](../models/) — `*_best.h5`, `*_history.json` (gitignored, copied back manually)
[reports/](../reports/) — this journal, the report, the HTML dashboard, results JSONs
[figures/](../figures/) — closed-set plots; `figures/openset/` for open-set plots

---

## 2. Phase 1 — Environment & Data Acquisition

### Stack

| Component | Choice | Reason |
|---|---|---|
| Python | 3.11 | TensorFlow 2.21 wheel availability |
| TensorFlow | 2.21 (CUDA) | RTX 5090 (Blackwell, sm_120) |
| Image I/O | OpenCV 4.x | HoughCircles, Gabor kernels, `filter2D` |
| Package mgr | `uv` + `.venv` | Reproducible installs |
| GPU | NVIDIA RTX 5090, 32 GB VRAM | All training |

The TensorFlow XLA autotuner crashes on Blackwell (sm_120a), so `TF_XLA_FLAGS=--tf_xla_auto_jit=0` is set at the top of [scripts/run_evaluation.py:23](../scripts/run_evaluation.py#L23) and inference uses `model(x, training=False)` rather than `model.predict()`. Pip-installed NVIDIA shared libraries are wired into `LD_LIBRARY_PATH` by the same prelude.

### Dataset

CASIA-IrisV4, four subsets:

| Subset | Raw images | Notes |
|---|---|---|
| CASIA-Iris-Interval | 2,639 | Clean baseline, NIR illuminator |
| CASIA-Iris-Lamp | 16,212 | Ring illuminator, specular reflections |
| CASIA-Iris-Thousand | 20,000 | Large scale, heavy occlusion variation |
| CASIA-Iris-Syn | 10,000 | Synthetic deformations |
| **Total raw** | **48,851** | |

The four subsets have inconsistent directory nesting (Interval/Lamp wrap once, Thousand wraps twice, Syn is flat). The path resolver in [scripts/generate_test_strips.py:32-58](../scripts/generate_test_strips.py#L32-L58) handles this via a per-subset prefix table — important for any code that needs to walk back from `data/processed/` to `data/raw/`.

---

## 3. Phase 2 — Preprocessing Pipeline

Driver: [src/preprocessing/batch_processor.py](../src/preprocessing/batch_processor.py).
Per-image steps: [src/preprocessing/segmentation.py](../src/preprocessing/segmentation.py).

For each raw `.jpg` we produce one `(128, 128, 1) float32` `.npy` tensor in `data/processed/`. The pipeline is deterministic and idempotent — re-running skips files that already exist.

### Step 1 — Denoise

[`denoise_image(image_path)`](../src/preprocessing/segmentation.py#L18). Reads grayscale and applies `cv2.GaussianBlur` with kernel `(5, 5)` and sigma `1.0`. The point of this step is to suppress lash speckle so that HoughCircles finds the pupil reliably, *not* to denoise for the CNN — the CNN sees the un-denoised normalised strip downstream.

### Step 2 — Iris Segmentation

[`segment_iris(blurred)`](../src/preprocessing/segmentation.py#L48) does an iterative HoughCircles search. The key design choice: rather than tuning a single `param2` (accumulator threshold) per dataset, we sweep `param2` from a high value down to a floor and accept the first sweep step that returns at least one circle:

```python
def _detect_all(image, dp, min_dist, p1, p2_start, p2_min, p2_step, min_r, max_r):
    p2 = p2_start
    while p2 >= p2_min:
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, min_dist,
                                   param1=p1, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is not None and len(circles[0]) > 0:
            return circles[0]
        p2 -= p2_step
    return None
```

Two passes:

1. **Pupil pass** with `min_r=15, max_r=70`. Pupil is darker; we threshold the inverted image first.
2. **Iris pass** centred at the pupil with `min_r = pupil_r * 1.5, max_r = pupil_r * 5.0`.

Concentric-circle assumption — pupil and iris share the same centre — keeps the rubber-sheet downstream simple. About 62.7% of raw images survive segmentation; the remainder are skipped (logged in [batch_processor.py:91](../src/preprocessing/batch_processor.py#L91)).

### Step 3 — Daugman Rubber-Sheet Normalisation

[`normalize_iris(image, pupil, iris, width=512, height=64)`](../src/preprocessing/segmentation.py#L137). Maps the annular iris region to a rectangular polar strip:

```
  x(rho, theta) = x_pupil(theta) + rho * (x_iris(theta) - x_pupil(theta))
  y(rho, theta) = y_pupil(theta) + rho * (y_iris(theta) - y_pupil(theta))
  rho ∈ [0, 1)   indexes rows (radial)
  theta ∈ [0, 2π) indexes cols (angular)
```

Implementation uses `np.meshgrid` to build the `(64, 512)` `(map_x, map_y)` arrays and then `cv2.remap` with `INTER_LINEAR` and `BORDER_REPLICATE`. The strip is angularly-cyclic (column 0 ≡ column 512) — this fact comes back in the Gabor experiment in [Section 9](#9-strip-gabor-experiment-section-102).

Convention used here: `theta=0` is at 3 o'clock; `theta` increases towards the bottom (because `cv2`'s y-axis points down and we use `sin(theta)` directly). So `theta = 3π/2` corresponds to 12 o'clock — column 384 of 512 — which is where eyelid occlusion concentrates.

### Step 4 — Scale to CNN Input

[`scale_pixels(strip, target_shape=(128, 128))`](../src/preprocessing/segmentation.py#L205) does an isotropic `cv2.resize` from `(64, 512)` to `(128, 128)`, casts to `float32`, divides by 255, and adds a channel axis. **The information loss from this resize was hypothesised to bottleneck the Gabor baseline; that hypothesis was tested in Phase 7 and rejected** ([Section 9](#9-strip-gabor-experiment-section-102)).

### Step 5 — Batch Driver

[`run_all()`](../src/preprocessing/batch_processor.py#L108) iterates each subset, mirrors the directory structure under `data/processed/`, and saves the `.npy`. Final yield: **30,626 tensors over 4,115 unique identities**.

---

## 4. Phase 3 — IrisNet Architecture

[src/models/architecture.py](../src/models/architecture.py). A MiniIrisXception-style CNN, 423 K parameters, output is a 512-D L2-normalised embedding.

### Block-level structure

| Stage | Block | In → Out shape | Parameters |
|---|---|---|---|
| Init | `Conv2D(32, 3) + BN + ReLU + MaxPool(2)` | (128, 128, 1) → (64, 64, 32) | 320 |
| Entry 1 | `_entry_block(64)` | (64, 64, 32) → (32, 32, 64) | 7,808 |
| Entry 2 | `_entry_block(128)` | (32, 32, 64) → (16, 16, 128) | 28,288 |
| Middle 1 | `_middle_block(128, drop=0.2)` | (16, 16, 128) → (16, 16, 128) | 33,920 |
| Middle 2 | `_middle_block(128, drop=0.2)` | (16, 16, 128) → (16, 16, 128) | 33,920 |
| Middle 3 | `_middle_block(128, drop=0.2)` | (16, 16, 128) → (16, 16, 128) | 33,920 |
| Exit 1 | `_entry_block(256)` | (16, 16, 128) → (8, 8, 256) | 109,824 |
| Head | `GAP + Dense(512, no bias) + L2-norm` | (8, 8, 256) → (512,) | 131,072 |
| **Total** | | | **422,976** |

### Building blocks

[`_entry_block`](../src/models/architecture.py#L22) — depthwise-separable conv pair, BN, ReLU, max-pool, plus a 1×1 conv shortcut to match channel count and stride. Doubles channels and halves spatial resolution per call.

[`_middle_block`](../src/models/architecture.py#L44) — same separable-conv pattern but with an identity shortcut (no spatial change). Dropout `0.2` on the second branch.

[`build_irisnet`](../src/models/architecture.py#L62) — composes the stages and ends with `GlobalAveragePooling2D → Dense(512) → tf.math.l2_normalize`. The L2 normalisation is what makes cosine similarity = dot product downstream — both the softmax and ArcFace heads sit on top of this normalised embedding.

### Why depthwise-separable

The raw count would be ~3× larger with vanilla `Conv2D(3×3)` pairs; depthwise-separable splits each spatial conv into a depthwise pass (3×3 per channel) plus a pointwise pass (1×1 across channels). Same receptive field, ~9× fewer parameters per separable pair. A 423K-param backbone fits comfortably in 32 GB VRAM at batch 64 with margin for the ArcFace 512×N classifier head.

---

## 5. Phase 4 — Gabor Filter Baseline

[src/models/gabor_baseline.py](../src/models/gabor_baseline.py).

Classical iris encoder: convolve the (128, 128) image with a bank of 2D Gabor filter pairs (real cos + imaginary sin), binarise the sign of each response, and concatenate into one 1-D bit vector.

### Filter bank

Constants at the top of [gabor_baseline.py:28-38](../src/models/gabor_baseline.py#L28-L38):

```
SCALES       = 4   (kernel sizes 9, 13, 17, 23 px; sigmas 2, 3, 4, 6)
ORIENTATIONS = 8   (angles 0, π/8, 2π/8, …, 7π/8)
Filter pairs = 4 × 8 = 32 quadrature pairs
SUBSAMPLE    = 2   (decimate response 2× per axis to drop adjacent-pixel correlation)
Bits per code = 32 × 2 (real+imag) × 64 × 64 = 262,144
```

Kernels are L2-normalised so amplitudes are scale-comparable. The bank is built once at import time ([line 80](../src/models/gabor_baseline.py#L80)) and cached.

### Encoding

[`extract_iris_code(img)`](../src/models/gabor_baseline.py#L97):

1. Strip channel axis → 2-D grayscale.
2. Subtract the image mean (zero DC). **This step is essential** — without it the [0, 1] pixel range creates a positive offset in every filter response and ~75% of bits flip to 1, destroying discriminability.
3. For each `(k_real, k_imag)`: `cv2.filter2D` real and imag, subsample 2× on both axes, binarise `> 0`.
4. Stack and ravel.

### Comparison

[`calculate_hamming_distance(code1, code2)`](../src/models/gabor_baseline.py#L267) — fractional Hamming Distance:

```
HD = popcount(code1 XOR code2) / total_bits
```

Vectorised batch version in [src/evaluation/pairs.py:83](../src/evaluation/pairs.py#L83): processes pairs in chunks of 1,000 to keep memory bounded.

Score reported as `1 − HD` so higher = more similar (consistent with cosine).

---

## 6. Phase 5 — Model Training

### 6.1 Data loader

[src/utils/data_loader.py](../src/utils/data_loader.py).

Two-pass design: first discover all `(identity → list of files)` from `data/processed/`, then split, then build `tf.data.Dataset`s.

#### Discovery — [`_discover`](../src/utils/data_loader.py#L67)

Walks the processed tree, groups files by identity-key (`subset/<id>/<L|R>` or `subset/<id>` for the flat Syn layout). Returns `{identity_str: [path, ...]}`. Identity strings serve as stable IDs across re-runs.

#### Splits

Two modes, selected by `build_datasets(split_mode='stratified'|'identity_disjoint')`:

| Mode | Used by | Function |
|---|---|---|
| `stratified` (default) | Phase 5 closed-set | [`_stratified_split`](../src/utils/data_loader.py#L95) |
| `identity_disjoint` | Phase 7 open-set | [`_identity_disjoint_split`](../src/utils/data_loader.py#L132) |

**Stratified rules:**
- ≥3 samples: 70/20/10 train/val/test, with at least 1 sample per split.
- 2 samples: 1 train + 1 test (val borrows from train at training time).
- 1 sample: train only.

This means closed-set test identities are *also* train identities (different images, same person). The model has seen the identity during training; it has not seen the test image.

**Identity-disjoint rules** (open-set, [Section 8](#8-phase-7--open-set-evaluation)):
- Pick `OPENSET_TEST_IDENT_FRAC=10%` of identities with ≥2 samples → all their samples → test only.
- Remaining identities → 80/20 sample-level split for train/val. Singletons → train only.
- The model never sees a test identity. Only embedding generalisation is being tested.

Test split manifests are written to `data/test_split.json` and `data/test_split_openset.json` so evaluation can reload the exact split without rerunning the partition logic.

#### Augmentation — [`_make_tf_dataset:248`](../src/utils/data_loader.py#L248)

Applied to train only; never to val/test:

```python
augmentor = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=0.05, fill_mode='reflect'),
    tf.keras.layers.RandomZoom(0.05, fill_mode='reflect'),
    tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode='reflect'),
    tf.keras.layers.GaussianNoise(0.01),
])
```

Magnitudes are deliberately conservative — iris geometry is fragile. Rotation simulates head tilt; zoom simulates capture distance; translation simulates segmentation centring imperfections; Gaussian noise simulates sensor noise. Augmentation was strengthened from the pre-Phase-5 baseline (rotation only) as part of fixing the ArcFace embedding collapse ([Section 6.4](#64-arcface-collapse--diagnosis--fix)).

### 6.2 Softmax training

Driver: [src/models/train_softmax.py](../src/models/train_softmax.py).

Architecture is the IrisNet backbone plus a `Dense(num_classes, activation='softmax')` head.

| Setting | Value |
|---|---|
| Optimizer | Adam(lr=1e-3) |
| Loss | CategoricalCrossentropy |
| Batch size | 32 |
| Epochs (max) | 50 |
| Callbacks | ModelCheckpoint(val_loss, save_best_only), EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6) |
| min_samples filter | 1 (singletons OK) |

Outputs:
- `models/softmax_best.h5` — best-by-val_loss
- `models/softmax_history.json` — per-epoch loss/accuracy

Closed-set run: 4,115 classes, val accuracy 84.56% by epoch 50, train accuracy 97.81% (mild over-fit, expected at ~7.5 samples/class).

### 6.3 ArcFace training (final config)

Driver: [src/models/train_arcface.py](../src/models/train_arcface.py). Layer: [src/models/arcface_loss.py:29](../src/models/arcface_loss.py#L29).

#### Architecture (training time)

Two inputs:
```
image (128, 128, 1)  ─┐
                      ├─ IrisNet ─ ArcFaceLayer ─ scaled logits
label_onehot          ─┘
```

`label_onehot` is needed by ArcFace because the angular margin is added only to the true-class angle. At inference we drop the head and use the IrisNet backbone alone.

`_adapt_dataset_for_arcface` ([train_arcface.py:178](../src/models/train_arcface.py#L178)) re-maps the standard `(image, label_onehot)` dataset to `({'image': x, 'label_onehot': y}, y)` so `model.fit()` sees both inputs and the target.

#### ArcFaceLayer mechanics — [arcface_loss.py:29](../src/models/arcface_loss.py#L29)

Per the ArcFace paper:

```
cos_theta_yi = embedding · W_yi          (W column for true class)
target       = cos(theta_yi + m)         (penalise true class by margin m)
logit_i      = scale * cos_theta_i       (others unchanged)
```

The margin and scale were converted from constants to `tf.Variable` so a callback can anneal them ([Section 6.4](#64-arcface-collapse--diagnosis--fix)).

The "easy-margin" boundary guard is in `call()`: when `cos(theta) < cos(π − m)`, `theta + m` exceeds π and the target flips sign incorrectly. The fallback `safe_logit = cos_theta − sin(m) · m` keeps the penalty monotonic across the boundary.

#### Hyperparameters

| Setting | Value | Note |
|---|---|---|
| Optimizer | SGD(lr=0.01, momentum=0.9, weight_decay=5e-4) | Not Adam — see [Section 6.4](#64-arcface-collapse--diagnosis--fix) |
| Loss | CategoricalCrossentropy(from_logits=True) | ArcFace outputs scaled logits, not probabilities |
| Batch size | 64 | RTX 5090 has VRAM headroom |
| Epochs | 50 | |
| Margin schedule | 0 → 0.5 (linear over 5–19) | Warmup epochs 0–4 (m=0); ramp 5–19; full 20+ |
| Scale schedule | 16 → 64 (linear over 5–19) | Same schedule |
| LR schedule | PiecewiseConstantDecay at epochs 25/35/45 (×0.1 each) | Standard ArcFace step decay |
| Min samples/class | 2 | Single-sample classes can't form genuine pairs and produce degenerate ArcFace gradients |

The `MarginScaleAnnealingCallback` ([train_arcface.py:69](../src/models/train_arcface.py#L69)) is the entire annealing mechanism: at each `on_epoch_begin` it recomputes `(m, s)` and writes to the layer's `margin_var` / `scale_var`. EarlyStopping is intentionally **not** used — the margin ramp causes temporary val-loss spikes that mislead patience-based stopping.

Outputs:
- `models/arcface_best.h5` — best-by-val_accuracy full model
- `models/arcface_backbone.weights.h5` — backbone weights for inference
- `models/arcface_history.json`

### 6.4 ArcFace collapse — diagnosis & fix

The first ArcFace training produced a *catastrophically degenerate* model: every test image mapped to the same point on the hypersphere, EER on the test set was 47.6% (random chance). Yet training loss decreased monotonically (5.05 → 1.58 over 50 epochs) and training accuracy hit 100% by epoch 10.

Root-cause analysis (five compounding issues):

1. **Adam optimizer + no LR decay.** `ReduceLROnPlateau(monitor='val_loss', patience=5)` never fired because val_loss kept slowly decreasing; LR stayed at 1e-3 the entire run. Adam's per-parameter adaptive rates let the 2.1M-param ArcFace W-matrix learn fast while the 423K-param backbone got starved of effective updates.
2. **Full margin m=0.5 from epoch 0.** With random weights, the easiest minimum is "collapse all embeddings to one point on the sphere; let W classify via column-angle." The W-matrix has the capacity (512 × 4115 ≈ 2.1M params), so it does.
3. **High scale s=64 from epoch 0.** Scale × 64 saturates softmax in early epochs; `acos` gradient vanishes when `cos(theta) → ±1`, choking gradient flow to the backbone.
4. **Singleton classes.** ~1,950 of 4,115 classes had only one sample. Margin loss can't enforce intra-class compactness with one sample, so those gradients are degenerate and pull the backbone toward memorisation.
5. **Weak augmentation.** Only `RandomRotation(0.05)` was applied, so the model memorised quickly and lost gradient signal.

**Diagnosis tell**: training accuracy hit 100% by epoch 10 while loss kept decreasing for another 40 epochs. That gap is the W-matrix optimising on its own with no embedding signal — the signature of collapse.

**Fix** — six combined changes, all in train_arcface.py and arcface_loss.py:

1. Switch optimizer to **SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)**. SGD has no per-parameter adaptive rates; backbone and W-matrix receive gradients at the same effective scale. Weight decay 5e-4 prevents the backbone from collapsing to a fixed point. This is what InsightFace, AdaFace, and ElasticFace all use.
2. **PiecewiseConstantDecay** for LR (boundaries 25/35/45 epochs, multiplicative 0.1 each step). Deterministic, doesn't depend on val-loss behaviour.
3. **Margin/scale annealing**: warmup epochs 0–4 with m=0, s=16 (pure scaled-cosine softmax); ramp 5–19 linearly to m=0.5, s=64; full target after.
4. **Boundary guard** in `ArcFaceLayer.call()` (above) — prevents the cos(theta+m) sign flip when theta+m > π.
5. **min_samples=2** filter in `build_datasets()` — exclude singleton classes before the splitter even sees them.
6. **Stronger augmentation** — zoom + translate + Gaussian noise on top of rotation.

Re-running with the fixed config: ArcFace converged smoothly, val_accuracy crossed 50% by epoch 10 (no more 100% trap), test EER dropped to 3.46%, beating the Softmax baseline.

The lesson, encoded in the report's Section 9.3: **monitor embedding diversity (`emb.std(axis=0).mean()`), not just classification metrics**. A collapsed model has near-zero embedding std but can still hit 100% training accuracy.

---

## 7. Phase 6 — Closed-Set Evaluation

Driver: [scripts/run_evaluation.py](../scripts/run_evaluation.py) (no `--openset` flag).

### Step 1 — Load test split

[`load_test_split('data/test_split.json')`](../src/utils/data_loader.py#L395) returns `(paths, labels, num_classes)`. Test set: 4,364 samples, 3,960 unique identities, of which 3,557 have only 1 sample (singletons can't form genuine pairs).

### Step 2 — Generate verification pairs

[`generate_pairs(labels, seed=42, impostor_ratio=100)`](../src/evaluation/pairs.py#L14):

- **Genuine** = all `(i, j)` with `labels[i] == labels[j]`. Yields **405 pairs** (most identities are singletons).
- **Impostor** = uniformly random `(i, j)` with `labels[i] != labels[j]`, sampled to match `100×` the genuine count = **40,500 pairs**.

The 1:100 ratio matches biometric convention (FAR is the operationally interesting low-end).

### Step 3 — Extract representations

| System | Function | Output shape | Time on RTX 5090 |
|---|---|---|---|
| ArcFace | [`extract_arcface_embeddings`](../src/evaluation/embeddings.py#L43) | (4364, 512) float32 | ~5 s |
| Softmax | [`extract_softmax_embeddings`](../src/evaluation/embeddings.py#L68) | (4364, 512) float32 | ~5 s |
| Gabor | [`extract_gabor_codes`](../src/evaluation/embeddings.py#L99) | (4364, 262144) bool | ~25 s |

Both deep extractors rebuild the model architecture from code and load h5 weights, then take a sub-model up to the `l2_norm` layer. This avoids Lambda-deserialisation issues that bite when loading the saved h5 directly into Keras 3.

The number of classes per checkpoint is inferred at evaluation time from the h5 weight shapes ([`_infer_softmax_num_classes`](../scripts/run_evaluation.py#L65), [`_infer_arcface_num_classes`](../scripts/run_evaluation.py#L84)) so the eval script doesn't have to be told. This becomes load-bearing in [Phase 7](#8-phase-7--open-set-evaluation) where the two models trained with different `min_samples` end up with different class counts (3,719 vs 3,564).

### Step 4 — Score pairs

- ArcFace, Softmax: [`compute_cosine_scores`](../src/evaluation/pairs.py#L65). Since embeddings are L2-normalised, cosine = dot product. ~1 ms per 1,000 pairs.
- Gabor: [`compute_hamming_scores`](../src/evaluation/pairs.py#L83), chunked at 1,000 pairs/chunk to bound peak memory. Returns `1 − HD`.

### Step 5 — Metrics

[src/utils/metrics.py](../src/utils/metrics.py).

[`compute_eer(genuine, impostor)`](../src/utils/metrics.py#L41) — sweeps thresholds, finds the threshold where FAR = FRR by linear interpolation between the two adjacent points in the sweep.

[`compute_tar_at_far(genuine, impostor, target_far)`](../src/utils/metrics.py#L73) — sweeps thresholds, finds the lowest threshold at which FAR ≤ target_far, returns the corresponding TAR.

Both work with similarity scores (higher = more similar). FAR = `mean(impostor_scores >= threshold)`, FRR = `mean(genuine_scores < threshold)`.

### Step 6 — Plots

[src/evaluation/plotting.py](../src/evaluation/plotting.py) generates:

- `figures/roc_curves.png` — TAR vs FAR (linear)
- `figures/det_curves.png` — FRR vs FAR on probit scale (standard biometric format)
- `figures/score_distributions.png` — genuine vs impostor histograms per system, with EER threshold marked
- `figures/training_curves.png` — loss + accuracy per epoch from `*_history.json`
- `figures/tsne_arcface.png`, `figures/tsne_softmax.png` — 2-D projection of the 512-D embeddings for the 20 most-sampled identities (closed-set has only 41 multi-sample test points total)

### Closed-set results

| Metric | ArcFace | Softmax | Gabor |
|---|---|---|---|
| EER (%) | **3.46** | 4.20 | 26.67 |
| TAR @ FAR=1% | **93.33** | 93.09 | 42.72 |
| TAR @ FAR=0.1% | 51.85 | **82.72** | 29.14 |
| EER threshold | 0.317 | 0.359 | 0.611 |
| Genuine mean | 0.752 | 0.772 | 0.686 |
| Impostor mean | 0.014 | 0.038 | 0.571 |
| Gap | **0.738** | 0.734 | 0.115 |

ArcFace wins EER and TAR@FAR=1%; Softmax wins TAR@FAR=0.1%. The TAR@FAR=0.1% loss for ArcFace was attributed to statistical noise — only 405 genuine pairs gave wide confidence intervals (ArcFace 95% CI [1.73%, 5.19%]). [Phase 7](#8-phase-7--open-set-evaluation) tests this directly.

JSON: [reports/phase6_closedset_results.json](phase6_closedset_results.json).

---

## 8. Phase 7 — Open-Set Evaluation

Goal: address the closed-set caveats noted in Section 10.1 and 10.3 of the report — the test identities were also training identities, and the 405-pair test had wide CIs. We retrain with an identity-disjoint split and re-evaluate.

### 8.1 Identity-disjoint split

Implementation: [`_identity_disjoint_split`](../src/utils/data_loader.py#L132). Triggered by `build_datasets(split_mode='identity_disjoint')`.

- Pool of eligible test identities: those with ≥2 samples (so they can produce genuine pairs).
- Pick `OPENSET_TEST_IDENT_FRAC = 10%` of them for test. **All** their samples go to test.
- Remaining identities: 80/20 sample-stratified split for train/val. Singleton identities → train only.
- The model never sees a test-identity sample during training.

Two label-index mappings are returned:
- `train_label_to_idx` — used by the model's classification head.
- `test_local_label_to_idx` — used at evaluation time by `generate_pairs` to group test samples into genuine pairs.

The test manifest is saved to `data/test_split_openset.json` so eval can reload it without re-running the split.

Numbers: **2,982 test samples / 396 held-out identities / 13,216 genuine pairs / 1,321,600 impostor pairs** (32.6× more genuine pairs than closed-set).

### 8.2 Retraining

`train_softmax.py --openset` and `train_arcface.py --openset` flags introduced. Both:
- Pass `split_mode='identity_disjoint'` to `build_datasets`.
- Write to `*_openset_best.h5` and `*_openset_history.json` so closed-set checkpoints are preserved alongside.

Class-count divergence: Softmax keeps `min_samples=1` (handles singletons fine); ArcFace keeps `min_samples=2` (degenerate gradients otherwise). Result: under the same identity-disjoint partition, **Softmax trains on 3,719 classes, ArcFace on 3,564** — they see slightly different training pools.

Run times on RTX 5090: Softmax ~25 min, ArcFace ~50 min. ArcFace val_accuracy peaks at 33.33% (the 3,564-way classification is genuinely hard; embedding quality is what matters and is verified by evaluation, not val_accuracy).

### 8.3 The class-count bug

First open-set evaluation attempt blew up with:

```
ValueError: Shape mismatch in layer softmax_head — Weight expects (512, 3564),
                                                      received (512, 3719)
```

Cause: the eval script was reading `num_classes` from the test split JSON, but `build_datasets` overwrote `data/test_split_openset.json` on each call, and the *last* write was from the ArcFace training (min_samples=2, 3,564 classes). Loading the Softmax checkpoint (3,719 classes) against this stored count failed.

Fix: at evaluation time, read each model's class count directly from its h5 weight shape. Helpers in [scripts/run_evaluation.py:65-92](../scripts/run_evaluation.py#L65-L92):

```python
def _infer_softmax_num_classes(h5_path: str) -> int:
    import h5py
    with h5py.File(h5_path, 'r') as f:
        for grp in ('model_weights/softmax_head/softmax_head',
                    'model_weights/softmax_head'):
            if grp in f:
                for key in ('kernel', 'kernel:0'):
                    if key in f[grp]:
                        return int(f[grp][key].shape[1])
    raise RuntimeError(...)

def _infer_arcface_num_classes(h5_path: str) -> int:
    # same idea, looks for model_weights/arcface/arcface/arcface_weights
```

These walk the h5 hierarchy (Keras 3 uses no `:0` suffix; older Keras 2 layouts do — both checked) and read the column count of the head's weight matrix. Robust across split modes and different `min_samples` settings.

### 8.4 Open-set results

| Metric | ArcFace | Softmax | Gabor |
|---|---|---|---|
| EER (%) | **3.26** | 4.52 | 25.29 |
| TAR @ FAR=1% | **94.31** | 91.16 | 41.15 |
| TAR @ FAR=0.1% | **81.55** | 78.47 | 28.22 |
| Genuine mean | 0.740 | 0.765 | 0.688 |
| Impostor mean | 0.014 | 0.064 | 0.571 |
| Gap | **0.726** | 0.702 | 0.117 |

ArcFace dominates **all three** metrics on unseen identities — including the strict TAR@FAR=0.1% it lost in closed-set. The closed-set ambiguity at low FAR was indeed statistical noise.

Counter-intuitively, the open-set EER (3.26%) is *lower* than closed-set EER (3.46%) for ArcFace. This is not because the model got better — it's because the closed-set 3.46% point estimate sat in the upper half of its [1.73%, 5.19%] confidence interval, and the open-set 3.26% with 32× more pairs is closer to the true population EER.

Gabor is essentially unchanged (26.67% → 25.29% EER) — expected for a non-learned baseline. Confirms it has no identity-specific memorisation to lose.

JSON: [reports/phase7_openset_results.json](phase7_openset_results.json). Figures: `figures/openset/`.

---

## 9. Strip-Gabor Experiment (Section 10.2)

The Phase 6 report's Section 10.2 hypothesised that the (128, 128) isotropic resize of the (64, 512) rubber-sheet strip was hurting Gabor — the angular dimension gets compressed 4×, distorting the spatial frequencies the filter bank was designed to capture. The user asked to engineer a fairer strip-based Gabor and see if it beats 26.67% EER.

This was tested directly. The hypothesis was rejected.

### 9.1 Strip generation

[scripts/generate_test_strips.py](../scripts/generate_test_strips.py) regenerates `(64, 512)` strips for every test image referenced in either split. Path resolver handles the four CASIA subsets' inconsistent raw layouts via a per-subset prefix table:

```python
SUBSET_RAW_PREFIX = {
    'CASIA-Iris-Interval':  'CASIA-Iris-Interval',
    'CASIA-Iris-Lamp':      'CASIA-Iris-Lamp',
    'CASIA-Iris-Syn':       '',                                   # flat
    'CASIA-Iris-Thousand':  'CASIA-Iris-Thousand/CASIA-Iris-Thousand',  # double-nested
}
```

`ProcessPoolExecutor` parallelises across CPU cores. For 6,915 test images: 6,871 strips produced (99.4%), 40 segmentation failures (0.6%, dropped from the strip-Gabor evaluation but still scored under the (128, 128) Gabor since their tensors exist).

Output: `data/processed_strip/` mirroring the layout of `data/processed/`. Gitignored — regenerable from raw.

### 9.2 Variant A (naive)

[`extract_iris_code_strip(strip)`](../src/models/gabor_baseline.py#L83) — same 32-pair filter bank, applied directly to the (64, 512) strip with `cv2.filter2D` defaults. No occlusion mask. Subsample 2× → 524,288 bits.

| | (128² resized) | (64×512 strip naive) |
|---|---|---|
| EER | 26.67% | **30.04%** |
| Gap | 0.115 | 0.092 |

3.4 EER points *worse*. Hypothesis falsified once.

### 9.3 Variant B (engineered: cyclic + occlusion)

[`extract_iris_code_strip_v2(strip)`](../src/models/gabor_baseline.py#L184) — same filter bank, but with two strip-aware corrections:

**Cyclic angular wrap.** The strip is angularly periodic (column 0 ≡ column 512). `cv2.filter2D`'s default reflect-padding is wrong at the seam. [`_filter_strip_cyclic`](../src/models/gabor_baseline.py#L153) pads the column ends by wrapping, then reflects on rows (radial axis is genuinely bounded), filters, and crops:

```python
padded = np.concatenate([img[:, -pad_x:], img, img[:, :pad_x]], axis=1)
padded = cv2.copyMakeBorder(padded, pad_y, pad_y, 0, 0,
                            borderType=cv2.BORDER_REFLECT_101)
full = cv2.filter2D(padded, cv2.CV_32F, kernel,
                    borderType=cv2.BORDER_CONSTANT)
return full[pad_y:pad_y+H, pad_x:pad_x+W]
```

**Eyelid occlusion mask.** Following the column convention from Phase 2 — `theta = 3π/2` is at the top (12 o'clock, column 384 of 512) — mask `±60°` around there: columns `320..448` (128 cols, 25%). Returns a per-bit mask alongside the code; `calculate_hamming_distance_masked` ([gabor_baseline.py:283](../src/models/gabor_baseline.py#L283)) and the vectorised `compute_hamming_scores_masked` ([pairs.py:114](../src/evaluation/pairs.py#L114)) restrict the comparison to bits valid in **both** samples' masks.

| | (128² resized) | strip naive | strip v2 (cyclic+mask) |
|---|---|---|---|
| EER | **26.67%** | 30.04% | 29.78% |
| TAR@FAR=1% | **42.72%** | 38.46% | 37.72% |
| Gap | **0.115** | 0.092 | 0.091 |

Cyclic wrap and eyelid mask moved EER by 0.3 points (essentially noise). Still 3.1 points worse than the resized baseline. Hypothesis rejected the second time.

### 9.4 Diagnosis: shared encoding bias

Across all three variants:

| Variant | Impostor mean | Gap |
|---|---|---|
| (128² resized) | 0.571 | 0.115 |
| Strip naive | 0.569 | 0.092 |
| Strip v2 | 0.567 | 0.091 |

Impostor pair similarity is ~0.57 in every case, well above the 0.50 floor for statistically independent codes. This residual correlation is the same in both spatial domains.

Likely cause: every iris image has roughly the same radial luminance gradient (pupil edge brighter, iris edge darker), and that shared structure produces a consistent DC-like response that survives binarisation regardless of the Gabor bank's geometry. Re-encoding in polar coordinates doesn't escape this — it preserves the same gradient.

**Implication for the report.** The Gabor 26.67% EER is not the cap of "classical Gabor with better preprocessing"; it's close to the cap of "fixed-filter binary IrisCode on this dataset under our preprocessing." Closing the gap would require attacking the shared-pattern bias directly (per-row contrast normalisation, learned masking, occlusion-aware comparison), not just preserving native resolution.

Section 10.2 of the main report was rewritten from "future work" to "hypothesis tested and rejected"; Future Work now lists per-row contrast normalisation as the actual lever. The strip encoders remain in `src/models/gabor_baseline.py` for reproducibility and future ablations but are *not* wired into `run_evaluation.py` — Gabor is reported as the single (128, 128) baseline.

---

## 10. Final State of the Repository

### Branch & history (newest first)

```
main:
  e220843  docs(report): Section 10.2 - strip-Gabor hypothesis tested and rejected
  c9d1a32  feat(gabor): strip-based encoder for ablating the (128, 128) resize
  a864701  feat(eval): closed-set results JSON for symmetry with open-set
  d487400  chore: ignore regeneratable rubber-sheet strip cache
  d50b70c  docs(phase7): document open-set evaluation in report and dashboard
  2e84d09  feat(phase7): add open-set evaluation artefacts
  d4c619f  feat(eval): standalone run_evaluation.py with --openset mode
  8637727  feat(train): add --openset flag to Softmax and ArcFace trainers
  4ce0126  feat(data): add identity-disjoint split for open-set evaluation
  de23eba  chore: ignore .claude/ session state
  b5fe2b4  docs: correct identity count in evaluation notebook (3,960...)
  ...      (Phase 6 + earlier)
```

### Reproducing every result from scratch

```bash
# Phase 2 — preprocessing (~30 min on RTX 5090, requires data/raw/)
python -m src.preprocessing.batch_processor

# Phase 5 — training (closed-set)
python -m src.models.train_softmax   # ~25 min
python -m src.models.train_arcface   # ~50 min

# Phase 5 — training (open-set)
python -m src.models.train_softmax  --openset   # ~25 min
python -m src.models.train_arcface  --openset   # ~50 min

# Phase 6 — closed-set evaluation
python -m scripts.run_evaluation
# -> reports/phase6_closedset_results.json
# -> figures/*.png

# Phase 7 — open-set evaluation
python -m scripts.run_evaluation --openset
# -> reports/phase7_openset_results.json
# -> figures/openset/*.png

# Optional — strip-Gabor experiment (ablation only, not part of the published baseline)
python -m scripts.generate_test_strips
# -> data/processed_strip/   (~30 s, gitignored)
# Then re-wire extract_gabor_codes_strip_v2 into scripts/run_evaluation.py if desired.
```

### Key artefacts on `main`

| Path | Status |
|---|---|
| `src/`, `scripts/` | All committed. Strip-Gabor encoders kept for reproducibility but not wired into the default eval. |
| `data/test_split.json`, `data/test_split_openset.json` | Committed — split manifests. |
| `data/raw/`, `data/processed/`, `data/processed_strip/` | All gitignored. Regenerable. |
| `models/*.h5`, `models/*.json` | All gitignored. Copied back manually after training. |
| `reports/phase6_evaluation_report.md` | Final report, Sections 1–12 incl. Phase 7 in Section 11 and falsified-hypothesis Section 10.2. |
| `reports/phase6_evaluation_report.html` | Dashboard reflecting both closed-set and open-set numbers. |
| `reports/phase6_closedset_results.json` | Closed-set metrics for symmetry with open-set. |
| `reports/phase7_openset_results.json` | Open-set metrics. |
| `reports/process_journal.md` | This file. |
| `figures/*.png` | Closed-set, 3 systems. |
| `figures/openset/*.png` | Open-set, 3 systems. |

### Headline numbers

| | ArcFace | Softmax | Gabor |
|---|---|---|---|
| **Closed-set EER** | 3.46% | 4.20% | 26.67% |
| **Closed-set TAR@FAR=1%** | 93.33% | 93.09% | 42.72% |
| **Closed-set TAR@FAR=0.1%** | 51.85% | **82.72%** | 29.14% |
| **Open-set EER** | **3.26%** | 4.52% | 25.29% |
| **Open-set TAR@FAR=1%** | **94.31%** | 91.16% | 41.15% |
| **Open-set TAR@FAR=0.1%** | **81.55%** | 78.47% | 28.22% |

ArcFace dominates every category in the rigorous open-set protocol; Softmax retains its single closed-set TAR@FAR=0.1% lead, which the open-set evaluation showed to be statistical noise.
