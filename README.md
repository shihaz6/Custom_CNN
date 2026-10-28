# Custom CNN for Rice Leaf Disease Classification

An end-to-end deep learning project that classifies rice leaf images into four disease
categories using **custom convolutional neural networks built from scratch** (no transfer
learning or pretrained backbones).

The notebook goes beyond a standard train/predict workflow: it audits the dataset for exact
and near-duplicate images, detects **leakage between splits**, rebuilds a group-aware split,
retrains every model variant on the corrected data, and only then evaluates the final model on
a held-out test set. Grad-CAM is used to check that the network looks at actual disease
patterns rather than background.

**Final result: 98.68% accuracy on the held-out test set (299 / 303 correct).**

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Interpretability and Error Analysis](#interpretability-and-error-analysis)
- [Repository Contents](#repository-contents)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Output Artifacts](#output-artifacts)
- [Limitations and Notes](#limitations-and-notes)

---

## Overview

| Item | Detail |
|---|---|
| Task | 4-class rice leaf disease image classification |
| Classes | Bacterialblight, Blast, Brownspot, Tungro |
| Input size | 224 × 224 × 3 (RGB) |
| Framework | TensorFlow / Keras |
| Models | 5 custom CNN configurations (V1–V5) |
| Final model | CNN V4 (V3 architecture, learning rate 0.0003) |
| Test accuracy | **0.9868** |
| Test macro-F1 | **0.9863** |
| Environment | Google Colab, NVIDIA T4 GPU |

---

## Results

### Held-out test set (final CNN V4)

| Metric | Score |
|---|---|
| Accuracy | **0.9868** |
| Macro Precision | 0.9863 |
| Macro Recall | 0.9863 |
| Macro F1 | 0.9863 |
| Cross-Entropy Loss | 0.1084 |
| Correct / Total | 299 / 303 |

**Per-class performance**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacterialblight | 0.99 | 0.99 | 0.99 | 73 |
| Blast | 0.97 | 0.97 | 0.97 | 68 |
| Brownspot | 0.99 | 0.99 | 0.99 | 87 |
| Tungro | 1.00 | 1.00 | 1.00 | 75 |
| **Macro avg** | **0.99** | **0.99** | **0.99** | **303** |

### Corrected validation set (used for model selection)

| Model | Configuration | Validation Accuracy | Macro F1 | Min Val Loss | Epochs |
|---|---|---|---|---|---|
| CNN V1 | Baseline | 0.9375 | 0.9365 | 0.1457 | 40 |
| CNN V2 | Augmentation | 0.9474 | 0.9463 | 0.1315 | 40 |
| CNN V3 | Augmentation + BatchNorm + Dropout | 0.9803 | 0.9795 | 0.0802 | 28 |
| **CNN V4** | **V3 + lower learning rate (0.0003)** | **0.9868** | **0.9867** | 0.1049 | 29 |
| CNN V5 | V3 without dropout | 0.9474 | 0.9464 | 0.1158 | 25 |

### Why the split was rebuilt

On the *original* random split, CNN V3 reached **0.9969** validation accuracy — a suspiciously
high score. A reproducibility check across three seeds showed large instability:

| Seed | Accuracy | Macro F1 | Min Val Loss |
|---|---|---|---|
| 42 | 0.9969 | 0.9969 | 0.0389 |
| 123 | 0.9782 | 0.9782 | 0.0928 |
| 2026 | 0.9564 | 0.9564 | 0.1189 |

A perceptual-hash audit then found near-duplicate pairs *across* partitions (minimum Hamming
distance of 6; 13 train↔validation pairs and 14 train↔test pairs at distance ≤ 6). This
indicated leakage, so the split was rebuilt with `StratifiedGroupKFold` on similarity groups.
After rebuilding, no similarity group or file is shared between partitions, and CNN V4 becomes
the strongest model with a more trustworthy score.

---

## Dataset

**Source:** [`nirmalsankalana/rice-leaf-disease-image`](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image)
on Kaggle, downloaded programmatically via `kagglehub`.

### Raw dataset

| Property | Value |
|---|---|
| Total readable images | 5,932 |
| Unreadable files | 0 |
| Dominant resolution | 300 × 300 (4,624 images) |
| Colour modes | RGB (5,776), RGBA (156) |

| Class | Images |
|---|---|
| Brownspot | 1,600 |
| Bacterialblight | 1,584 |
| Blast | 1,440 |
| Tungro | 1,308 |

### Cleaning pipeline

| Stage | Images | Method |
|---|---|---|
| Raw scanned images | 5,932 | Pillow validation |
| Exact-unique | 4,794 | SHA-256 hash deduplication |
| pHash representatives | 2,927 | Perceptual hash (`phash`, 8×8) |
| Final representatives | **2,138** | Union-Find on Hamming distance ≤ 8 |

Near-duplicate inspection with SSIM confirmed that grouped images were genuinely similar, and
**zero** cross-class duplicate pairs were found, so grouping did not mix disease labels.

Final class balance after cleaning: Brownspot 606, Tungro 538, Bacterialblight 514, Blast 480.

### Splits

| Split | Initial (random, stratified) | Corrected (group-aware) |
|---|---|---|
| Train | 1,496 | 1,531 |
| Validation | 321 | 304 |
| Test | 321 | 303 |
| Total | 2,138 | 2,138 |

The corrected split uses 7-fold `StratifiedGroupKFold` for the test partition (~15%) and a
further grouped split for validation, verified with 0 shared groups and 0 shared files.

---

## Methodology

The notebook is organised into 17 sections:

1. **Dataset loading and validation** — KaggleHub download, Pillow scan, resolution/aspect-ratio/mode statistics, class distribution.
2. **Duplicate and near-duplicate analysis** — SHA-256 exact matching, perceptual hashing, SSIM comparison, visual inspection of candidate pairs.
3. **Cleaned representative dataset** — one image retained per similarity group using a Union-Find structure over Hamming distance.
4. **Initial train/validation/test split** — stratified 70/15/15 with overlap checks.
5. **TensorFlow input pipeline** — `decode_image`, aspect-preserving `resize_with_pad`, 0–1 normalization, batching, prefetching, and shape/range assertions.
6. **CNN V1** — baseline convolutional model.
7. **CNN V2** — adds data augmentation.
8. **CNN V3** — adds batch normalization and dropout.
9. **CNN V4 / V5 ablations** — lower learning rate vs. removing dropout.
10. **Reproducibility and leakage audit** — multi-seed retraining plus cross-split perceptual-hash distance analysis.
11. **Corrected grouped split** — `StratifiedGroupKFold`-based partitions with explicit leakage verification.
12. **Retraining on corrected splits** — all five variants rebuilt from scratch on the corrected data.
13. **Final model selection** — highest validation macro-F1, preserved as a frozen checkpoint plus JSON metadata.
14. **Held-out test evaluation** — single inference pass on the untouched test set.
15. **Error analysis** — misclassified images and their full class-probability vectors.
16. **Grad-CAM interpretability** — heatmaps for both incorrect and correct predictions, using a logits-based implementation verified against the model's own outputs.
17. **Single-image prediction demo** — Colab upload widget for classifying a new leaf photo.

**Training configuration (all models):** Adam optimizer, sparse categorical cross-entropy,
batch size 32, up to 40 epochs, `EarlyStopping` on `val_loss` (patience 8, best weights
restored), `ModelCheckpoint` on best `val_loss`, and `CSVLogger` for per-epoch history.

**Preprocessing:** RGB conversion, aspect-preserving resize to 224 × 224, division by 255.

---

## Model Architectures

All models are small, fully custom CNNs with a shared skeleton:

```
Input (224×224×3)
  → [Optional augmentation]
  → Conv2D(32) → [BN] → ReLU → MaxPool
  → Conv2D(64) → [BN] → ReLU → MaxPool
  → Conv2D(128) → [BN] → ReLU → MaxPool
  → GlobalAveragePooling2D
  → Dense(128, ReLU) → [Dropout 0.3]
  → Dense(4, Softmax)
```

| Model | Augmentation | BatchNorm | Dropout | Learning rate | Purpose |
|---|---|---|---|---|---|
| V1 | ✗ | ✗ | ✗ | 0.001 | Baseline reference |
| V2 | ✓ | ✗ | ✗ | 0.001 | Effect of augmentation |
| V3 | ✓ | ✓ | ✓ | 0.001 | Stability + regularization |
| V4 | ✓ | ✓ | ✓ | **0.0003** | Smoother convergence |
| V5 | ✓ | ✓ | ✗ | 0.001 | Dropout ablation |

Augmentation consists of horizontal/vertical flips, ±5% rotation, ±8% zoom, and ±10% contrast.

---

## Interpretability and Error Analysis

**Grad-CAM.** A logits-based Grad-CAM implementation extracts the final convolutional feature
map, computes gradients of the target-class logit, and produces a normalized heatmap resized to
224 × 224. Prediction consistency between the feature-extraction model and the full model was
asserted to within `1e-6` before trusting the heatmaps. Visualizations are produced for:

- All misclassified test images
- One correctly classified example per class

**Error analysis.** Only 4 of 303 test images were misclassified, and all involved Blast as
either the true or predicted class:

| Actual → Predicted | Model confidence in predicted class |
|---|---|
| Bacterialblight → Blast | 0.6557 |
| Blast → Brownspot | 0.7265 |
| Blast → Bacterialblight | 0.8720 |
| Brownspot → Blast | 0.5176 |

The Blast class is the main source of confusion, consistent with its lower per-class F1 (0.97)
and with visible symptom overlap between Blast and Brownspot lesions.

---

## Repository Contents

```
.
├── Custom_CNN_submission.ipynb   # Full end-to-end notebook (214 cells, 17 sections)
└── README.md                     # This file
```

The notebook stores its own narrative documentation: every code cell is preceded by a markdown
cell describing its purpose and result.

---

## How to Run

The notebook is designed for **Google Colab with a GPU runtime** (T4 or better).

1. Open `Custom_CNN_submission.ipynb` in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run cells top to bottom. The first cells install dependencies and download the dataset
   automatically via `kagglehub`.
4. When prompted, authorise Google Drive access — split manifests and model checkpoints are
   written to `/content/drive/MyDrive/Rice_CNN_Project/`.
5. The final section opens an upload widget; choose a rice leaf image to get a prediction with
   per-class probabilities.

**Approximate runtime:** dataset download plus full training of all variants takes roughly
1–2 hours on a T4. If you only need the final model, you can skip straight to the
held-out test evaluation after restoring the saved `rice_disease_final_cnn.keras` checkpoint.

**Important:** all paths in the notebook are Colab-specific (`/content/...`). To run locally,
replace them with paths on your own machine.

---

## Requirements

```
tensorflow >= 2.15
numpy
pandas
matplotlib
seaborn
scikit-learn
scikit-image
scipy
pillow
kagglehub
imagehash
```

Install with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scikit-image scipy pillow kagglehub imagehash
```

A CUDA-capable GPU is strongly recommended but not required for inference.

---

## Output Artifacts

Saved to `/content/rice_cnn_models/corrected/` (and `/content/rice_cnn_models/` for the
pre-correction experiments):

| File | Description |
|---|---|
| `rice_disease_final_cnn.keras` | Frozen final CNN V4 checkpoint |
| `final_model_metadata.json` | Class mapping, input shape, metrics, preprocessing spec |
| `cnn_v{1..5}_clean_best.keras` | Best checkpoint per corrected-split variant |
| `cnn_v{1..5}_clean_history.csv` | Per-epoch training history |
| `final_cnn_comparison.csv` / `.png` | Cross-model comparison table and chart |
| `final_test_confusion_matrix.png` | Raw and row-normalized test confusion matrices |
| `misclassified_test_images.png` | All four misclassified test images with confidences |
| `verified_logits_gradcam.png` | Grad-CAM for misclassifications |
| `gradcam_correct_predictions.png` | Grad-CAM for correct predictions |

---

## Limitations and Notes

- **Dataset size.** After aggressive deduplication the dataset drops from 5,932 to 2,138 images,
  leaving only 303 test images. Four errors therefore move accuracy by ~1.3 percentage points.
- **Class confusion.** Blast is the weakest class and accounts for every test error, in both
  directions. More Blast/Brownspot examples or targeted augmentation would likely help.
- **Scope.** This is a four-class academic dataset; the model has not been validated on field
  images from other regions, cameras, or lighting conditions.
- **No transfer learning.** All models are trained from scratch by design, so absolute accuracy
  is lower than what a pretrained backbone (e.g. EfficientNet) would likely achieve.
- **The notebook's own conclusion** emphasises that the headline result rests on the corrected
  split and held-out evaluation, not on the inflated pre-audit validation score.

---

## Acknowledgements

- Dataset: *Rice Leaf Disease Image* by Nirmal Sankalana, hosted on Kaggle.
- Built with TensorFlow/Keras, scikit-learn, scikit-image, and imagehash.
