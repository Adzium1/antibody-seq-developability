# Antibody Sequence Representations and Simple Developability Proxies

This repository contains a small, self-contained project on learning simple representations of antibody sequences from repertoire data and predicting basic classification and developability-related properties.

The goal is not to build a new foundation model, but to demonstrate end-to-end competence on a realistic antibody dataset:
data curation from Observed Antibody Space (OAS), baseline models, and a lightweight sequence model, with a clear link to developability concepts.

---

## 1. Scientific motivation

Antibody discovery pipelines increasingly rely on large-scale sequence data and machine learning.  
Two key questions are:

1. Can we learn useful sequence representations directly from repertoire data (e.g. heavy-chain variable regions or CDR-H3)?
2. Do these representations capture simple biophysical properties that are proxies for developability risk (e.g. CDR length, hydrophobicity, charge)?

This mini-project explores these questions on a constrained scope:

- focus on **human heavy chains** from OAS,
- simple **classification** and **regression** tasks,
- comparison of **classical baselines** and a **small 1D CNN**.

The code and analysis are designed to be readable by a computational biology / ML-for-biologics audience (e.g. PhD applications).

---

## 2. Data

### 2.1 Source

The primary dataset is the **OAS Paired Sequence Data** released on Hugging Face:

- **HF dataset**: `bloyal/oas-paired-sequence-data`
- **Config used here**: `human` (paired VH/VL repertoires)

This dataset mirrors paired heavy- and light-chain sequences from the **Observed Antibody Space (OAS)** resource, which aggregates and cleans NGS repertoires across many studies, with standardized metadata (species, isotype, chain, study, subject, etc.).

### 2.2 Filtering used in this project

For this project we restrict ourselves to:

- **Species**: human
- **Chain**: heavy chain only
- **Isotypes**: IgM and IgG

We construct a pandas DataFrame with the following schema:

- `sequence_heavy` – amino-acid sequence of the heavy-chain variable region (or CDR-H3, depending on the experiment)
- `isotype` – IgM vs IgG label
- `subject_id` – donor identifier (when available)
- `study_id` – study identifier (when available)

Splits are performed **by subject or study**, not by individual sequence, to limit data leakage from near-identical clones across train and test sets.

---

## 3. Tasks

We consider two main tasks:

### 3.1 Task A – Isotype classification (IgM vs IgG)

- **Input**: heavy-chain sequence, either:
  - full VH, or
  - CDR-H3 only (extracted from the dataset’s CDR annotations, when available).
- **Output**: binary label
  - IgM (typically enriched in naïve B cells)
  - IgG (typically enriched in antigen-experienced / class-switched B cells)

This task serves as a sanity check that sequence models can capture global differences between naïve-like and class-switched repertoires.

### 3.2 Task B – Simple developability proxy (regression or binarised)

We define a **toy developability score** based only on CDR-H3:

For each CDR-H3 sequence we compute:

- length of CDR-H3
- mean hydrophobicity (Kyte–Doolittle scale)
- approximate net charge at pH ≈ 7
  - +1 for K/R, +0.1 for H, −1 for D/E

These components are z-scored across the dataset and combined into a scalar:

\[
\text{DevScore} = z_{\text{length}} + |z_{\text{charge}}| + z_{\text{hydrophobicity}}
\]

We then formulate:

- either a **regression** task (predict DevScore),
- or a **binary classification** task (e.g. low-risk vs high-risk according to DevScore quantiles).

This proxy is intentionally simple but conceptually aligned with published developability guidelines that emphasize CDR length, surface hydrophobicity and charge patterns.

---

## 4. Methods

### 4.1 Baseline feature representations

We start from purely sequence-based features:

- **One-hot encoding** of amino acids with padding/truncation to a fixed length.
- **k-mer counts** (e.g. di- or tri-peptides), normalised by sequence length.
- **Simple physicochemical encodings**:
  - per-residue hydrophobicity and charge,
  - aggregated as means over the sequence.

These features are used with classical models:

- **Logistic regression** (L2-regularised, class_weight='balanced') for Task A.
- **Random forest regression** for Task B.

### 4.2 Sequence model: 1D CNN

We implement a small 1D convolutional network:

- integer encoding of amino acids with a learned embedding layer,
- several Conv1D + ReLU blocks with different kernel sizes,
- global max pooling over sequence length,
- a linear head for:
  - binary classification (Task A, using BCEWithLogitsLoss), or
  - regression (Task B, using MSELoss).

This architecture is deliberately lightweight but sufficient to test whether a sequence model yields:

- improved performance over baselines,
- better cross-study generalisation.

### 4.3 Training and evaluation

- Data split:
  - train / validation / test splits are defined at the **subject or study** level.
- Optimisation:
  - Adam optimizer, small number of epochs, early stopping on validation loss.
- Metrics:
  - Task A: accuracy, ROC–AUC, confusion matrix.
  - Task B: mean squared error (MSE), coefficient of determination (R²).