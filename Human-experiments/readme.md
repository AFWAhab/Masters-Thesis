# Gene Expression Prediction Experiments

This directory contains the code used to explore CNN-based architectures and embedding strategies for predicting gene expression levels using DNABERT-2 and related metadata. All scripts contribute to a range of experiments described in the master's thesis.

⚠️ **All required datasets** (e.g. `.h5` files, CpG/TF tables, etc.) are too large to upload to this repository. Download instructions and links are provided in the thesis.

---

## File Overview

### `attempt_1.py`
Basic fully connected model using DNABERT-1 embeddings and full 8-feature half-life metadata. It standardizes inputs and performs 10 training runs with test R² evaluation and CSV logging.

### `attempt_2.py`
An enhanced MLP with batch normalization, deeper head layers, and dropout. Similar input as `cnn_attempt_1.py` but designed for better performance through deeper modeling.

### `attempt_3.py`
Further extends the `cnn_attempt_2.py` setup by including **transcription factor (TF)** metadata in addition to half-life data. Adds TF vector features to the input pipeline and logs R² results.

### `attempt_4.py`
Predictor using DNABERT-2 embeddings for **CpG-only** experiments (excludes full metadata). Fixed 50-epoch training without early stopping.

### `cnn_attempt_convolutional.py`
Multi-kernel CNN (MKCNN) with attention pooling, SWA (Stochastic Weight Averaging), and dilated convolutions. Uses DNABERT-2 embeddings and integrates half-life, TF, and CpG metadata.

### `cnn_attempt_convolutional_v2.py`
Variant of the above script that supports **dual DNABERT embeddings (1 & 2)** concatenated together. Optimized for dtype safety and uses a slightly deeper convolutional branch.

### `dnabert_2_embeddings.py`
Script for generating DNABERT-2 embeddings from raw promoter sequences in `.h5` files. Extracts tokenized windows and stores the result in a compressed `.h5` file for downstream use.

### `feature_engineering.py`
Performs brute-force subset testing of the 8 half-life features in combination with TF vectors. Trains a simple MLP to scan all possible feature masks and saves results to CSV.

### `hyperparam_optimization.py`
Performs Optuna-based hyperparameter search over MKCNN architecture parameters (e.g., kernel sizes, dropout rates, learning rates). Uses dual DNABERT embeddings, TF and CpG metadata.

---

## Notes on Data

All required `.h5` files (train/valid/test), embedding files, CpG tables, and transcription factor datasets must be downloaded separately. This is described in the masters thesis, but the data comes primarily from Xpresso & DeepLNCLoc.

---

## Requirements

- Python 3.10+
- PyTorch ≥ 1.13
- NumPy, Pandas, HDF5 (h5py)
- scikit-learn, SciPy
- `transformers` (for DNABERT-2)
- `openpyxl` (for TF Excel parsing)
- `biomart` Python client

CUDA acceleration is strongly recommended for embedding generation and model training.

This project was developed on:
- OS: Windows 10
- GPU: NVIDIA RTX 2080 (8GB VRAM)
- RAM: 16 GB DDR4 3200 MHZ
- CPU: I9 9900K