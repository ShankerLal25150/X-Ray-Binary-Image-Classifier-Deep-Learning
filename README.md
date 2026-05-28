# Deep Learning Chest X-Ray Classifier

### Comparative Study: Custom CNN vs. Transfer Learning for Pneumonia Detection

A deep learning project focused on binary classification of chest X-ray images (**Normal vs. Pneumonia**) using two different approaches:

* A **custom Convolutional Neural Network (CNN)** built from scratch
* A **Transfer Learning pipeline using EfficientNetB0**

The project emphasizes not only model performance, but also **engineering realism**, including:

* prevention of **data leakage**
* handling of **hardware limitations**
* medically appropriate preprocessing choices
* practical deployment considerations for constrained environments

---

# Overview

Most beginner medical imaging repositories unintentionally introduce methodological flaws such as:

* validating directly on the test set
* unrealistic augmentation policies
* ignoring hardware execution limitations
* presenting transfer learning as universally plug-and-play

This repository intentionally avoids those mistakes.

The goal was to build a more technically honest comparison between:

1. a lightweight custom CNN that can reliably train on CPU hardware
2. a modern transfer learning architecture that exposes real-world execution constraints

---

# Key Features

## Data Integrity First

A strict dataset separation strategy was enforced:

* **80% Training**
* **20% Validation**
* Completely isolated **Test Set**

This prevents **data leakage**, which is extremely common in introductory ML projects where validation performance becomes artificially inflated.

---

## Medical Context Awareness

Medical image augmentation was designed carefully.

### Disabled Augmentations

* Horizontal flipping was intentionally disabled to preserve anatomical correctness.
* Random transformations were kept conservative to avoid unrealistic pathology distortion.

Why this matters:

* Flipping chest X-rays can create anatomically impossible conditions such as **situs inversus**
* Aggressive augmentations can introduce medically invalid features

---

# Project Structure

```bash
├── Solution_1.py   # EfficientNetB0 Transfer Learning
├── Solution_2.py   # Custom CNN Baseline
├── dataset/
├── models/
├── results/
└── README.md
```

---

# Solution 2 — Production Baseline (Custom CNN)

## Why This Became the Primary Model

The custom CNN became the repository’s primary working solution because it trained reliably under limited hardware conditions.

Unlike the transfer learning implementation, it:

* converged consistently
* generalized properly
* executed without numerical instability on CPU-only systems

---

## Architecture

### Convolution Blocks

Progressive feature extraction:

```text
32 → 64 → 128 → 128
```

### Pooling Strategy

Instead of large dense layers, the network uses:

```python
GlobalAveragePooling2D()
```

Benefits:

* lower parameter count
* reduced overfitting
* improved generalization

---

## Regularization

Heavy regularization was necessary due to:

* relatively small medical dataset
* high overfitting risk

Implemented techniques:

* `Dropout(0.5)`
* Data augmentation
* Early stopping

---

## Outcome

The custom CNN:

* trained successfully
* achieved stable validation performance
* demonstrated proper generalization behavior
* validated the integrity of the preprocessing pipeline

This model serves as the repository’s **fully functional reference implementation**.

---

# Solution 1 — Transfer Learning with EfficientNetB0

## Objective

The goal was to leverage:

* ImageNet pre-trained weights
* EfficientNetB0 feature extraction capabilities
* fine-tuning for chest X-ray classification

This implementation uses:

* Keras Functional API
* frozen EfficientNet backbone
* custom classification head

---

## Important Engineering Detail

To avoid catastrophic forgetting during fine-tuning:

```python
training=False
```

was explicitly enforced for Batch Normalization layers inside EfficientNet.

This is considered best practice when adapting pre-trained CNN architectures to smaller domain-specific datasets.

---

# The Real Problem: Hardware Constraints

This repository intentionally documents a major practical issue that many tutorials ignore.

## Environment

* Native Windows environment
* CPU-only execution
* TensorFlow versions > 2.10
* No CUDA acceleration

---

## Observed Failure

Despite correct implementation:

* training appeared to execute normally
* loss calculations became numerically unstable internally
* gradients effectively vanished
* validation accuracy became permanently stuck at:

```text
50%
```

This occurred consistently during EfficientNet execution.

---

## Root Cause

The issue was not architectural logic.

The bottleneck was:

* CPU-only execution
* TensorFlow backend instability on native Windows
* computational overhead from EfficientNet graph operations

In practice, complex transfer learning pipelines are heavily dependent on:

* CUDA-enabled GPUs
* Linux-based deep learning environments
* optimized tensor computation libraries

---

# Key Takeaway

This repository demonstrates an important engineering reality:

> A theoretically correct deep learning architecture can still fail due to infrastructure limitations.

Transfer learning is not automatically superior if:

* the environment cannot execute it reliably
* numerical stability collapses during training
* hardware constraints dominate experimentation

Under constrained hardware:

* simpler architectures often outperform complex models in practice
* reproducibility matters more than theoretical sophistication

---

# Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

---

# Future Improvements

Potential extensions include:

* Linux + CUDA migration
* Mixed precision training
* Grad-CAM visualization
* Multi-class disease classification
* Hyperparameter optimization
* Cross-validation benchmarking
* Model deployment with Flask/FastAPI

---

# How to Run

## Install Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## Run Custom CNN

```bash
python Solution_2.py
```

---

## Run Transfer Learning Experiment

```bash
python Solution_1.py
```

---

# Disclaimer

This project is intended for:

* educational purposes
* engineering experimentation
* comparative deep learning analysis

It is **not** a clinical diagnostic system and should not be used for medical decision-making.


