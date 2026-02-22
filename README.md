# ML From Scratch (NumPy)

A compact machine learning framework implemented from scratch using **NumPy**.
The project focuses on algorithmic understanding, numerical stability, and clean software design.

## Features
- Linear Regression (closed-form + gradient descent)
- Logistic Regression (gradient descent)
- k-Nearest Neighbors
- Train/test split
- k-fold cross-validation
- Metrics: MSE, Accuracy, F1, ROC-AUC

## Why this project
- **Data Science**: demonstrates statistical learning foundations, optimization, and model evaluation.
- **Computer Science**: demonstrates modular architecture, OOP-ready code structure, computational complexity awareness.

## Installation
Requires Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .

## Future direction

Implement logistic regression with batches, conjugate gradient and local curvature estimation. This should speed up logistic regression over big datasets.
