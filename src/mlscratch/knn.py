from __future__ import annotations
import numpy as np

class KNNClassifier:
    def __init__(self, k: int = 5):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_ is None or self.y_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)

        # compute squared distances (vectorized)
        # dist^2 = ||x||^2 + ||xi||^2 - 2 x·xi
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        train_norm = np.sum(self.X_**2, axis=1)
        d2 = X_norm + train_norm - 2 * (X @ self.X_.T)

        nn = np.argpartition(d2, self.k - 1, axis=1)[:, :self.k]
        votes = self.y_[nn]
        return (np.mean(votes, axis=1) >= 0.5).astype(int)
