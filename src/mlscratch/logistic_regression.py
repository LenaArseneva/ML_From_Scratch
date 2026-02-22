from __future__ import annotations
import numpy as np

class LogisticRegression:
    def __init__(self, fit_intercept: bool = True, lr: float = 0.1, n_iter: int = 3000):
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.n_iter = n_iter
        self.coef_: np.ndarray | None = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.c_[np.ones((X.shape[0], 1)), X]

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # stable sigmoid
        z = np.clip(z, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        Xb = self._add_intercept(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        w = np.zeros(Xb.shape[1], dtype=float)
        n = Xb.shape[0]

        for _ in range(self.n_iter):
            p = self._sigmoid(Xb @ w)
            grad = (1.0 / n) * (Xb.T @ (p - y))
            w -= self.lr * grad

        self.coef_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        Xb = self._add_intercept(np.asarray(X, dtype=float))
        p1 = self._sigmoid(Xb @ self.coef_)
        return np.c_[1 - p1, p1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
