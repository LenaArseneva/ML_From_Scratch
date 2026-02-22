from ___future__ import annotations
import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept: bool = True, method: str = "closed_form",
                 lr: float = 0.05, n_iter: int = 2000):
        self.fit_intercept = fit_intercept
        self.method = method
        self.lr = lr
        self.n_iter = n_iter
        self.coef_: np.ndarray | None = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        Xb = self._add_intercept(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)

        if self.method == "closed_form":
            # ridge-like tiny regularization for stability
            reg = 1e-8 * np.eye(Xb.shape[1])
            self.coef_ = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
            return self

        if self.method == "gd":
            w = np.zeros(Xb.shape[1], dtype=float)
            n = Xb.shape[0]
            for _ in range(self.n_iter):
                pred = Xb @ w
                grad = (2.0 / n) * (Xb.T @ (pred - y))
                w -= self.lr * grad
            self.coef_ = w
            return self

        raise ValueError("method must be 'closed_form' or 'gd'")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        Xb = self._add_intercept(np.asarray(X, dtype=float))
        return Xb @ self.coef_
