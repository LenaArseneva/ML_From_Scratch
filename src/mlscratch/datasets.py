from __future__ import annotations
import numpy as np

def make_binary_classification(
    n_samples: int = 600,
    n_features: int = 2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple synthetic dataset: two Gaussian blobs.
    Returns X (n_samples, n_features), y (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0

    mean0 = np.zeros(n_features)
    mean1 = np.ones(n_features) * 1.5
    cov = np.eye(n_features)

    X0 = rng.multivariate_normal(mean0, cov, size=n0)
    X1 = rng.multivariate_normal(mean1, cov, size=n1)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]

