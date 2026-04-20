"""Fit linear permutation operators W_π from paired activation data.

Given activation vectors h(x) and h(π·x) from many prompt pairs,
fit W_π such that h(π·x) ≈ W_π h(x) via least-squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from scipy.linalg import lstsq


@dataclass
class OperatorFit:
    """Result of fitting a permutation operator."""

    W: np.ndarray  # (d, d) linear operator
    residual_norm: float  # Frobenius norm of residual
    relative_error: float  # residual / target norm
    cosine_similarity: float  # mean cosine sim between predicted and actual
    n_samples: int  # number of prompt pairs used
    test_relative_error: float | None = None  # held-out error
    test_cosine_similarity: float | None = None


def _compute_metrics(
    W: np.ndarray, X: np.ndarray, Y: np.ndarray
) -> tuple[float, float, float]:
    """Compute residual norm, relative error, and mean cosine similarity."""
    Y_pred = (W @ X.T).T
    residual_norm = np.linalg.norm(Y - Y_pred, "fro")
    target_norm = np.linalg.norm(Y, "fro")
    relative_error = residual_norm / (target_norm + 1e-10)

    dots = np.sum(Y * Y_pred, axis=1)
    norms_y = np.linalg.norm(Y, axis=1)
    norms_pred = np.linalg.norm(Y_pred, axis=1)
    cosines = dots / (norms_y * norms_pred + 1e-10)
    cosine_similarity = float(np.mean(cosines))

    return float(residual_norm), relative_error, cosine_similarity


def fit_operator(
    activations_original: torch.Tensor,
    activations_permuted: torch.Tensor,
    test_fraction: float = 0.2,
    ridge_alpha: float = 0.0,
) -> OperatorFit:
    """Fit a linear operator W such that activations_permuted ≈ W @ activations_original.

    Args:
        activations_original: (n_samples, d) tensor of activations from original prompts.
        activations_permuted: (n_samples, d) tensor of activations from permuted prompts.
        test_fraction: Fraction of samples to hold out for testing generalization.
            Set to 0 to use all samples for fitting (no test metrics).
        ridge_alpha: L2 regularization strength. Helps when n_samples < d.

    Returns:
        OperatorFit with the fitted operator and quality metrics.
    """
    X = activations_original.detach().cpu().numpy()  # (n, d)
    Y = activations_permuted.detach().cpu().numpy()  # (n, d)

    n = X.shape[0]
    d = X.shape[1]

    # Train/test split
    if test_fraction > 0 and n > 5:
        n_test = max(1, int(n * test_fraction))
        # Deterministic split: last n_test samples are test
        X_train, X_test = X[:-n_test], X[-n_test:]
        Y_train, Y_test = Y[:-n_test], Y[-n_test:]
    else:
        X_train, X_test = X, None
        Y_train, Y_test = None, None

    # Fit W via ridge regression: W = Y^T X^T (X X^T + αI)^{-1}
    # Equivalently solve (X^T X + αI) W^T = X^T Y
    if ridge_alpha > 0:
        XtX = X_train.T @ X_train + ridge_alpha * np.eye(d)
        XtY = X_train.T @ Y_train
        W_T = np.linalg.solve(XtX, XtY)
    else:
        W_T, _, _, _ = lstsq(X_train, Y_train)
    W = W_T.T  # (d, d)

    # Train metrics
    residual_norm, relative_error, cosine_similarity = _compute_metrics(W, X_train, Y_train)

    # Test metrics
    test_relative_error = None
    test_cosine_similarity = None
    if X_test is not None:
        _, test_relative_error, test_cosine_similarity = _compute_metrics(W, X_test, Y_test)

    return OperatorFit(
        W=W,
        residual_norm=residual_norm,
        relative_error=relative_error,
        cosine_similarity=cosine_similarity,
        n_samples=X_train.shape[0],
        test_relative_error=test_relative_error,
        test_cosine_similarity=test_cosine_similarity,
    )
