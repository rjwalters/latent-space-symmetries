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


def fit_operator(
    activations_original: torch.Tensor,
    activations_permuted: torch.Tensor,
) -> OperatorFit:
    """Fit a linear operator W such that activations_permuted ≈ W @ activations_original.

    Args:
        activations_original: (n_samples, d) tensor of activations from original prompts.
        activations_permuted: (n_samples, d) tensor of activations from permuted prompts.

    Returns:
        OperatorFit with the fitted operator and quality metrics.
    """
    X = activations_original.detach().cpu().numpy()  # (n, d)
    Y = activations_permuted.detach().cpu().numpy()  # (n, d)

    # Solve Y^T = W @ X^T  =>  Y^T = W X^T  =>  W = Y^T X^T+ (pseudoinverse)
    # Equivalently solve X^T W^T = Y^T via lstsq
    W_T, residuals, rank, sv = lstsq(X, Y)
    W = W_T.T  # (d, d)

    # Compute quality metrics
    Y_pred = (W @ X.T).T
    residual_norm = np.linalg.norm(Y - Y_pred, "fro")
    target_norm = np.linalg.norm(Y, "fro")
    relative_error = residual_norm / (target_norm + 1e-10)

    # Mean cosine similarity
    dots = np.sum(Y * Y_pred, axis=1)
    norms_y = np.linalg.norm(Y, axis=1)
    norms_pred = np.linalg.norm(Y_pred, axis=1)
    cosines = dots / (norms_y * norms_pred + 1e-10)
    cosine_similarity = float(np.mean(cosines))

    return OperatorFit(
        W=W,
        residual_norm=float(residual_norm),
        relative_error=float(relative_error),
        cosine_similarity=cosine_similarity,
        n_samples=X.shape[0],
    )
