"""Feature-space analysis utilities for SAE-encoded activations.

Provides tools for:
- Identifying permutation-sensitive features
- Computing feature-space operators
- Testing group structure in sparse feature basis
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq


@dataclass
class FeatureStats:
    """Statistics about SAE features in relation to permutation structure."""
    d_sae: int
    n_active: int  # features with any activation
    n_perm_sensitive: int  # features strongly affected by permutations
    sensitivity: np.ndarray  # (d_sae,) mean |Δ| per feature
    sensitive_indices: np.ndarray  # indices of most sensitive features


def compute_feature_sensitivity(
    sae_features: dict[str, np.ndarray],
    samples: list[dict],
    threshold_percentile: float = 90.0,
) -> FeatureStats:
    """Identify which SAE features are most affected by permutations.

    Computes mean |feature_diff| across all permutation samples.
    """
    diffs = []
    for s in samples:
        base = sae_features[s["base_prompt"]]
        perm = sae_features[s["permuted_prompt"]]
        diffs.append(perm - base)
    diffs = np.stack(diffs)

    sensitivity = np.mean(np.abs(diffs), axis=0)
    n_active = int(np.sum(sensitivity > 0))

    # Threshold on active features only
    active_sensitivities = sensitivity[sensitivity > 0]
    if len(active_sensitivities) > 0:
        threshold = np.percentile(active_sensitivities, threshold_percentile)
        sensitive_idx = np.where(sensitivity > threshold)[0]
    else:
        sensitive_idx = np.array([], dtype=int)

    return FeatureStats(
        d_sae=len(sensitivity),
        n_active=n_active,
        n_perm_sensitive=len(sensitive_idx),
        sensitivity=sensitivity,
        sensitive_indices=sensitive_idx,
    )


def fit_feature_operator(
    sae_features: dict[str, np.ndarray],
    samples: list[dict],
    feature_indices: np.ndarray | None = None,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, float, float]:
    """Fit a linear operator in SAE feature space.

    Args:
        sae_features: prompt -> feature activations
        samples: list of dicts with base_prompt, permuted_prompt
        feature_indices: which features to use (None = all)
        test_fraction: held-out fraction for evaluation

    Returns:
        (W, test_error, test_cosine)
    """
    if feature_indices is not None:
        X = np.stack([sae_features[s["base_prompt"]][feature_indices] for s in samples])
        Y = np.stack([sae_features[s["permuted_prompt"]][feature_indices] for s in samples])
    else:
        X = np.stack([sae_features[s["base_prompt"]] for s in samples])
        Y = np.stack([sae_features[s["permuted_prompt"]] for s in samples])

    n = X.shape[0]
    n_test = max(1, int(n * test_fraction))
    X_train, X_test = X[:-n_test], X[-n_test:]
    Y_train, Y_test = Y[:-n_test], Y[-n_test:]

    W_T, _, _, _ = lstsq(X_train, Y_train)
    W = W_T.T

    # Test metrics
    Y_pred = (W @ X_test.T).T
    test_err = float(np.linalg.norm(Y_test - Y_pred, "fro") / (np.linalg.norm(Y_test, "fro") + 1e-10))

    dots = np.sum(Y_test * Y_pred, axis=1)
    norms_y = np.linalg.norm(Y_test, axis=1)
    norms_p = np.linalg.norm(Y_pred, axis=1)
    cos = float(np.mean(dots / (norms_y * norms_p + 1e-10)))

    return W, test_err, cos
