"""Test whether fitted operators satisfy group relations.

For S_n, the key relations to test:
- Composition: W_{π1} W_{π2} ≈ W_{π1∘π2}
- Identity: W_e ≈ I
- Inverse: W_{π} W_{π^{-1}} ≈ I
- S_3 presentation: s²=e, r³=e, srs=r⁻¹
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RelationError:
    """Error measurement for a group relation."""

    name: str
    frobenius_error: float  # ||LHS - RHS||_F
    relative_error: float  # ||LHS - RHS||_F / ||RHS||_F
    operator_norm_error: float  # ||LHS - RHS||_2 (largest singular value of diff)


def _relation_error(name: str, lhs: np.ndarray, rhs: np.ndarray) -> RelationError:
    diff = lhs - rhs
    frob = float(np.linalg.norm(diff, "fro"))
    rhs_norm = float(np.linalg.norm(rhs, "fro"))
    op_norm = float(np.linalg.norm(diff, 2))
    return RelationError(
        name=name,
        frobenius_error=frob,
        relative_error=frob / (rhs_norm + 1e-10),
        operator_norm_error=op_norm,
    )


def composition_error(
    W_pi1: np.ndarray,
    W_pi2: np.ndarray,
    W_pi1_pi2: np.ndarray,
) -> RelationError:
    """Test W_{π1} W_{π2} ≈ W_{π1∘π2}."""
    return _relation_error("composition", W_pi1 @ W_pi2, W_pi1_pi2)


def identity_error(W_e: np.ndarray) -> RelationError:
    """Test W_e ≈ I."""
    I = np.eye(W_e.shape[0])
    return _relation_error("identity", W_e, I)


def inverse_error(W_pi: np.ndarray, W_pi_inv: np.ndarray) -> RelationError:
    """Test W_π W_{π^{-1}} ≈ I."""
    I = np.eye(W_pi.shape[0])
    return _relation_error("inverse", W_pi @ W_pi_inv, I)


def s3_relation_errors(
    W_s: np.ndarray,
    W_r: np.ndarray,
) -> dict[str, RelationError]:
    """Test the S_3 presentation relations: s²=e, r³=e, srs=r⁻¹.

    Args:
        W_s: Operator for the transposition generator s = (0 1).
        W_r: Operator for the 3-cycle generator r = (0 1 2).

    Returns:
        Dict of relation name -> RelationError.
    """
    d = W_s.shape[0]
    I = np.eye(d)

    results = {}
    results["s^2 = e"] = _relation_error("s^2 = e", W_s @ W_s, I)
    results["r^3 = e"] = _relation_error("r^3 = e", W_r @ W_r @ W_r, I)

    # srs = r^{-1}: we test srs r = e equivalently, or directly compare srs to r^{-1}
    # r^{-1} = r^2 for a 3-cycle
    W_r_inv = W_r @ W_r
    srs = W_s @ W_r @ W_s
    results["srs = r^{-1}"] = _relation_error("srs = r^{-1}", srs, W_r_inv)

    return results


def parity_analysis(
    operators: dict,
    permutations: list,
) -> dict[str, float]:
    """Compare fit quality for even vs odd permutations.

    Args:
        operators: Dict mapping Permutation -> OperatorFit.
        permutations: List of Permutation objects with .parity attribute.

    Returns:
        Dict with mean relative errors for even and odd permutations.
    """
    even_errors = []
    odd_errors = []
    for perm in permutations:
        if perm.is_identity:
            continue
        fit = operators.get(perm)
        if fit is None:
            continue
        if perm.is_even:
            even_errors.append(fit.relative_error)
        else:
            odd_errors.append(fit.relative_error)

    return {
        "mean_even_error": float(np.mean(even_errors)) if even_errors else float("nan"),
        "mean_odd_error": float(np.mean(odd_errors)) if odd_errors else float("nan"),
        "even_count": len(even_errors),
        "odd_count": len(odd_errors),
    }
