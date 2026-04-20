"""Tests for group relation testing utilities.

Uses exact permutation matrices to verify the test functions
correctly identify perfect group structure.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from src.datasets.permutations import Permutation, PermutationGroup
from src.operators.group_tests import (
    composition_error,
    identity_error,
    inverse_error,
    s3_relation_errors,
)


def permutation_matrix(perm: Permutation) -> np.ndarray:
    """Convert a Permutation to its standard permutation matrix."""
    n = perm.n
    P = np.zeros((n, n))
    for i in range(n):
        P[perm(i), i] = 1.0
    return P


class TestGroupRelationsExact:
    """With exact permutation matrices, all errors should be ~0."""

    def test_identity_error_zero(self):
        e = Permutation.identity(3)
        W_e = permutation_matrix(e)
        err = identity_error(W_e)
        assert err.frobenius_error < 1e-10

    def test_composition_error_zero(self):
        g = PermutationGroup(3)
        for a in g.elements:
            for b in g.elements:
                W_a = permutation_matrix(a)
                W_b = permutation_matrix(b)
                W_ab = permutation_matrix(a * b)
                err = composition_error(W_a, W_b, W_ab)
                assert err.frobenius_error < 1e-10, f"Failed for {a} * {b}"

    def test_inverse_error_zero(self):
        g = PermutationGroup(3)
        for p in g.elements:
            W = permutation_matrix(p)
            W_inv = permutation_matrix(p.inverse())
            err = inverse_error(W, W_inv)
            assert err.frobenius_error < 1e-10

    def test_s3_relations_zero(self):
        s = Permutation.transposition(3, 0, 1)
        r = Permutation.cycle(3, 0, 1, 2)
        errors = s3_relation_errors(permutation_matrix(s), permutation_matrix(r))
        for name, err in errors.items():
            assert err.frobenius_error < 1e-10, f"Failed: {name}"


class TestGroupRelationsApproximate:
    """With noisy operators, errors should be nonzero but bounded."""

    def test_noisy_operators_have_nonzero_error(self):
        s = Permutation.transposition(3, 0, 1)
        W_s = permutation_matrix(s) + np.random.randn(3, 3) * 0.1
        err = identity_error(W_s @ W_s)
        assert err.frobenius_error > 0.01
