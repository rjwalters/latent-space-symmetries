"""Tests for core permutation group utilities."""

from src.datasets.permutations import Permutation, PermutationGroup


class TestPermutation:
    def test_identity(self):
        e = Permutation.identity(3)
        assert e.mapping == (0, 1, 2)
        assert e.is_identity
        assert e.is_even

    def test_transposition(self):
        s = Permutation.transposition(3, 0, 1)
        assert s.mapping == (1, 0, 2)
        assert not s.is_even
        assert s.parity == 1

    def test_transposition_is_involution(self):
        s = Permutation.transposition(3, 0, 1)
        assert (s * s).is_identity

    def test_3_cycle(self):
        r = Permutation.cycle(3, 0, 1, 2)
        assert r.mapping == (1, 2, 0)
        assert r.is_even

    def test_3_cycle_cubed_is_identity(self):
        r = Permutation.cycle(3, 0, 1, 2)
        assert (r * r * r).is_identity

    def test_composition(self):
        s = Permutation.transposition(3, 0, 1)
        r = Permutation.cycle(3, 0, 1, 2)
        # srs should equal r^{-1} = r^2
        srs = s * r * s
        r_inv = r * r
        assert srs.mapping == r_inv.mapping

    def test_inverse(self):
        r = Permutation.cycle(3, 0, 1, 2)
        r_inv = r.inverse()
        assert (r * r_inv).is_identity
        assert (r_inv * r).is_identity

    def test_apply_to_list(self):
        s = Permutation.transposition(3, 0, 1)
        result = s.apply_to_list(["A", "B", "C"])
        assert result == ["B", "A", "C"]

    def test_cycle_notation(self):
        r = Permutation.cycle(3, 0, 1, 2)
        cycles = r.cycle_notation()
        assert len(cycles) == 1
        assert set(cycles[0]) == {0, 1, 2}


class TestPermutationGroup:
    def test_s2_size(self):
        g = PermutationGroup(2)
        assert len(g.elements) == 2

    def test_s3_size(self):
        g = PermutationGroup(3)
        assert len(g.elements) == 6

    def test_s3_parity_split(self):
        g = PermutationGroup(3)
        assert len(g.even_elements) == 3  # A_3
        assert len(g.odd_elements) == 3

    def test_s4_size(self):
        g = PermutationGroup(4)
        assert len(g.elements) == 24

    def test_s3_transpositions(self):
        g = PermutationGroup(3)
        assert len(g.transpositions) == 3

    def test_closure(self):
        """Verify S_3 is closed under composition."""
        g = PermutationGroup(3)
        elements_set = set(g.elements)
        for a in g.elements:
            for b in g.elements:
                assert a * b in elements_set

    def test_generators_generate_s3(self):
        g = PermutationGroup(3)
        gens = g.generators()
        s, r = gens["s"], gens["r"]
        # s and r should generate all of S_3
        generated = set()
        frontier = {Permutation.identity(3)}
        while frontier:
            current = frontier.pop()
            if current in generated:
                continue
            generated.add(current)
            frontier.add(current * s)
            frontier.add(current * r)
            frontier.add(s * current)
            frontier.add(r * current)
        assert len(generated) == 6
