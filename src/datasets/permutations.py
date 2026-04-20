"""Core permutation group utilities.

Provides Permutation and PermutationGroup classes for S_n,
with support for composition, inversion, parity classification,
and enumeration of group elements, transpositions, and cycles.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence


@dataclass(frozen=True)
class Permutation:
    """A permutation of {0, 1, ..., n-1} stored as a mapping tuple."""

    mapping: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.mapping)

    def __call__(self, i: int) -> int:
        return self.mapping[i]

    def __mul__(self, other: Permutation) -> Permutation:
        """Compose: (self * other)(i) = self(other(i))."""
        assert self.n == other.n
        return Permutation(tuple(self.mapping[other.mapping[i]] for i in range(self.n)))

    def inverse(self) -> Permutation:
        inv = [0] * self.n
        for i, j in enumerate(self.mapping):
            inv[j] = i
        return Permutation(tuple(inv))

    @cached_property
    def parity(self) -> int:
        """Return 0 for even, 1 for odd."""
        visited = [False] * self.n
        cycle_count = 0
        for i in range(self.n):
            if not visited[i]:
                cycle_count += 1
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = self.mapping[j]
        return (self.n - cycle_count) % 2

    @property
    def is_even(self) -> bool:
        return self.parity == 0

    @property
    def is_identity(self) -> bool:
        return all(self.mapping[i] == i for i in range(self.n))

    def cycle_notation(self) -> list[tuple[int, ...]]:
        """Return cycle decomposition, omitting fixed points."""
        visited = [False] * self.n
        cycles = []
        for i in range(self.n):
            if visited[i] or self.mapping[i] == i:
                visited[i] = True
                continue
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = self.mapping[j]
            if len(cycle) > 1:
                cycles.append(tuple(cycle))
        return cycles

    def apply_to_list(self, items: Sequence) -> list:
        """Permute a list: result[i] = items[π(i)]."""
        return [items[self.mapping[i]] for i in range(self.n)]

    def __repr__(self) -> str:
        cycles = self.cycle_notation()
        if not cycles:
            return "e"
        return "".join(str(c) for c in cycles)

    @staticmethod
    def identity(n: int) -> Permutation:
        return Permutation(tuple(range(n)))

    @staticmethod
    def transposition(n: int, i: int, j: int) -> Permutation:
        """Swap elements i and j."""
        mapping = list(range(n))
        mapping[i], mapping[j] = mapping[j], mapping[i]
        return Permutation(tuple(mapping))

    @staticmethod
    def cycle(n: int, *elements: int) -> Permutation:
        """Create a cycle (e0 e1 e2 ... ek) -> e0->e1, e1->e2, ..., ek->e0."""
        mapping = list(range(n))
        for idx in range(len(elements)):
            mapping[elements[idx]] = elements[(idx + 1) % len(elements)]
        return Permutation(tuple(mapping))


class PermutationGroup:
    """The symmetric group S_n, with utilities for subgroup analysis."""

    def __init__(self, n: int):
        self.n = n

    @cached_property
    def elements(self) -> list[Permutation]:
        return [Permutation(p) for p in itertools.permutations(range(self.n))]

    @cached_property
    def even_elements(self) -> list[Permutation]:
        return [p for p in self.elements if p.is_even]

    @cached_property
    def odd_elements(self) -> list[Permutation]:
        return [p for p in self.elements if not p.is_even]

    @cached_property
    def transpositions(self) -> list[Permutation]:
        return [
            Permutation.transposition(self.n, i, j)
            for i in range(self.n)
            for j in range(i + 1, self.n)
        ]

    @cached_property
    def three_cycles(self) -> list[Permutation]:
        return [
            Permutation.cycle(self.n, *combo)
            for combo in itertools.combinations(range(self.n), 3)
            for _ in [None]  # include both orientations
            for p in [
                Permutation.cycle(self.n, combo[0], combo[1], combo[2]),
                Permutation.cycle(self.n, combo[0], combo[2], combo[1]),
            ]
        ]

    def generators(self) -> dict[str, Permutation]:
        """Return standard generators: adjacent transposition s and n-cycle r."""
        s = Permutation.transposition(self.n, 0, 1)
        r = Permutation.cycle(self.n, *range(self.n))
        return {"s": s, "r": r}

    def cayley_table(self) -> dict[tuple[Permutation, Permutation], Permutation]:
        return {(a, b): a * b for a in self.elements for b in self.elements}
