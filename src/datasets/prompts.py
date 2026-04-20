"""Symmetry-bearing prompt families for permutation experiments.

Each PromptFamily defines a template with named entity slots and a method
to generate all permuted variants with their associated Permutation objects.
Entity names are chosen for matched tokenization length to minimize artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.datasets.permutations import Permutation, PermutationGroup


# Entity name sets chosen for single-token behavior in common tokenizers
# and roughly matched pretraining frequency.
ENTITY_SETS = {
    2: ["Alice", "Bob"],
    3: ["Alice", "Bob", "Carol"],
    4: ["Alice", "Bob", "Carol", "Diana"],
}

# Abstract symbol sets (for controlling lexical asymmetry)
SYMBOL_SETS = {
    2: ["X", "Y"],
    3: ["X", "Y", "Z"],
    4: ["X", "Y", "Z", "W"],
}


@dataclass
class PromptFamily:
    """A template for generating permutation-paired prompts.

    Attributes:
        name: Identifier for this prompt family.
        template: String with {e0}, {e1}, ... placeholders for entities.
        n_entities: Number of exchangeable entities.
        entity_set: Which entity names to use (defaults to ENTITY_SETS[n]).
        expected_answer_fn: Given entity list, returns the expected completion.
            Used for behavioral validation. May be None if not applicable.
    """

    name: str
    template: str
    n_entities: int
    entity_set: list[str] | None = None
    expected_answer_fn: callable | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.entity_set is None:
            self.entity_set = ENTITY_SETS[self.n_entities]

    def render(self, entities: list[str]) -> str:
        """Fill the template with the given entity list."""
        replacements = {f"e{i}": name for i, name in enumerate(entities)}
        return self.template.format(**replacements)

    def base_prompt(self) -> str:
        return self.render(self.entity_set)

    def permuted_prompt(self, perm: Permutation) -> str:
        permuted_entities = perm.apply_to_list(self.entity_set)
        return self.render(permuted_entities)


def generate_permuted_prompts(
    family: PromptFamily,
) -> list[dict]:
    """Generate all permuted variants for a prompt family.

    Returns a list of dicts with keys:
        - permutation: the Permutation object
        - prompt: the rendered prompt string
        - entities: the entity list used
        - parity: 0 (even) or 1 (odd)
        - is_identity: bool
    """
    group = PermutationGroup(family.n_entities)
    results = []
    for perm in group.elements:
        entities = perm.apply_to_list(family.entity_set)
        results.append(
            {
                "permutation": perm,
                "prompt": family.render(entities),
                "entities": entities,
                "parity": perm.parity,
                "is_identity": perm.is_identity,
            }
        )
    return results


# --- Built-in prompt families ---

SET_LIST_S2 = PromptFamily(
    name="set_list_s2",
    template="{e0} and {e1} are standing in a room.",
    n_entities=2,
)

SET_LIST_S3 = PromptFamily(
    name="set_list_s3",
    template="{e0}, {e1}, and {e2} are standing in a room.",
    n_entities=3,
)

ROLE_ASSIGNMENT_S3 = PromptFamily(
    name="role_assignment_s3",
    template="{e0} gave {e1} the book. {e2} watched.",
    n_entities=3,
)

POINTER_S3 = PromptFamily(
    name="pointer_s3",
    template="Among {e0}, {e1}, and {e2}, the second person listed is",
    n_entities=3,
    expected_answer_fn=lambda entities: entities[1],
)

OBJECT_SET_S3 = PromptFamily(
    name="object_set_s3",
    template="The bag contains a {e0}, a {e1}, and a {e2}.",
    n_entities=3,
    entity_set=["red cube", "blue sphere", "green key"],
)

ALL_FAMILIES = [SET_LIST_S2, SET_LIST_S3, ROLE_ASSIGNMENT_S3, POINTER_S3, OBJECT_SET_S3]
