"""Probe candidate prompt templates to find ones where the model
reliably completes with a permutation-sensitive entity name."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation.capture import load_model, DEFAULT_MODEL

CANDIDATES = [
    # S_2: who comes first
    "{e0} and {e1} went to the store. {e0}",
    "{e0} and {e1} are friends. {e0} is older than {e1}. The older one is",
    "Between {e0} and {e1}, the first one mentioned is",
    "{e0} and {e1} entered the room. The first to enter was",
    "{e0} beat {e1} in a race. The winner was",
    "{e0} is taller than {e1}. The tallest person is",

    # S_3: who comes first / role tracking
    "{e0}, {e1}, and {e2} walked into a bar. {e0}",
    "{e0}, {e1}, and {e2} are in a race. {e0} finished first, {e1} finished second, and {e2} finished last. The winner is",
    "{e0} gave a gift to {e1}. The giver was",
    "{e0} gave a gift to {e1}. The recipient was",
    "{e0} told {e1} a secret about {e2}. The person who told the secret was",
    "{e0} told {e1} a secret about {e2}. The person the secret was about was",
    "Q: {e0}, {e1}, and {e2} are standing in a line. Who is first in line?\nA:",
    "{e0} is the teacher, {e1} is the student, and {e2} is the principal. The teacher is",
    "Names: {e0}, {e1}, {e2}. The first name in the list is",
]

ENTITIES_2 = ["Alice", "Bob"]
ENTITIES_3 = ["Alice", "Bob", "Carol"]


def probe(model, template: str, entity_sets: list[list[str]], k: int = 5):
    """Test a template with identity and one swap."""
    n = len(entity_sets[0])
    print(f"\n{'─'*70}")
    print(f"Template: {template}")

    for entities in entity_sets:
        replacements = {f"e{i}": name for i, name in enumerate(entities)}
        prompt = template.format(**replacements)
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        topk = torch.topk(probs, k)
        completions = []
        for idx, prob in zip(topk.indices, topk.values):
            tok = model.to_string(idx.unsqueeze(0))
            completions.append(f"{tok!r}({prob:.3f})")
        label = "→".join(entities)
        print(f"  {label:>25s}:  {', '.join(completions)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = load_model(model_name=args.model, device=args.device)
    print("Loaded.\n")

    for template in CANDIDATES:
        n_slots = template.count("{e")  # rough check
        if "{e2}" in template:
            # S_3: test identity, (0,1) swap, (0,1,2) cycle
            entity_sets = [
                ["Alice", "Bob", "Carol"],
                ["Bob", "Alice", "Carol"],  # swap 0,1
                ["Bob", "Carol", "Alice"],  # cycle (0,1,2)
            ]
        else:
            # S_2: test identity and swap
            entity_sets = [
                ["Alice", "Bob"],
                ["Bob", "Alice"],
            ]
        probe(model, template, entity_sets)


if __name__ == "__main__":
    main()
