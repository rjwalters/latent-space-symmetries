"""Cache permutation-related activations to disk for SAE analysis.

Runs the model once and saves last-token residual stream activations
for all permutation prompts. Subsequent scripts (SAE analysis, etc.)
can load these without re-running the model.

Usage:
    python scripts/cache_activations.py --layers 20 21 22
    python scripts/cache_activations.py --layers 21 --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation.capture import load_model, cache_activations, DEFAULT_MODEL
from src.datasets.permutations import Permutation
from src.datasets.prompts import generate_bulk_samples


def main():
    parser = argparse.ArgumentParser(description="Cache activations to disk")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 21, 22])
    parser.add_argument("--output", default="data/activations/perm_cache.npz")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    print(f"Loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    samples = generate_bulk_samples(3)

    # Collect all unique prompts
    all_prompts = set()
    for s in samples:
        all_prompts.add(s["base_prompt"])
        all_prompts.add(s["permuted_prompt"])
    all_prompts = sorted(all_prompts)  # deterministic order

    print(f"Caching {len(all_prompts)} prompts across layers {args.layers}...")

    # Cache: prompt -> {layer -> activation}
    activations = {}
    for prompt in tqdm(all_prompts, desc="Caching"):
        cached = cache_activations(
            model, prompt, hook_points=["resid_post"], layers=args.layers
        )
        resid = cached.activations["resid_post"]  # (n_layers, seq_len, d_model)
        for i, layer in enumerate(args.layers):
            key = f"layer{layer}/{prompt}"
            activations[key] = resid[i, -1, :].detach().cpu().numpy()

    # Save metadata
    sample_meta = []
    for s in samples:
        sample_meta.append({
            "base_prompt": s["base_prompt"],
            "permuted_prompt": s["permuted_prompt"],
            "permutation": str(s["permutation"]),
            "perm_tuple": s["permutation"].mapping,
            "parity": s["permutation"].parity,
        })

    # Save as npz
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    save_dict["layers"] = np.array(args.layers)
    save_dict["prompts"] = np.array(all_prompts)
    save_dict["n_samples"] = np.array(len(samples))

    # Store sample metadata as structured arrays
    save_dict["base_prompts"] = np.array([s["base_prompt"] for s in sample_meta])
    save_dict["permuted_prompts"] = np.array([s["permuted_prompt"] for s in sample_meta])
    save_dict["perm_strs"] = np.array([s["permutation"] for s in sample_meta])
    save_dict["perm_tuples"] = np.array([s["perm_tuple"] for s in sample_meta])
    save_dict["parities"] = np.array([s["parity"] for s in sample_meta])

    # Store activations as layer_N/prompt_idx arrays
    prompt_to_idx = {p: i for i, p in enumerate(all_prompts)}
    for layer in args.layers:
        layer_acts = np.stack([
            activations[f"layer{layer}/{p}"] for p in all_prompts
        ])  # (n_prompts, d_model)
        save_dict[f"acts_layer{layer}"] = layer_acts

    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(all_prompts)} prompts × {len(args.layers)} layers × {model.cfg.d_model} dims")


if __name__ == "__main__":
    main()
