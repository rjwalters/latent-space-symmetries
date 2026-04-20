"""Phase 4: Activation capture and linear alignment search.

For each permutation in S_n, collect (base, permuted) activation pairs across
many prompts (templates × entity sets), then fit a linear operator W_π at each
layer and measure fit quality.

Key questions:
- At which layers do permutation-induced activation differences become linearly predictable?
- Does alignment quality vary by layer depth (early/mid/late)?
- Is there a "sweet spot" layer range for operator fitting?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation.capture import load_model, cache_activations, DEFAULT_MODEL
from src.datasets.permutations import Permutation, PermutationGroup
from src.datasets.prompts import generate_bulk_samples
from src.operators.fitting import fit_operator


def run_alignment(
    model,
    n_entities: int,
    layers: list[int],
    token_position: int = -1,
    verbose: bool = True,
) -> dict:
    """Fit permutation operators across layers using bulk prompt samples.

    For each permutation π, collects last-token activations from many
    (base_prompt, permuted_prompt) pairs, then fits W_π via least-squares.
    """
    samples = generate_bulk_samples(n_entities)
    group = PermutationGroup(n_entities)

    # Group samples by permutation
    perm_samples = defaultdict(list)
    for s in samples:
        perm_samples[s["permutation"]].append(s)

    n_perms = len(perm_samples)
    n_samples_per = len(next(iter(perm_samples.values())))
    total_prompts = len(samples) * 2  # base + permuted for each

    if verbose:
        print(f"\nS_{n_entities}: {n_perms} non-identity permutations, "
              f"{n_samples_per} samples each, {total_prompts} total forward passes")

    # Collect activations: for each sample, cache base and permuted
    # activations at the specified token position
    perm_base_acts = defaultdict(lambda: {l: [] for l in layers})
    perm_perm_acts = defaultdict(lambda: {l: [] for l in layers})

    all_prompts = set()
    for s in samples:
        all_prompts.add(s["base_prompt"])
        all_prompts.add(s["permuted_prompt"])

    # Cache all unique prompts once
    if verbose:
        print(f"Caching activations for {len(all_prompts)} unique prompts...")
    prompt_cache = {}
    for prompt in tqdm(all_prompts, disable=not verbose, desc="Caching"):
        cached = cache_activations(
            model, prompt, hook_points=["resid_post"], layers=layers
        )
        # Store last-token activation per layer
        resid = cached.activations["resid_post"]  # (n_layers, seq_len, d_model)
        prompt_cache[prompt] = {
            layers[i]: resid[i, token_position, :] for i in range(len(layers))
        }

    # Organize by permutation
    for perm, samps in perm_samples.items():
        for s in samps:
            for layer in layers:
                perm_base_acts[perm][layer].append(prompt_cache[s["base_prompt"]][layer])
                perm_perm_acts[perm][layer].append(prompt_cache[s["permuted_prompt"]][layer])

    # Fit operators per (permutation, layer)
    if verbose:
        print("Fitting operators...")

    layer_results = {layer: {} for layer in layers}
    for perm in perm_samples:
        for layer in layers:
            base_tensor = torch.stack(perm_base_acts[perm][layer])
            perm_tensor = torch.stack(perm_perm_acts[perm][layer])
            fit = fit_operator(base_tensor, perm_tensor)
            layer_results[layer][str(perm)] = {
                "relative_error": fit.relative_error,
                "cosine_similarity": fit.cosine_similarity,
                "test_relative_error": fit.test_relative_error,
                "test_cosine_similarity": fit.test_cosine_similarity,
                "parity": perm.parity,
                "n_samples": fit.n_samples,
            }

    # Summarize per layer
    summary = {}
    for layer in layers:
        fits = layer_results[layer]
        test_errors = [f["test_relative_error"] for f in fits.values()
                       if f["test_relative_error"] is not None]
        test_cosines = [f["test_cosine_similarity"] for f in fits.values()
                        if f["test_cosine_similarity"] is not None]
        even_test = [f["test_relative_error"] for f in fits.values()
                     if f["parity"] == 0 and f["test_relative_error"] is not None]
        odd_test = [f["test_relative_error"] for f in fits.values()
                    if f["parity"] == 1 and f["test_relative_error"] is not None]
        summary[layer] = {
            "mean_test_error": float(np.mean(test_errors)) if test_errors else None,
            "mean_test_cosine": float(np.mean(test_cosines)) if test_cosines else None,
            "mean_even_test_error": float(np.mean(even_test)) if even_test else None,
            "mean_odd_test_error": float(np.mean(odd_test)) if odd_test else None,
            "per_permutation": fits,
        }

    if verbose:
        print(f"\n{'Layer':>6s}  {'TestErr':>10s}  {'TestCos':>8s}  {'EvenErr':>8s}  {'OddErr':>8s}")
        print(f"{'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
        for layer in layers:
            r = summary[layer]
            err_str = f"{r['mean_test_error']:.4f}" if r['mean_test_error'] is not None else "   n/a"
            cos_str = f"{r['mean_test_cosine']:.4f}" if r['mean_test_cosine'] is not None else "   n/a"
            even_str = f"{r['mean_even_test_error']:.4f}" if r['mean_even_test_error'] is not None else "   n/a"
            odd_str = f"{r['mean_odd_test_error']:.4f}" if r['mean_odd_test_error'] is not None else "   n/a"
            print(f"{layer:>6d}  {err_str:>10s}  {cos_str:>8s}  {even_str:>8s}  {odd_str:>8s}")

        best_layer = min(layers, key=lambda l: summary[l]["mean_test_error"] or float("inf"))
        worst_layer = max(layers, key=lambda l: summary[l]["mean_test_error"] or 0)
        print(f"\n  Best layer:  {best_layer} (test_err={summary[best_layer]['mean_test_error']:.4f})")
        print(f"  Worst layer: {worst_layer} (test_err={summary[worst_layer]['mean_test_error']:.4f})")

    return {
        "n_entities": n_entities,
        "n_samples_per_perm": n_samples_per,
        "token_position": token_position,
        "layers": {str(k): v for k, v in summary.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Activation capture and linear alignment search")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-entities", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"Loaded. {n_layers} layers, d_model={d_model}")

    layers = list(range(n_layers))
    all_results = {}

    for n in args.n_entities:
        result = run_alignment(model, n, layers, verbose=not args.quiet)
        all_results[f"S_{n}"] = result

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def default_serializer(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        output_path.write_text(json.dumps(all_results, indent=2, default=default_serializer))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
