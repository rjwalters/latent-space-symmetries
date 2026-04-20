"""Phase 5: S_3 operator fitting and group relation testing.

Fit W_π for all 5 non-identity permutations of S_3 at selected layers,
then test whether the fitted operators satisfy the S_3 presentation:
  s² = e,  r³ = e,  srs = r⁻¹

Also tests composition closure: W_{π1} W_{π2} ≈ W_{π1∘π2} for all pairs.
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
from src.operators.group_tests import (
    identity_error,
    composition_error,
    inverse_error,
    s3_relation_errors,
    parity_analysis,
    RelationError,
)


def fit_all_operators(
    model,
    layers: list[int],
    verbose: bool = True,
) -> dict[int, dict[Permutation, np.ndarray]]:
    """Fit W_π for all S_3 permutations at each layer.

    Returns dict: layer -> {permutation -> W matrix}.
    """
    samples = generate_bulk_samples(3)
    group = PermutationGroup(3)

    # Group samples by permutation
    perm_samples = defaultdict(list)
    for s in samples:
        perm_samples[s["permutation"]].append(s)

    # Cache all unique prompts
    all_prompts = set()
    for s in samples:
        all_prompts.add(s["base_prompt"])
        all_prompts.add(s["permuted_prompt"])

    if verbose:
        print(f"Caching {len(all_prompts)} unique prompts across {len(layers)} layers...")
    prompt_cache = {}
    for prompt in tqdm(all_prompts, disable=not verbose, desc="Caching"):
        cached = cache_activations(
            model, prompt, hook_points=["resid_post"], layers=layers
        )
        resid = cached.activations["resid_post"]
        prompt_cache[prompt] = {
            layers[i]: resid[i, -1, :] for i in range(len(layers))
        }

    # Fit operators
    if verbose:
        print("Fitting operators for all permutations...")

    operators = {}  # layer -> {perm -> W}
    fit_results = {}  # layer -> {perm -> OperatorFit}

    for layer in layers:
        operators[layer] = {}
        fit_results[layer] = {}

        # Identity operator = I
        d = model.cfg.d_model
        operators[layer][Permutation.identity(3)] = np.eye(d)

        for perm, samps in perm_samples.items():
            base_acts = torch.stack([prompt_cache[s["base_prompt"]][layer] for s in samps])
            perm_acts = torch.stack([prompt_cache[s["permuted_prompt"]][layer] for s in samps])
            fit = fit_operator(base_acts, perm_acts)
            operators[layer][perm] = fit.W
            fit_results[layer][perm] = fit

    return operators, fit_results


def test_group_relations(
    operators: dict[int, dict],
    layers: list[int],
    verbose: bool = True,
) -> dict:
    """Test S_3 group relations at each layer."""
    group = PermutationGroup(3)
    gens = group.generators()
    s_perm = gens["s"]  # (0,1)
    r_perm = gens["r"]  # (0,1,2)

    results = {}

    for layer in layers:
        ops = operators[layer]
        W_s = ops[s_perm]
        W_r = ops[r_perm]

        # S_3 presentation relations
        s3_rels = s3_relation_errors(W_s, W_r)

        # Identity check for W_e
        id_perm = Permutation.identity(3)
        id_err = identity_error(ops[id_perm])

        # Inverse checks for all permutations
        inv_errors = {}
        for perm in ops:
            if perm.is_identity:
                continue
            perm_inv = perm.inverse()
            if perm_inv in ops:
                inv_err = inverse_error(ops[perm], ops[perm_inv])
                inv_errors[str(perm)] = {
                    "frobenius": inv_err.frobenius_error,
                    "relative": inv_err.relative_error,
                }

        # Composition closure: test all pairs
        comp_errors = []
        for p1 in group.elements:
            if p1.is_identity:
                continue
            for p2 in group.elements:
                if p2.is_identity:
                    continue
                composed = p1 * p2
                if composed in ops and p1 in ops and p2 in ops:
                    err = composition_error(ops[p1], ops[p2], ops[composed])
                    comp_errors.append({
                        "p1": str(p1),
                        "p2": str(p2),
                        "p1p2": str(composed),
                        "relative_error": err.relative_error,
                    })

        mean_comp_err = np.mean([c["relative_error"] for c in comp_errors]) if comp_errors else None
        mean_inv_err = np.mean([v["relative"] for v in inv_errors.values()]) if inv_errors else None

        layer_result = {
            "s3_relations": {
                name: {"frobenius": r.frobenius_error, "relative": r.relative_error}
                for name, r in s3_rels.items()
            },
            "identity_error": id_err.relative_error,
            "mean_inverse_error": float(mean_inv_err) if mean_inv_err is not None else None,
            "inverse_errors": inv_errors,
            "mean_composition_error": float(mean_comp_err) if mean_comp_err is not None else None,
            "composition_errors": comp_errors,
        }
        results[layer] = layer_result

    if verbose:
        print(f"\n{'Layer':>6s}  {'s²=e':>8s}  {'r³=e':>8s}  {'srs=r⁻¹':>8s}  "
              f"{'MeanInv':>8s}  {'MeanComp':>8s}")
        print(f"{'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for layer in layers:
            r = results[layer]
            s3 = r["s3_relations"]
            print(f"{layer:>6d}  "
                  f"{s3['s^2 = e']['relative']:>8.4f}  "
                  f"{s3['r^3 = e']['relative']:>8.4f}  "
                  f"{s3['srs = r^{-1}']['relative']:>8.4f}  "
                  f"{r['mean_inverse_error']:>8.4f}  "
                  f"{r['mean_composition_error']:>8.4f}")

    return results


def test_fit_quality(
    fit_results: dict[int, dict],
    layers: list[int],
    verbose: bool = True,
) -> dict:
    """Report per-permutation fit quality at each layer."""
    results = {}
    for layer in layers:
        fits = fit_results[layer]
        layer_data = {}
        for perm, fit in fits.items():
            layer_data[str(perm)] = {
                "train_error": fit.relative_error,
                "test_error": fit.test_relative_error,
                "test_cosine": fit.test_cosine_similarity,
                "parity": perm.parity,
            }
        results[layer] = layer_data

    if verbose:
        # Show per-permutation test errors at each layer
        sample_layer = layers[0]
        perms = list(fit_results[sample_layer].keys())
        perm_strs = [str(p) for p in perms]

        print(f"\nPer-permutation test errors:")
        header = f"{'Layer':>6s}  " + "  ".join(f"{p:>12s}" for p in perm_strs)
        print(header)
        print("─" * len(header))
        for layer in layers:
            fits = fit_results[layer]
            vals = []
            for perm in perms:
                te = fits[perm].test_relative_error
                vals.append(f"{te:>12.4f}" if te is not None else f"{'n/a':>12s}")
            print(f"{layer:>6d}  " + "  ".join(vals))

    return results


def main():
    parser = argparse.ArgumentParser(description="S_3 group relation testing")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 2, 5, 8, 10, 13, 16, 18, 20, 21, 22, 23])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    print(f"Loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    operators, fit_results = fit_all_operators(
        model, args.layers, verbose=not args.quiet
    )

    fit_quality = test_fit_quality(fit_results, args.layers, verbose=not args.quiet)
    group_results = test_group_relations(operators, args.layers, verbose=not args.quiet)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def default_ser(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        save_data = {
            "fit_quality": {str(k): v for k, v in fit_quality.items()},
            "group_relations": {str(k): v for k, v in group_results.items()},
        }
        output_path.write_text(json.dumps(save_data, indent=2, default=default_ser))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
