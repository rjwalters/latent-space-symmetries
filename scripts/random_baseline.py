"""Random baseline: fit operators on unrelated prompt pairs.

Establishes what operator fit quality and group relation errors look like
when there is NO permutation relationship between prompts. This answers:

1. Is the real model's operator fit quality (test_err ~0.08) significantly
   better than chance? (Yes, if random pairs give much higher error.)
2. Are the real model's group relation errors (~1.0) distinguishable from
   random operators? (If not, the model's "operators" have no more group
   structure than noise.)

Methodology:
- Use the same prompts and activation cache as the real experiments
- Shuffle the pairing so base and "permuted" prompts are unrelated
- Fit operators and test group relations on the shuffled pairs
- Compare to real paired results
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
    composition_error,
    identity_error,
    inverse_error,
    s3_relation_errors,
)


def run_random_baseline(
    model, layers: list[int], n_shuffles: int = 5, verbose: bool = True
) -> dict:
    """Fit operators on shuffled (unrelated) prompt pairs.

    For each shuffle:
    1. Generate the same S_3 prompt pairs as the real experiment
    2. Randomly reassign "permuted" activations to different base activations
    3. Fit operators and test group relations
    4. Average over shuffles for stable estimates
    """
    samples = generate_bulk_samples(3)
    group = PermutationGroup(3)
    gens = group.generators()
    s_perm = gens["s"]
    r_perm = gens["r"]

    perm_samples = defaultdict(list)
    for s in samples:
        perm_samples[s["permutation"]].append(s)

    # Cache activations (same as real experiment)
    all_prompts = set()
    for s in samples:
        all_prompts.add(s["base_prompt"])
        all_prompts.add(s["permuted_prompt"])

    if verbose:
        print(f"Caching {len(all_prompts)} prompts across {len(layers)} layers...")
    prompt_cache = {}
    for prompt in tqdm(all_prompts, disable=not verbose, desc="Caching"):
        cached = cache_activations(
            model, prompt, hook_points=["resid_post"], layers=layers
        )
        resid = cached.activations["resid_post"]
        prompt_cache[prompt] = {
            layers[i]: resid[i, -1, :].detach().cpu().numpy()
            for i in range(len(layers))
        }

    rng = np.random.default_rng(42)
    results_by_layer = {}

    for layer in layers:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layer {layer}")
            print(f"{'='*60}")

        shuffle_results = []

        for shuffle_idx in range(n_shuffles):
            # --- Real pairing (for comparison, first shuffle only) ---
            if shuffle_idx == 0:
                real_ops = {}
                real_ops[Permutation.identity(3)] = np.eye(model.cfg.d_model)
                real_fit_info = {}
                for perm, samps in perm_samples.items():
                    base_acts = np.stack([prompt_cache[s["base_prompt"]][layer] for s in samps])
                    perm_acts = np.stack([prompt_cache[s["permuted_prompt"]][layer] for s in samps])
                    fit = fit_operator(
                        torch.tensor(base_acts), torch.tensor(perm_acts)
                    )
                    real_ops[perm] = fit.W
                    real_fit_info[str(perm)] = {
                        "test_error": fit.test_relative_error,
                        "test_cosine": fit.test_cosine_similarity,
                    }

                real_s3 = s3_relation_errors(real_ops[s_perm], real_ops[r_perm])
                real_comp_errors = []
                for p1 in group.elements:
                    for p2 in group.elements:
                        if p1.is_identity or p2.is_identity:
                            continue
                        composed = p1 * p2
                        if all(p in real_ops for p in [p1, p2, composed]):
                            err = composition_error(real_ops[p1], real_ops[p2], real_ops[composed])
                            real_comp_errors.append(err.relative_error)

            # --- Shuffled pairing ---
            shuffled_ops = {}
            shuffled_ops[Permutation.identity(3)] = np.eye(model.cfg.d_model)
            shuffled_fit_info = {}

            for perm, samps in perm_samples.items():
                base_acts = np.stack([prompt_cache[s["base_prompt"]][layer] for s in samps])
                perm_acts = np.stack([prompt_cache[s["permuted_prompt"]][layer] for s in samps])
                # Shuffle the permuted activations
                perm_acts = perm_acts[rng.permutation(len(perm_acts))]
                fit = fit_operator(
                    torch.tensor(base_acts), torch.tensor(perm_acts)
                )
                shuffled_ops[perm] = fit.W
                shuffled_fit_info[str(perm)] = {
                    "test_error": fit.test_relative_error,
                    "test_cosine": fit.test_cosine_similarity,
                }

            # Group relations on shuffled operators
            s3_rels = s3_relation_errors(shuffled_ops[s_perm], shuffled_ops[r_perm])

            comp_errors = []
            for p1 in group.elements:
                for p2 in group.elements:
                    if p1.is_identity or p2.is_identity:
                        continue
                    composed = p1 * p2
                    if all(p in shuffled_ops for p in [p1, p2, composed]):
                        err = composition_error(shuffled_ops[p1], shuffled_ops[p2], shuffled_ops[composed])
                        comp_errors.append(err.relative_error)

            inv_errors = []
            for perm in shuffled_ops:
                if perm.is_identity:
                    continue
                perm_inv = perm.inverse()
                if perm_inv in shuffled_ops:
                    inv_err = inverse_error(shuffled_ops[perm], shuffled_ops[perm_inv])
                    inv_errors.append(inv_err.relative_error)

            shuffle_results.append({
                "fit_quality": shuffled_fit_info,
                "s3_relations": {
                    name: r.relative_error for name, r in s3_rels.items()
                },
                "mean_inverse_error": float(np.mean(inv_errors)),
                "mean_composition_error": float(np.mean(comp_errors)),
                "mean_test_error": float(np.mean([
                    v["test_error"] for v in shuffled_fit_info.values()
                    if v["test_error"] is not None
                ])),
            })

        # Aggregate across shuffles
        mean_shuffled_test = float(np.mean([r["mean_test_error"] for r in shuffle_results]))
        mean_shuffled_s2 = float(np.mean([r["s3_relations"]["s^2 = e"] for r in shuffle_results]))
        mean_shuffled_r3 = float(np.mean([r["s3_relations"]["r^3 = e"] for r in shuffle_results]))
        mean_shuffled_srs = float(np.mean([r["s3_relations"]["srs = r^{-1}"] for r in shuffle_results]))
        mean_shuffled_inv = float(np.mean([r["mean_inverse_error"] for r in shuffle_results]))
        mean_shuffled_comp = float(np.mean([r["mean_composition_error"] for r in shuffle_results]))

        mean_real_test = float(np.mean([
            v["test_error"] for v in real_fit_info.values()
            if v["test_error"] is not None
        ]))
        mean_real_comp = float(np.mean(real_comp_errors))

        if verbose:
            print(f"\n{'':>16s}  {'TestErr':>8s}  {'s²=e':>8s}  {'r³=e':>8s}  "
                  f"{'srs':>8s}  {'Inv':>8s}  {'Comp':>8s}")
            print("─" * 80)
            print(f"{'Real pairs':>16s}  {mean_real_test:>8.4f}  "
                  f"{real_s3['s^2 = e'].relative_error:>8.4f}  "
                  f"{real_s3['r^3 = e'].relative_error:>8.4f}  "
                  f"{real_s3['srs = r^{-1}'].relative_error:>8.4f}  "
                  f"{'─':>8s}  {mean_real_comp:>8.4f}")
            print(f"{'Shuffled (mean)':>16s}  {mean_shuffled_test:>8.4f}  "
                  f"{mean_shuffled_s2:>8.4f}  "
                  f"{mean_shuffled_r3:>8.4f}  "
                  f"{mean_shuffled_srs:>8.4f}  "
                  f"{mean_shuffled_inv:>8.4f}  {mean_shuffled_comp:>8.4f}")

            # Effect size
            test_ratio = mean_shuffled_test / (mean_real_test + 1e-10)
            print(f"\n  Shuffled/Real test error ratio: {test_ratio:.2f}x")
            print(f"  → Real operator fit is {test_ratio:.1f}x better than random")

        results_by_layer[layer] = {
            "real": {
                "mean_test_error": mean_real_test,
                "s3_relations": {
                    name: r.relative_error for name, r in real_s3.items()
                },
                "mean_composition_error": mean_real_comp,
            },
            "shuffled": {
                "mean_test_error": mean_shuffled_test,
                "std_test_error": float(np.std([r["mean_test_error"] for r in shuffle_results])),
                "s3_relations": {
                    "s^2 = e": mean_shuffled_s2,
                    "r^3 = e": mean_shuffled_r3,
                    "srs = r^{-1}": mean_shuffled_srs,
                },
                "mean_inverse_error": mean_shuffled_inv,
                "mean_composition_error": mean_shuffled_comp,
            },
        }

    return results_by_layer


def main():
    parser = argparse.ArgumentParser(description="Random baseline for permutation operators")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 10, 16, 21])
    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    print(f"Loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    results = run_random_baseline(
        model, args.layers, n_shuffles=args.n_shuffles,
        verbose=not args.quiet,
    )

    # Final summary
    print(f"\n\n{'='*80}")
    print(f"SUMMARY: Real Pairs vs Random Baseline")
    print(f"{'='*80}")
    print(f"\n{'Layer':>6s}  {'':>8s}  {'TestErr':>8s}  {'s²=e':>8s}  "
          f"{'r³=e':>8s}  {'srs':>8s}  {'Comp':>8s}")
    print("─" * 70)
    for layer, r in results.items():
        print(f"{layer:>6d}  {'Real':>8s}  {r['real']['mean_test_error']:>8.4f}  "
              f"{r['real']['s3_relations']['s^2 = e']:>8.4f}  "
              f"{r['real']['s3_relations']['r^3 = e']:>8.4f}  "
              f"{r['real']['s3_relations']['srs = r^{-1}']:>8.4f}  "
              f"{r['real']['mean_composition_error']:>8.4f}")
        print(f"{'':>6s}  {'Shuffled':>8s}  {r['shuffled']['mean_test_error']:>8.4f}  "
              f"{r['shuffled']['s3_relations']['s^2 = e']:>8.4f}  "
              f"{r['shuffled']['s3_relations']['r^3 = e']:>8.4f}  "
              f"{r['shuffled']['s3_relations']['srs = r^{-1}']:>8.4f}  "
              f"{r['shuffled']['mean_composition_error']:>8.4f}")
        ratio = r['shuffled']['mean_test_error'] / (r['real']['mean_test_error'] + 1e-10)
        print(f"{'':>6s}  {'Ratio':>8s}  {ratio:>8.1f}x")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def ser(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        save = {str(k): v for k, v in results.items()}
        output_path.write_text(json.dumps(save, indent=2, default=ser))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
