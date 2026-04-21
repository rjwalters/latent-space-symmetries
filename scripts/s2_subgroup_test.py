"""S_2 subgroup composition test.

Tests whether pairwise swap operators compose within S_2 subgroups of S_3.
For each transposition (0,1), (1,2), (0,2):
  - Fit the swap operator W_s
  - Test s² = e (involution property)
  - Compare across layers

This tests the hypothesis that the model has learned "linguistically natural"
binary swap structure (subject-object reversal) even though it hasn't learned
full S_3 group algebra.

Also tests cross-transposition composition: does W_{(0,1)} W_{(1,2)} ≈ W_{(0,1,2)}?
This separates "each swap is an involution" from "swaps compose as a group."
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
    _relation_error,
)


def run_s2_analysis(model, layers: list[int], verbose: bool = True) -> dict:
    """Fit all S_3 operators, then test S_2 subgroup properties separately."""
    samples = generate_bulk_samples(3)
    group = PermutationGroup(3)
    gens = group.generators()

    perm_samples = defaultdict(list)
    for s in samples:
        perm_samples[s["permutation"]].append(s)

    # Cache activations
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

    # Identify the three transpositions and two 3-cycles
    transpositions = [p for p in group.elements if not p.is_identity and p.parity == 1]
    three_cycles = [p for p in group.elements if not p.is_identity and p.parity == 0]

    if verbose:
        print(f"\nTranspositions: {[str(t) for t in transpositions]}")
        print(f"3-cycles: {[str(c) for c in three_cycles]}")

    results_by_layer = {}

    for layer in layers:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Layer {layer}")
            print(f"{'='*70}")

        # Fit all operators
        operators = {}
        operators[Permutation.identity(3)] = np.eye(model.cfg.d_model)
        fit_info = {}

        for perm, samps in perm_samples.items():
            base_acts = np.stack([prompt_cache[s["base_prompt"]][layer] for s in samps])
            perm_acts = np.stack([prompt_cache[s["permuted_prompt"]][layer] for s in samps])
            fit = fit_operator(torch.tensor(base_acts), torch.tensor(perm_acts))
            operators[perm] = fit.W
            fit_info[perm] = fit

        # --- Test 1: Involution property for each transposition ---
        # Each s should satisfy s² = e
        I = np.eye(model.cfg.d_model)
        involution_errors = {}
        for t in transpositions:
            W_t = operators[t]
            err = _relation_error(f"{t}^2 = e", W_t @ W_t, I)
            involution_errors[str(t)] = {
                "relative_error": err.relative_error,
                "frobenius_error": err.frobenius_error,
                "test_fit_error": fit_info[t].test_relative_error,
            }

        # --- Test 2: Cross-transposition composition ---
        # (0,1)(1,2) = (0,1,2), (1,2)(0,1) = (0,2,1), etc.
        cross_comp = []
        for t1 in transpositions:
            for t2 in transpositions:
                if t1 == t2:
                    continue
                composed = t1 * t2
                if composed in operators:
                    err = composition_error(operators[t1], operators[t2], operators[composed])
                    cross_comp.append({
                        "t1": str(t1), "t2": str(t2),
                        "composed": str(composed),
                        "composed_parity": "even" if composed.is_even else "odd",
                        "relative_error": err.relative_error,
                    })

        # --- Test 3: Full S_3 relations for comparison ---
        s_perm = gens["s"]
        r_perm = gens["r"]
        s3_rels = s3_relation_errors(operators[s_perm], operators[r_perm])

        # --- Test 4: Eigenvalue analysis of each transposition ---
        # A true involution has eigenvalues ±1 only
        eigen_analysis = {}
        for t in transpositions:
            W_t = operators[t]
            eigenvalues = np.linalg.eigvals(W_t)
            # Sort by distance from +1 and -1
            dist_from_1 = np.abs(np.abs(eigenvalues) - 1.0)
            n_near_unit = np.sum(dist_from_1 < 0.1)
            # Count eigenvalues near +1 vs -1
            real_eigs = eigenvalues[np.abs(eigenvalues.imag) < 0.01].real
            n_near_plus1 = np.sum(np.abs(real_eigs - 1.0) < 0.1)
            n_near_minus1 = np.sum(np.abs(real_eigs + 1.0) < 0.1)
            eigen_analysis[str(t)] = {
                "n_near_unit_circle": int(n_near_unit),
                "n_near_plus1": int(n_near_plus1),
                "n_near_minus1": int(n_near_minus1),
                "top5_eigenvalue_magnitudes": sorted(np.abs(eigenvalues), reverse=True)[:5].tolist() if hasattr(sorted(np.abs(eigenvalues), reverse=True)[:5], 'tolist') else list(sorted(np.abs(eigenvalues), reverse=True)[:5]),
                "det": float(np.real(np.linalg.det(W_t))),
            }

        if verbose:
            print(f"\n  --- S_2 Involution Tests (s² = e) ---")
            print(f"  {'Transposition':>15s}  {'s²=e err':>10s}  {'fit err':>10s}")
            print(f"  {'─'*15}  {'─'*10}  {'─'*10}")
            for t in transpositions:
                inv = involution_errors[str(t)]
                print(f"  {str(t):>15s}  {inv['relative_error']:>10.4f}  "
                      f"{inv['test_fit_error']:>10.4f}")

            mean_involution = np.mean([v["relative_error"] for v in involution_errors.values()])
            print(f"\n  Mean involution error: {mean_involution:.4f}")

            print(f"\n  --- Cross-Transposition Composition ---")
            print(f"  {'t1':>10s} × {'t2':>10s} = {'result':>10s}  {'parity':>6s}  {'error':>8s}")
            print(f"  {'─'*10}   {'─'*10}   {'─'*10}  {'─'*6}  {'─'*8}")
            for c in cross_comp:
                print(f"  {c['t1']:>10s} × {c['t2']:>10s} = {c['composed']:>10s}  "
                      f"{c['composed_parity']:>6s}  {c['relative_error']:>8.4f}")

            mean_cross = np.mean([c["relative_error"] for c in cross_comp])
            print(f"\n  Mean cross-composition error: {mean_cross:.4f}")

            print(f"\n  --- Full S_3 Relations (for comparison) ---")
            for name, err in s3_rels.items():
                print(f"  {name}: {err.relative_error:.4f}")

            print(f"\n  --- Eigenvalue Analysis ---")
            for t_str, ea in eigen_analysis.items():
                print(f"  {t_str}: det={ea['det']:.4f}, "
                      f"near_+1={ea['n_near_plus1']}, near_-1={ea['n_near_minus1']}, "
                      f"top5_|λ|={[f'{x:.3f}' for x in ea['top5_eigenvalue_magnitudes']]}")

        results_by_layer[layer] = {
            "involution_errors": involution_errors,
            "cross_composition": cross_comp,
            "s3_relations": {
                name: {"frobenius": r.frobenius_error, "relative": r.relative_error}
                for name, r in s3_rels.items()
            },
            "eigenvalue_analysis": {k: {kk: vv for kk, vv in v.items()
                                         if kk != "top5_eigenvalue_magnitudes"}
                                    for k, v in eigen_analysis.items()},
        }

    # Summary
    if verbose:
        print(f"\n\n{'='*70}")
        print(f"SUMMARY: S_2 Involution vs S_3 Composition")
        print(f"{'='*70}")
        print(f"\n{'Layer':>6s}  {'Mean s²=e':>10s}  {'Mean cross':>10s}  "
              f"{'S3 s²=e':>8s}  {'S3 r³=e':>8s}  {'S3 srs':>8s}")
        print("─" * 65)
        for layer in layers:
            r = results_by_layer[layer]
            mean_inv = np.mean([v["relative_error"] for v in r["involution_errors"].values()])
            mean_cross = np.mean([c["relative_error"] for c in r["cross_composition"]])
            s3 = r["s3_relations"]
            print(f"{layer:>6d}  {mean_inv:>10.4f}  {mean_cross:>10.4f}  "
                  f"{s3['s^2 = e']['relative']:>8.4f}  "
                  f"{s3['r^3 = e']['relative']:>8.4f}  "
                  f"{s3['srs = r^{-1}']['relative']:>8.4f}")

        print(f"\nKey: If mean_s²=e << S3_r³=e, the model has learned pairwise")
        print(f"swap structure without full group composition.")

    return results_by_layer


def main():
    parser = argparse.ArgumentParser(description="S_2 subgroup composition test")
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

    results = run_s2_analysis(model, args.layers, verbose=not args.quiet)

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
