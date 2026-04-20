"""Subspace search for permutation group structure.

Instead of fitting operators in full d_model space, find a lower-dimensional
subspace where permutation effects concentrate, then test group relations there.

Strategy:
1. Collect activation differences Δh = h(π·x) - h(x) for all permutations
2. PCA on the differences to find the permutation-sensitive subspace
3. Project activations into this subspace
4. Fit operators and test group relations in the reduced space
5. Compare A_3 (even perms only) vs full S_3 relation quality

Key hypothesis: even permutations may compose more cleanly in a subspace,
suggesting A_n structure is more natural than S_n.
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
from scipy.linalg import lstsq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation.capture import load_model, cache_activations, DEFAULT_MODEL
from src.datasets.permutations import Permutation, PermutationGroup
from src.datasets.prompts import generate_bulk_samples
from src.operators.group_tests import (
    composition_error,
    identity_error,
    inverse_error,
    s3_relation_errors,
)


def collect_activations(model, layers, verbose=True):
    """Collect base and permuted activations for all S_3 samples."""
    samples = generate_bulk_samples(3)
    perm_samples = defaultdict(list)
    for s in samples:
        perm_samples[s["permutation"]].append(s)

    all_prompts = set()
    for s in samples:
        all_prompts.add(s["base_prompt"])
        all_prompts.add(s["permuted_prompt"])

    if verbose:
        print(f"Caching {len(all_prompts)} prompts...")
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

    return samples, perm_samples, prompt_cache


def pca_on_differences(samples, prompt_cache, layer, n_components=None):
    """PCA on activation differences to find permutation-sensitive subspace.

    Returns:
        V: (n_components, d_model) projection matrix (rows are principal components)
        explained_variance_ratio: fraction of variance per component
        singular_values: singular values
    """
    # Collect all differences
    diffs = []
    for s in samples:
        base_act = prompt_cache[s["base_prompt"]][layer]
        perm_act = prompt_cache[s["permuted_prompt"]][layer]
        diffs.append(perm_act - base_act)
    diffs = np.stack(diffs)  # (n_samples, d_model)

    # Center
    mean_diff = diffs.mean(axis=0)
    diffs_centered = diffs - mean_diff

    # SVD
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)

    total_var = np.sum(S**2)
    explained_ratio = S**2 / (total_var + 1e-10)

    if n_components is not None:
        Vt = Vt[:n_components]
        explained_ratio = explained_ratio[:n_components]
        S = S[:n_components]

    return Vt, explained_ratio, S, mean_diff


def fit_operator_in_subspace(
    base_acts: np.ndarray,
    perm_acts: np.ndarray,
    V: np.ndarray,
    test_fraction: float = 0.2,
):
    """Fit operator in PCA subspace. Returns W, train_error, test_error."""
    # Project into subspace
    X = base_acts @ V.T  # (n, k)
    Y = perm_acts @ V.T  # (n, k)

    n = X.shape[0]
    n_test = max(1, int(n * test_fraction))
    X_train, X_test = X[:-n_test], X[-n_test:]
    Y_train, Y_test = Y[:-n_test], Y[-n_test:]

    W_T, _, _, _ = lstsq(X_train, Y_train)
    W = W_T.T

    # Train error
    Y_train_pred = (W @ X_train.T).T
    train_err = np.linalg.norm(Y_train - Y_train_pred, "fro") / (np.linalg.norm(Y_train, "fro") + 1e-10)

    # Test error
    Y_test_pred = (W @ X_test.T).T
    test_err = np.linalg.norm(Y_test - Y_test_pred, "fro") / (np.linalg.norm(Y_test, "fro") + 1e-10)

    # Cosine similarity on test
    dots = np.sum(Y_test * Y_test_pred, axis=1)
    norms_y = np.linalg.norm(Y_test, axis=1)
    norms_p = np.linalg.norm(Y_test_pred, axis=1)
    cos = float(np.mean(dots / (norms_y * norms_p + 1e-10)))

    return W, train_err, test_err, cos


def run_subspace_analysis(
    model, layer, samples, perm_samples, prompt_cache,
    subspace_dims: list[int],
    verbose=True,
):
    """Run full subspace analysis at one layer across multiple subspace dimensions."""
    group = PermutationGroup(3)
    gens = group.generators()
    s_perm = gens["s"]
    r_perm = gens["r"]

    if verbose:
        print(f"\n{'='*80}")
        print(f"Layer {layer}")
        print(f"{'='*80}")

    # PCA
    V_full, explained_ratio, sv, mean_diff = pca_on_differences(
        samples, prompt_cache, layer
    )

    if verbose:
        cumvar = np.cumsum(explained_ratio)
        print(f"\nVariance explained by top components:")
        for k in [5, 10, 20, 50, 100, 200]:
            if k <= len(cumvar):
                print(f"  Top {k:>3d}: {cumvar[k-1]:.4f}")
        print(f"  Total components with >1% variance: "
              f"{np.sum(explained_ratio > 0.01)}")

    results_by_dim = {}

    for n_dim in subspace_dims:
        V = V_full[:n_dim]

        # Fit operators for all permutations
        operators = {}
        fit_info = {}
        d = n_dim
        operators[Permutation.identity(3)] = np.eye(d)

        for perm, samps in perm_samples.items():
            base_acts = np.stack([prompt_cache[s["base_prompt"]][layer] for s in samps])
            perm_acts = np.stack([prompt_cache[s["permuted_prompt"]][layer] for s in samps])
            W, train_err, test_err, cos = fit_operator_in_subspace(
                base_acts, perm_acts, V
            )
            operators[perm] = W
            fit_info[perm] = {
                "train_error": train_err,
                "test_error": test_err,
                "test_cosine": cos,
                "parity": perm.parity,
            }

        # Group relations
        W_s = operators[s_perm]
        W_r = operators[r_perm]
        s3_rels = s3_relation_errors(W_s, W_r)

        # Inverse errors
        inv_errors = {}
        for perm in operators:
            if perm.is_identity:
                continue
            perm_inv = perm.inverse()
            if perm_inv in operators:
                inv_err = inverse_error(operators[perm], operators[perm_inv])
                inv_errors[str(perm)] = inv_err.relative_error

        # Composition errors — split by parity
        even_comp_errors = []
        odd_comp_errors = []
        mixed_comp_errors = []
        all_comp_errors = []

        for p1 in group.elements:
            for p2 in group.elements:
                if p1.is_identity or p2.is_identity:
                    continue
                composed = p1 * p2
                if composed in operators and p1 in operators and p2 in operators:
                    err = composition_error(operators[p1], operators[p2], operators[composed])
                    all_comp_errors.append(err.relative_error)

                    # Classify by parity of inputs
                    if p1.is_even and p2.is_even:
                        even_comp_errors.append(err.relative_error)
                    elif not p1.is_even and not p2.is_even:
                        odd_comp_errors.append(err.relative_error)
                    else:
                        mixed_comp_errors.append(err.relative_error)

        # Even-only fit quality vs odd
        even_test_errors = [v["test_error"] for p, v in fit_info.items() if v["parity"] == 0]
        odd_test_errors = [v["test_error"] for p, v in fit_info.items() if v["parity"] == 1]

        dim_result = {
            "n_dim": n_dim,
            "variance_explained": float(np.sum(explained_ratio[:n_dim])),
            "fit_quality": {str(p): v for p, v in fit_info.items()},
            "mean_test_error": float(np.mean([v["test_error"] for v in fit_info.values()])),
            "mean_even_test_error": float(np.mean(even_test_errors)) if even_test_errors else None,
            "mean_odd_test_error": float(np.mean(odd_test_errors)) if odd_test_errors else None,
            "s3_relations": {
                name: {"frobenius": r.frobenius_error, "relative": r.relative_error}
                for name, r in s3_rels.items()
            },
            "mean_inverse_error": float(np.mean(list(inv_errors.values()))) if inv_errors else None,
            "composition": {
                "mean_all": float(np.mean(all_comp_errors)) if all_comp_errors else None,
                "mean_even_even": float(np.mean(even_comp_errors)) if even_comp_errors else None,
                "mean_odd_odd": float(np.mean(odd_comp_errors)) if odd_comp_errors else None,
                "mean_mixed": float(np.mean(mixed_comp_errors)) if mixed_comp_errors else None,
            },
        }
        results_by_dim[n_dim] = dim_result

    if verbose:
        print(f"\n{'Dim':>5s}  {'VarExp':>7s}  {'TestErr':>8s}  "
              f"{'s²=e':>7s}  {'r³=e':>7s}  {'srs':>7s}  "
              f"{'Inv':>7s}  {'Comp':>7s}  "
              f"{'EE comp':>8s}  {'OO comp':>8s}  {'Mix comp':>8s}")
        print("─" * 105)
        for n_dim in subspace_dims:
            r = results_by_dim[n_dim]
            s3 = r["s3_relations"]
            c = r["composition"]
            print(f"{n_dim:>5d}  {r['variance_explained']:>7.4f}  {r['mean_test_error']:>8.4f}  "
                  f"{s3['s^2 = e']['relative']:>7.4f}  "
                  f"{s3['r^3 = e']['relative']:>7.4f}  "
                  f"{s3['srs = r^{-1}']['relative']:>7.4f}  "
                  f"{r['mean_inverse_error']:>7.4f}  "
                  f"{c['mean_all']:>7.4f}  "
                  f"{c['mean_even_even'] or 0:>8.4f}  "
                  f"{c['mean_odd_odd'] or 0:>8.4f}  "
                  f"{c['mean_mixed'] or 0:>8.4f}")

        # Highlight parity differences
        print(f"\nParity comparison (fit quality):")
        for n_dim in subspace_dims:
            r = results_by_dim[n_dim]
            even = r["mean_even_test_error"]
            odd = r["mean_odd_test_error"]
            if even is not None and odd is not None:
                diff = odd - even
                better = "even" if diff > 0 else "odd"
                print(f"  dim={n_dim:>3d}: even_err={even:.4f}, odd_err={odd:.4f}, "
                      f"Δ={abs(diff):.4f} ({better} fits better)")

    return results_by_dim


def main():
    parser = argparse.ArgumentParser(description="Subspace search for group structure")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 10, 16, 20, 21, 22])
    parser.add_argument("--dims", type=int, nargs="+", default=[3, 5, 10, 20, 50, 100])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    print(f"Loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    samples, perm_samples, prompt_cache = collect_activations(
        model, args.layers, verbose=not args.quiet
    )

    all_results = {}
    for layer in args.layers:
        layer_results = run_subspace_analysis(
            model, layer, samples, perm_samples, prompt_cache,
            subspace_dims=args.dims, verbose=not args.quiet,
        )
        all_results[layer] = layer_results

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def ser(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        save = {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in all_results.items()}
        output_path.write_text(json.dumps(save, indent=2, default=ser))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
