"""Analyze permutation group structure in SAE feature space.

After training SAEs, encode permutation-related activations into the
sparse feature basis and test whether group relations hold better there
than in raw activation space.

Key questions:
1. Do permutation operators in feature space satisfy group relations better?
2. Are permutation effects concentrated on a small subset of features?
3. Does the sign/standard irrep structure emerge more cleanly?

Usage:
    python scripts/sae_group_analysis.py --sae-path data/sae_checkpoints/layer21_2M --cache data/activations/perm_cache.npz --layer 21
    python scripts/sae_group_analysis.py --sae-path data/sae_checkpoints/layer21_2M --layer 21 --top-k 50 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import lstsq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sae_lens import SAE
from src.activation.capture import load_model, cache_activations, DEFAULT_MODEL
from src.datasets.permutations import Permutation, PermutationGroup
from src.datasets.prompts import generate_bulk_samples
from src.operators.group_tests import (
    composition_error,
    inverse_error,
    s3_relation_errors,
)


def load_sae(sae_path: str, device: str = "cpu") -> SAE:
    """Load a trained SAE from disk."""
    sae = SAE.load_from_disk(sae_path, device=device)
    return sae


def load_cached_activations(cache_path: str, layer: int):
    """Load pre-cached activations from npz file.

    Returns same format as collect_activations_for_layer:
        samples, perm_samples, prompt_cache
    """
    data = np.load(cache_path, allow_pickle=True)

    prompts = data["prompts"]
    acts = data[f"acts_layer{layer}"]  # (n_prompts, d_model)
    prompt_to_act = {p: acts[i] for i, p in enumerate(prompts)}

    # Reconstruct samples and perm_samples
    base_prompts = data["base_prompts"]
    permuted_prompts = data["permuted_prompts"]
    perm_tuples = data["perm_tuples"]
    parities = data["parities"]

    samples = []
    perm_samples = defaultdict(list)
    for i in range(len(base_prompts)):
        perm = Permutation(tuple(int(x) for x in perm_tuples[i]))
        s = {
            "base_prompt": str(base_prompts[i]),
            "permuted_prompt": str(permuted_prompts[i]),
            "permutation": perm,
        }
        samples.append(s)
        perm_samples[perm].append(s)

    return samples, perm_samples, prompt_to_act


def encode_activations(
    sae: SAE,
    activations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Encode raw activations through SAE encoder to get sparse features."""
    encoded = {}
    with torch.no_grad():
        for prompt, act in activations.items():
            x = torch.tensor(act, dtype=torch.float32, device=sae.device).unsqueeze(0)
            features = sae.encode(x)
            encoded[prompt] = features[0].cpu().numpy()
    return encoded


def collect_activations_for_layer(model, layer: int, verbose=True):
    """Collect raw activations for all permutation samples at a given layer."""
    samples = generate_bulk_samples(3)
    perm_samples = defaultdict(list)
    for s in samples:
        perm_samples[s["permutation"]].append(s)

    all_prompts = set()
    for s in samples:
        all_prompts.add(s["base_prompt"])
        all_prompts.add(s["permuted_prompt"])

    if verbose:
        print(f"Caching {len(all_prompts)} prompts for layer {layer}...")
    prompt_cache = {}
    for prompt in tqdm(all_prompts, disable=not verbose, desc="Caching"):
        cached = cache_activations(
            model, prompt, hook_points=["resid_post"], layers=[layer]
        )
        resid = cached.activations["resid_post"]
        prompt_cache[prompt] = resid[0, -1, :].detach().cpu().numpy()

    return samples, perm_samples, prompt_cache


def analyze_sparsity(
    sae_features: dict[str, np.ndarray],
    samples: list,
    perm_samples: dict,
    verbose=True,
):
    """Analyze how sparse the permutation effects are in feature space."""
    # Which features activate differently under permutation?
    feature_diffs = []
    for s in samples:
        base_feat = sae_features[s["base_prompt"]]
        perm_feat = sae_features[s["permuted_prompt"]]
        feature_diffs.append(perm_feat - base_feat)
    feature_diffs = np.stack(feature_diffs)  # (n_samples, d_sae)

    # Mean absolute diff per feature
    mean_abs_diff = np.mean(np.abs(feature_diffs), axis=0)
    # Variance of diff per feature
    var_diff = np.var(feature_diffs, axis=0)

    # How many features are "permutation-sensitive"?
    threshold = np.percentile(mean_abs_diff[mean_abs_diff > 0], 90)
    n_active = np.sum(mean_abs_diff > 0)
    n_sensitive = np.sum(mean_abs_diff > threshold)

    if verbose:
        d_sae = feature_diffs.shape[1]
        print(f"\n--- Sparsity Analysis ---")
        print(f"  Total features: {d_sae}")
        print(f"  Features with any permutation effect: {n_active} ({n_active/d_sae*100:.1f}%)")
        print(f"  Highly sensitive features (>90th pctile): {n_sensitive}")
        print(f"  Top-10 feature indices by mean |Δ|: {np.argsort(mean_abs_diff)[-10:][::-1]}")
        print(f"  Mean activation diff (all features): {np.mean(mean_abs_diff):.6f}")
        print(f"  Mean activation diff (sensitive only): {np.mean(mean_abs_diff[mean_abs_diff > threshold]):.6f}")

    # Per-parity analysis
    if verbose:
        even_diffs = []
        odd_diffs = []
        for s in samples:
            diff = sae_features[s["permuted_prompt"]] - sae_features[s["base_prompt"]]
            if s["permutation"].is_even:
                even_diffs.append(diff)
            else:
                odd_diffs.append(diff)
        if even_diffs:
            even_diffs = np.stack(even_diffs)
            print(f"  Mean |Δ| for even perms: {np.mean(np.abs(even_diffs)):.6f}")
        if odd_diffs:
            odd_diffs = np.stack(odd_diffs)
            print(f"  Mean |Δ| for odd perms: {np.mean(np.abs(odd_diffs)):.6f}")

    return mean_abs_diff, var_diff


def fit_operators_in_feature_space(
    sae_features: dict[str, np.ndarray],
    perm_samples: dict,
    top_k: int | None = None,
    feature_mask: np.ndarray | None = None,
    verbose=True,
):
    """Fit permutation operators in SAE feature space.

    Args:
        top_k: If set, only use the top-k most permutation-sensitive features
        feature_mask: Boolean mask selecting which features to use
    """
    group = PermutationGroup(3)

    # Determine feature subset
    if feature_mask is not None:
        indices = np.where(feature_mask)[0]
    elif top_k is not None:
        # Need to compute sensitivity first
        all_diffs = []
        for perm, samps in perm_samples.items():
            for s in samps:
                base = sae_features[s["base_prompt"]]
                perm_act = sae_features[s["permuted_prompt"]]
                all_diffs.append(perm_act - base)
        all_diffs = np.stack(all_diffs)
        sensitivity = np.mean(np.abs(all_diffs), axis=0)
        indices = np.argsort(sensitivity)[-top_k:]
    else:
        # Use all features that have any activity
        sample_feat = next(iter(sae_features.values()))
        indices = np.arange(len(sample_feat))

    d = len(indices)
    if verbose:
        print(f"\n--- Operator Fitting in Feature Space (dim={d}) ---")

    operators = {Permutation.identity(3): np.eye(d)}
    fit_info = {}

    for perm, samps in perm_samples.items():
        X = np.stack([sae_features[s["base_prompt"]][indices] for s in samps])
        Y = np.stack([sae_features[s["permuted_prompt"]][indices] for s in samps])

        # Train/test split
        n = X.shape[0]
        n_test = max(1, int(n * 0.2))
        X_train, X_test = X[:-n_test], X[-n_test:]
        Y_train, Y_test = Y[:-n_test], Y[-n_test:]

        W_T, _, _, _ = lstsq(X_train, Y_train)
        W = W_T.T

        # Test error
        Y_pred = (W @ X_test.T).T
        test_err = np.linalg.norm(Y_test - Y_pred, "fro") / (np.linalg.norm(Y_test, "fro") + 1e-10)

        # Cosine similarity
        dots = np.sum(Y_test * Y_pred, axis=1)
        norms_y = np.linalg.norm(Y_test, axis=1)
        norms_p = np.linalg.norm(Y_pred, axis=1)
        cos = float(np.mean(dots / (norms_y * norms_p + 1e-10)))

        operators[perm] = W
        fit_info[perm] = {"test_error": test_err, "cosine": cos, "parity": perm.parity}

    if verbose:
        print(f"  {'Permutation':>12s}  {'Parity':>6s}  {'TestErr':>8s}  {'Cosine':>8s}")
        print("  " + "─" * 45)
        for perm, info in sorted(fit_info.items(), key=lambda x: str(x[0])):
            p = "even" if info["parity"] == 0 else "odd"
            print(f"  {str(perm):>12s}  {p:>6s}  {info['test_error']:>8.4f}  {info['cosine']:>8.4f}")

    return operators, fit_info, indices


def test_group_relations(operators, verbose=True):
    """Test S_3 group relations in feature space."""
    group = PermutationGroup(3)
    gens = group.generators()
    W_s = operators[gens["s"]]
    W_r = operators[gens["r"]]

    s3_rels = s3_relation_errors(W_s, W_r)

    # Composition errors by parity
    even_comp = []
    odd_comp = []
    mixed_comp = []

    for p1 in group.elements:
        for p2 in group.elements:
            if p1.is_identity or p2.is_identity:
                continue
            composed = p1 * p2
            if composed in operators and p1 in operators and p2 in operators:
                err = composition_error(operators[p1], operators[p2], operators[composed])
                if p1.is_even and p2.is_even:
                    even_comp.append(err.relative_error)
                elif not p1.is_even and not p2.is_even:
                    odd_comp.append(err.relative_error)
                else:
                    mixed_comp.append(err.relative_error)

    # Inverse errors
    inv_errors = []
    for perm in operators:
        if perm.is_identity:
            continue
        perm_inv = perm.inverse()
        if perm_inv in operators:
            inv_err = inverse_error(operators[perm], operators[perm_inv])
            inv_errors.append(inv_err.relative_error)

    if verbose:
        print(f"\n--- Group Relations in Feature Space ---")
        print(f"  S_3 presentation relations:")
        for name, r in s3_rels.items():
            print(f"    {name}: relative_error = {r.relative_error:.4f}")
        print(f"\n  Composition errors:")
        print(f"    Even×Even: {np.mean(even_comp):.4f}" if even_comp else "    Even×Even: N/A")
        print(f"    Odd×Odd:   {np.mean(odd_comp):.4f}" if odd_comp else "    Odd×Odd: N/A")
        print(f"    Mixed:     {np.mean(mixed_comp):.4f}" if mixed_comp else "    Mixed: N/A")
        print(f"    All:       {np.mean(even_comp + odd_comp + mixed_comp):.4f}")
        print(f"\n  Mean inverse error: {np.mean(inv_errors):.4f}" if inv_errors else "")

        # Determinants
        det_s = np.linalg.det(W_s) if W_s.shape[0] <= 50 else None
        det_r = np.linalg.det(W_r) if W_r.shape[0] <= 50 else None
        if det_s is not None:
            print(f"\n  det(W_s) = {det_s:.6f} (want ±1)")
            print(f"  det(W_r) = {det_r:.6f} (want ±1)")

    return s3_rels, even_comp, odd_comp, mixed_comp


def main():
    parser = argparse.ArgumentParser(description="Analyze group structure in SAE feature space")
    parser.add_argument("--sae-path", type=str, required=True, help="Path to trained SAE")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to pre-cached activations (.npz). Avoids re-running model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--top-k", type=int, nargs="+", default=[0, 100, 50, 20, 10],
                        help="Feature subset sizes (0 = all active features)")
    args = parser.parse_args()

    # Load activations (from cache or by running model)
    if args.cache:
        print(f"Loading cached activations from: {args.cache}")
        samples, perm_samples, prompt_cache = load_cached_activations(args.cache, args.layer)
        print(f"  {len(prompt_cache)} prompts, {len(samples)} samples")
    else:
        print(f"Loading model: {args.model}")
        model = load_model(model_name=args.model, device=args.device)
        print(f"Loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
        samples, perm_samples, prompt_cache = collect_activations_for_layer(
            model, args.layer
        )

    print(f"\nLoading SAE from: {args.sae_path}")
    sae = load_sae(args.sae_path, device=args.device)
    print(f"SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Encode through SAE
    print(f"\nEncoding {len(prompt_cache)} activations through SAE...")
    sae_features = encode_activations(sae, prompt_cache)

    # Sparsity analysis
    analyze_sparsity(sae_features, samples, perm_samples)

    # Fit operators and test group relations at various feature subset sizes
    for top_k in args.top_k:
        k = None if top_k == 0 else top_k
        print(f"\n{'='*80}")
        if k is None:
            print(f"Using ALL active features")
        else:
            print(f"Using top-{k} most permutation-sensitive features")
        print(f"{'='*80}")

        operators, fit_info, indices = fit_operators_in_feature_space(
            sae_features, perm_samples, top_k=k
        )
        test_group_relations(operators)


if __name__ == "__main__":
    main()
