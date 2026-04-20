"""Deep dive into the low-dimensional permutation subspace.

Analyzes the structure of the 3-5D PCA subspace where group relations
are partially satisfied. Key questions:

1. What do the PCA directions correspond to?
2. Are the projected operators close to known S_3 irreps?
3. Detailed composition table — which pairs compose well/poorly?
4. Spectral structure of the operators (eigenvalues, orthogonality)
5. Geometry of projected activation differences by permutation
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


def pca_subspace(samples, prompt_cache, layer, n_components):
    """Get PCA subspace from activation differences."""
    diffs = []
    for s in samples:
        base_act = prompt_cache[s["base_prompt"]][layer]
        perm_act = prompt_cache[s["permuted_prompt"]][layer]
        diffs.append(perm_act - base_act)
    diffs = np.stack(diffs)
    mean_diff = diffs.mean(axis=0)
    diffs_centered = diffs - mean_diff
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    return Vt[:n_components], S, mean_diff


def fit_subspace_operator(base_acts, perm_acts, V):
    """Fit operator in PCA subspace. No test split — use all data."""
    X = base_acts @ V.T
    Y = perm_acts @ V.T
    W_T, _, _, _ = lstsq(X, Y)
    return W_T.T


def analyze_operator_structure(operators, group, verbose=True):
    """Analyze spectral and algebraic properties of fitted operators."""
    gens = group.generators()
    s_perm = gens["s"]
    r_perm = gens["r"]

    W_s = operators[s_perm]
    W_r = operators[r_perm]
    d = W_s.shape[0]
    I = np.eye(d)

    if verbose:
        print(f"\n--- Operator Spectral Analysis (dim={d}) ---")

    results = {}
    for perm in group.elements:
        if perm.is_identity:
            continue
        W = operators[perm]
        eigvals = np.linalg.eigvals(W)
        # Sort by real part descending
        idx = np.argsort(-eigvals.real)
        eigvals = eigvals[idx]

        # Check orthogonality: W^T W ≈ I ?
        WtW = W.T @ W
        orthogonality_error = np.linalg.norm(WtW - I, "fro") / np.linalg.norm(I, "fro")

        # Determinant
        det = np.linalg.det(W)

        # Check if W is close to an involution (W² ≈ I) for transpositions
        W2_err = np.linalg.norm(W @ W - I, "fro") / np.linalg.norm(I, "fro")

        parity_str = "even" if perm.is_even else "odd"
        results[str(perm)] = {
            "eigenvalues": eigvals.tolist(),
            "det": complex(det),
            "orthogonality_error": float(orthogonality_error),
            "involution_error": float(W2_err),
            "parity": perm.parity,
        }

        if verbose:
            eig_str = ", ".join(
                f"{e.real:.3f}{'+' if e.imag >= 0 else ''}{e.imag:.3f}j"
                if abs(e.imag) > 0.01 else f"{e.real:.4f}"
                for e in eigvals
            )
            print(f"\n  {str(perm):>12s} ({parity_str})")
            print(f"    eigenvalues: [{eig_str}]")
            print(f"    det:         {det.real:.4f}")
            print(f"    orthog err:  {orthogonality_error:.4f}")
            print(f"    W²≈I err:    {W2_err:.4f}")

    # Check known S_3 irrep structure
    # The standard 2D irrep has s -> reflection, r -> rotation by 2π/3
    # In that rep: det(s) = -1, det(r) = 1
    # Eigenvalues of s: {1, -1}, eigenvalues of r: {e^{2πi/3}, e^{-2πi/3}}
    if verbose:
        print(f"\n--- Comparison with S_3 irreps ---")
        print(f"  Standard 2D irrep expectations:")
        print(f"    s: eigenvalues {{1, -1}}, det = -1")
        print(f"    r: eigenvalues {{e^(2πi/3), e^(-2πi/3)}}, det = 1")
        print(f"  Sign rep (1D): s -> -1, r -> 1")
        print(f"  Trivial rep (1D): s -> 1, r -> 1")

    return results


def detailed_composition_table(operators, group, verbose=True):
    """Full composition error table for all pairs."""
    I = np.eye(list(operators.values())[0].shape[0])

    if verbose:
        print(f"\n--- Full Composition Table ---")
        print(f"  Error = ||W_p1 W_p2 - W_(p1*p2)||_F / ||W_(p1*p2)||_F")

    elements = [p for p in group.elements if not p.is_identity]
    labels = [str(p) for p in elements]

    table = np.zeros((len(elements), len(elements)))

    for i, p1 in enumerate(elements):
        for j, p2 in enumerate(elements):
            composed = p1 * p2
            if composed.is_identity:
                W_target = I
            else:
                W_target = operators[composed]
            W_product = operators[p1] @ operators[p2]
            err = np.linalg.norm(W_product - W_target, "fro") / (np.linalg.norm(W_target, "fro") + 1e-10)
            table[i, j] = err

    if verbose:
        # Print table
        col_header = "p1\\p2"
        header = f"{col_header:>12s}  " + "  ".join(f"{l:>10s}" for l in labels)
        print(f"\n{header}")
        print("─" * len(header))
        for i, label in enumerate(labels):
            row = "  ".join(f"{table[i, j]:>10.4f}" for j in range(len(elements)))
            parity = "E" if elements[i].is_even else "O"
            print(f"{label:>10s}({parity})  {row}")

        # Analyze by parity pattern
        print(f"\n--- Composition by parity of operands ---")
        patterns = {"E×E": [], "O×O": [], "E×O": [], "O×E": []}
        for i, p1 in enumerate(elements):
            for j, p2 in enumerate(elements):
                k1 = "E" if p1.is_even else "O"
                k2 = "E" if p2.is_even else "O"
                patterns[f"{k1}×{k2}"].append(table[i, j])

        for pattern, errors in patterns.items():
            composed_parity = {
                "E×E": "→E", "O×O": "→E", "E×O": "→O", "O×E": "→O"
            }[pattern]
            print(f"  {pattern} {composed_parity}: mean={np.mean(errors):.4f}, "
                  f"std={np.std(errors):.4f}, "
                  f"min={np.min(errors):.4f}, max={np.max(errors):.4f}")

    return table, labels


def analyze_pca_directions(V, model, verbose=True):
    """Understand what the PCA directions correspond to.

    Check correlation with entity token embeddings.
    """
    if verbose:
        print(f"\n--- PCA Direction Analysis ---")

    # Get entity token embeddings
    entity_names = ["Alice", "Bob", "Carol"]
    embed = model.W_E.detach().cpu().numpy()  # (vocab_size, d_model)

    entity_embeds = {}
    for name in entity_names:
        tokens = model.to_tokens(name, prepend_bos=False)
        # Take first token if multi-token
        tok_id = tokens[0, 0].item()
        entity_embeds[name] = embed[tok_id]

    # Project entity embeddings onto PCA directions
    if verbose:
        print(f"\n  Entity embedding projections onto PCA directions:")
        print(f"  {'':>8s}  " + "  ".join(f"{'PC'+str(i):>8s}" for i in range(V.shape[0])))
        for name, emb in entity_embeds.items():
            projs = V @ emb
            proj_str = "  ".join(f"{p:>8.3f}" for p in projs)
            print(f"  {name:>8s}  {proj_str}")

    # Compute pairwise differences in PCA space
    if verbose:
        print(f"\n  Entity difference vectors in PCA space:")
        pairs = [("Alice", "Bob"), ("Alice", "Carol"), ("Bob", "Carol")]
        for n1, n2 in pairs:
            diff = entity_embeds[n1] - entity_embeds[n2]
            proj = V @ diff
            proj_str = "  ".join(f"{p:>8.3f}" for p in proj)
            print(f"  {n1}-{n2}:  {proj_str}")

    # Check if PCA directions align with any specific embedding differences
    # Compute cosine similarity between PCA dirs and entity difference dirs
    if verbose:
        print(f"\n  Cosine similarity: PCA directions vs entity differences:")
        diffs_dict = {}
        for n1, n2 in [("Alice", "Bob"), ("Alice", "Carol"), ("Bob", "Carol")]:
            diff = entity_embeds[n1] - entity_embeds[n2]
            diff_norm = diff / (np.linalg.norm(diff) + 1e-10)
            diffs_dict[f"{n1}-{n2}"] = diff_norm

        print(f"  {'':>8s}  " + "  ".join(f"{k:>12s}" for k in diffs_dict.keys()))
        for i in range(V.shape[0]):
            pc_dir = V[i] / (np.linalg.norm(V[i]) + 1e-10)
            cos_sims = [float(np.dot(pc_dir, d)) for d in diffs_dict.values()]
            cos_str = "  ".join(f"{c:>12.4f}" for c in cos_sims)
            print(f"  {'PC'+str(i):>8s}  {cos_str}")

    return entity_embeds


def analyze_activation_geometry(samples, perm_samples, prompt_cache, layer, V, verbose=True):
    """Analyze the geometry of activation differences in PCA space."""
    if verbose:
        print(f"\n--- Activation Geometry in PCA Space ---")

    # Project all differences into PCA space, grouped by permutation
    perm_projections = {}
    group = PermutationGroup(3)

    for perm in group.elements:
        if perm.is_identity:
            continue
        samps = perm_samples[perm]
        projs = []
        for s in samps:
            base = prompt_cache[s["base_prompt"]][layer]
            perm_a = prompt_cache[s["permuted_prompt"]][layer]
            diff = perm_a - base
            proj = V @ diff  # (n_dim,)
            projs.append(proj)
        perm_projections[perm] = np.stack(projs)

    if verbose:
        print(f"\n  Mean activation difference per permutation (in PCA space):")
        print(f"  {'Perm':>12s} {'parity':>6s}  " +
              "  ".join(f"{'PC'+str(i):>8s}" for i in range(V.shape[0])) +
              f"  {'||mean||':>8s}")
        for perm in group.elements:
            if perm.is_identity:
                continue
            projs = perm_projections[perm]
            mean = projs.mean(axis=0)
            norm = np.linalg.norm(mean)
            parity = "even" if perm.is_even else "odd"
            mean_str = "  ".join(f"{m:>8.3f}" for m in mean)
            print(f"  {str(perm):>12s} {parity:>6s}  {mean_str}  {norm:>8.3f}")

        # Spread (std) per permutation
        print(f"\n  Std of activation differences per permutation:")
        print(f"  {'Perm':>12s} {'parity':>6s}  " +
              "  ".join(f"{'PC'+str(i):>8s}" for i in range(V.shape[0])))
        for perm in group.elements:
            if perm.is_identity:
                continue
            projs = perm_projections[perm]
            std = projs.std(axis=0)
            parity = "even" if perm.is_even else "odd"
            std_str = "  ".join(f"{s:>8.3f}" for s in std)
            print(f"  {str(perm):>12s} {parity:>6s}  {std_str}")

        # Pairwise cosine similarity of mean difference vectors
        print(f"\n  Cosine similarity between mean difference vectors:")
        perms = [p for p in group.elements if not p.is_identity]
        means = {p: perm_projections[p].mean(axis=0) for p in perms}

        labels = [str(p) for p in perms]
        header = f"{'':>12s}  " + "  ".join(f"{l:>10s}" for l in labels)
        print(f"  {header}")
        for i, p1 in enumerate(perms):
            row = []
            for j, p2 in enumerate(perms):
                m1 = means[p1]
                m2 = means[p2]
                cos = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-10)
                row.append(f"{cos:>10.4f}")
            print(f"  {labels[i]:>12s}  " + "  ".join(row))

    return perm_projections


def main():
    parser = argparse.ArgumentParser(description="Deep dive into permutation subspace")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 21])
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    print(f"Loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    samples, perm_samples, prompt_cache = collect_activations(
        model, args.layers, verbose=True
    )

    group = PermutationGroup(3)

    for layer in args.layers:
        print(f"\n{'='*80}")
        print(f"LAYER {layer}, SUBSPACE DIM={args.dim}")
        print(f"{'='*80}")

        V, sv, mean_diff = pca_subspace(samples, prompt_cache, layer, args.dim)

        # Variance explained
        total_var = np.sum(sv**2)
        var_explained = np.sum(sv[:args.dim]**2) / total_var
        print(f"\nVariance explained by top {args.dim} PCs: {var_explained:.4f}")

        # 1. What do the PCA directions correspond to?
        analyze_pca_directions(V, model)

        # 2. Fit operators in subspace
        print(f"\n--- Fitting Operators in {args.dim}D Subspace ---")
        operators = {}
        operators[Permutation.identity(3)] = np.eye(args.dim)
        for perm, samps in perm_samples.items():
            base_acts = np.stack([prompt_cache[s["base_prompt"]][layer] for s in samps])
            perm_acts = np.stack([prompt_cache[s["permuted_prompt"]][layer] for s in samps])
            W = fit_subspace_operator(base_acts, perm_acts, V)
            operators[perm] = W

        # 3. Spectral analysis of operators
        analyze_operator_structure(operators, group)

        # 4. Detailed composition table
        detailed_composition_table(operators, group)

        # 5. Activation geometry
        analyze_activation_geometry(samples, perm_samples, prompt_cache, layer, V)

        # 6. Print actual operator matrices
        print(f"\n--- Operator Matrices (dim={args.dim}) ---")
        gens = group.generators()
        for label, perm in [("s=(0,1)", gens["s"]), ("r=(0,1,2)", gens["r"])]:
            W = operators[perm]
            print(f"\n  W_{label}:")
            for row in W:
                print(f"    [{', '.join(f'{v:>8.4f}' for v in row)}]")

        # Check: is there a basis where W_s is diagonal with ±1 and W_r is a rotation?
        # Diagonalize W_s
        eigvals_s, eigvecs_s = np.linalg.eig(operators[gens["s"]])
        print(f"\n  W_s eigenvalues: {[f'{e.real:.4f}' for e in eigvals_s]}")

        # Transform W_r into W_s eigenbasis
        P = eigvecs_s
        if np.abs(np.linalg.det(P)) > 1e-6:
            P_inv = np.linalg.inv(P)
            W_r_in_s_basis = P_inv @ operators[gens["r"]] @ P
            W_s_in_s_basis = P_inv @ operators[gens["s"]] @ P
            print(f"\n  W_s in its own eigenbasis (should be diagonal):")
            for row in W_s_in_s_basis.real:
                print(f"    [{', '.join(f'{v:>8.4f}' for v in row)}]")
            print(f"\n  W_r in W_s eigenbasis:")
            for row in W_r_in_s_basis.real:
                print(f"    [{', '.join(f'{v:>8.4f}' for v in row)}]")


if __name__ == "__main__":
    main()
