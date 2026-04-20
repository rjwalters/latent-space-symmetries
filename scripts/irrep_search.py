"""Search for S_3 irreducible representations in activation space.

Instead of PCA (max variance), search for a 2D subspace where the
permutation operators are closest to orthogonal matrices — i.e., where
they most closely resemble the standard 2D irrep of S_3.

The standard 2D irrep of S_3:
  s = (0,1) -> [[1, 0], [0, -1]]   (reflection)
  r = (0,1,2) -> [[-1/2, -√3/2], [√3/2, -1/2]]   (120° rotation)

Strategy:
1. Fit full-rank operators W_π in d_model space
2. Use Procrustes-like optimization to find a 2D projection where
   the projected operators best approximate the standard irrep
3. Also try: find subspace where operators are most orthogonal
   (det closest to ±1, W^T W closest to I)

Additionally, check for the sign representation (1D):
  s -> -1, r -> +1
by looking for a direction where s flips sign and r preserves it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import lstsq, polar
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation.capture import load_model, cache_activations, DEFAULT_MODEL
from src.datasets.permutations import Permutation, PermutationGroup
from src.datasets.prompts import generate_bulk_samples


# Standard 2D irrep matrices for S_3
S3_STANDARD_IRREP = {
    "s": np.array([[1, 0], [0, -1]], dtype=float),
    "r": np.array([[-0.5, -np.sqrt(3)/2], [np.sqrt(3)/2, -0.5]], dtype=float),
}
# All 6 elements
def _build_standard_irrep():
    s = S3_STANDARD_IRREP["s"]
    r = S3_STANDARD_IRREP["r"]
    I = np.eye(2)
    r2 = r @ r
    return {
        Permutation.identity(3): I,
        Permutation.transposition(3, 0, 1): s,
        Permutation.cycle(3, 0, 1, 2): r,
        Permutation.cycle(3, 0, 2, 1): r2,
        Permutation.transposition(3, 1, 2): s @ r,   # = sr
        Permutation.transposition(3, 0, 2): s @ r2,  # = sr²
    }

STANDARD_IRREP = _build_standard_irrep()


def collect_and_fit(model, layer, verbose=True):
    """Collect activations and fit full-rank operators."""
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

    # Fit operators
    d = model.cfg.d_model
    operators = {Permutation.identity(3): np.eye(d)}
    for perm, samps in perm_samples.items():
        X = np.stack([prompt_cache[s["base_prompt"]] for s in samps])
        Y = np.stack([prompt_cache[s["permuted_prompt"]] for s in samps])
        W_T, _, _, _ = lstsq(X, Y)
        operators[perm] = W_T.T

    return operators, prompt_cache, perm_samples


def search_sign_representation(operators, d_model, verbose=True):
    """Search for a 1D subspace (direction) where s -> -1, r -> +1.

    This is the sign representation of S_3.
    Find direction v such that W_s v ≈ -v and W_r v ≈ v.
    Equivalently, v is an eigenvector of W_s with eigenvalue -1
    and of W_r with eigenvalue +1.
    """
    group = PermutationGroup(3)
    gens = group.generators()
    W_s = operators[gens["s"]]
    W_r = operators[gens["r"]]

    # Find eigenvectors of W_s near eigenvalue -1
    eigvals_s, eigvecs_s = np.linalg.eig(W_s)
    # Sort by closeness to -1
    dists_to_minus1 = np.abs(eigvals_s - (-1))
    idx = np.argsort(dists_to_minus1.real)

    if verbose:
        print(f"\n--- Sign Representation Search ---")
        print(f"  Looking for direction v where W_s v ≈ -v, W_r v ≈ +v")
        print(f"\n  W_s eigenvalues closest to -1:")
        for i in range(min(5, len(idx))):
            j = idx[i]
            ev = eigvals_s[j]
            v = eigvecs_s[:, j].real
            v = v / np.linalg.norm(v)
            # Check W_r v
            Wrv = W_r @ v
            r_eigenvalue = np.dot(Wrv, v) / np.dot(v, v)
            r_residual = np.linalg.norm(Wrv - r_eigenvalue * v) / np.linalg.norm(v)
            print(f"    λ_s = {ev.real:+.6f}, W_r·v ratio = {r_eigenvalue:.4f}, "
                  f"W_r residual = {r_residual:.4f}")

    # Try optimization: find v that minimizes ||W_s v + v||² + ||W_r v - v||²
    def objective(v_flat):
        v = v_flat / (np.linalg.norm(v_flat) + 1e-10)
        err_s = np.linalg.norm(W_s @ v + v)**2
        err_r = np.linalg.norm(W_r @ v - v)**2
        return err_s + err_r

    # Try multiple random initializations
    best_loss = float("inf")
    best_v = None
    for trial in range(20):
        v0 = np.random.randn(d_model)
        v0 /= np.linalg.norm(v0)
        result = minimize(objective, v0, method="L-BFGS-B", options={"maxiter": 500})
        if result.fun < best_loss:
            best_loss = result.fun
            best_v = result.x / np.linalg.norm(result.x)

    # Verify
    Wsv = W_s @ best_v
    Wrv = W_r @ best_v
    s_ratio = np.dot(Wsv, best_v)
    r_ratio = np.dot(Wrv, best_v)
    s_residual = np.linalg.norm(Wsv - s_ratio * best_v)
    r_residual = np.linalg.norm(Wrv - r_ratio * best_v)

    if verbose:
        print(f"\n  Best optimized sign-rep direction:")
        print(f"    W_s·v ratio: {s_ratio:.6f} (want -1)")
        print(f"    W_r·v ratio: {r_ratio:.6f} (want +1)")
        print(f"    W_s residual: {s_residual:.6f}")
        print(f"    W_r residual: {r_residual:.6f}")
        print(f"    Objective: {best_loss:.6f}")

        # Check all group elements
        print(f"\n  All permutations applied to sign-rep direction:")
        for perm in group.elements:
            Wv = operators[perm] @ best_v
            ratio = np.dot(Wv, best_v)
            residual = np.linalg.norm(Wv - ratio * best_v)
            expected = 1 if perm.is_even else -1
            parity = "even" if perm.is_even else "odd"
            print(f"    {str(perm):>12s} ({parity}): ratio={ratio:+.4f} "
                  f"(want {expected:+d}), residual={residual:.4f}")

    return best_v, best_loss


def search_standard_irrep(operators, d_model, verbose=True):
    """Search for a 2D subspace carrying the standard irrep of S_3.

    Optimize a projection matrix P (d_model, 2) such that
    P^T W_π P ≈ ρ(π) for all π, where ρ is the standard 2D irrep.
    """
    group = PermutationGroup(3)

    if verbose:
        print(f"\n--- Standard 2D Irrep Search ---")

    # Optimization: find P such that sum_π ||P^T W_π P - ρ(π)||² is minimized
    # P is d_model × 2 with orthonormal columns
    # Parameterize as P = Q[:, :2] where Q comes from QR of random matrix

    def objective(params):
        # params is a flattened d_model × 2 matrix
        P = params.reshape(d_model, 2)
        # Orthonormalize
        Q, R = np.linalg.qr(P)
        P_orth = Q[:, :2]

        total_err = 0.0
        for perm in group.elements:
            if perm.is_identity:
                continue
            W = operators[perm]
            projected = P_orth.T @ W @ P_orth  # (2, 2)
            target = STANDARD_IRREP[perm]
            total_err += np.linalg.norm(projected - target, "fro")**2
        return total_err

    def objective_with_grad(params):
        """Use finite differences for gradient."""
        return objective(params)

    best_loss = float("inf")
    best_P = None

    for trial in range(30):
        P0 = np.random.randn(d_model, 2)
        Q0, _ = np.linalg.qr(P0)
        P0 = Q0[:, :2]

        result = minimize(
            objective_with_grad,
            P0.flatten(),
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_P = result.x.reshape(d_model, 2)

    # Orthonormalize result
    Q, R = np.linalg.qr(best_P)
    P = Q[:, :2]

    if verbose:
        print(f"  Optimization objective (sum of squared Frobenius errors): {best_loss:.6f}")
        print(f"\n  Projected operators vs standard irrep:")
        for perm in group.elements:
            if perm.is_identity:
                continue
            W = operators[perm]
            projected = P.T @ W @ P
            target = STANDARD_IRREP[perm]
            err = np.linalg.norm(projected - target, "fro")
            parity = "even" if perm.is_even else "odd"
            print(f"\n    {str(perm):>12s} ({parity}):")
            print(f"      Projected: [{projected[0,0]:+.4f}, {projected[0,1]:+.4f}]")
            print(f"                 [{projected[1,0]:+.4f}, {projected[1,1]:+.4f}]")
            print(f"      Target:    [{target[0,0]:+.4f}, {target[0,1]:+.4f}]")
            print(f"                 [{target[1,0]:+.4f}, {target[1,1]:+.4f}]")
            print(f"      Error:     {err:.4f}")

        # Check group relations in projected space
        gens = group.generators()
        W_s_proj = P.T @ operators[gens["s"]] @ P
        W_r_proj = P.T @ operators[gens["r"]] @ P
        I2 = np.eye(2)

        s2_err = np.linalg.norm(W_s_proj @ W_s_proj - I2, "fro") / np.linalg.norm(I2, "fro")
        r3_err = np.linalg.norm(W_r_proj @ W_r_proj @ W_r_proj - I2, "fro") / np.linalg.norm(I2, "fro")
        srs_target = W_r_proj @ W_r_proj  # r⁻¹ = r²
        srs_err = np.linalg.norm(W_s_proj @ W_r_proj @ W_s_proj - srs_target, "fro") / np.linalg.norm(srs_target, "fro")

        print(f"\n  Group relations in projected space:")
        print(f"    s² = e:    error = {s2_err:.4f}")
        print(f"    r³ = e:    error = {r3_err:.4f}")
        print(f"    srs = r⁻¹: error = {srs_err:.4f}")

        det_s = np.linalg.det(W_s_proj)
        det_r = np.linalg.det(W_r_proj)
        print(f"    det(s) = {det_s:.4f} (want -1)")
        print(f"    det(r) = {det_r:.4f} (want +1)")

    return P, best_loss


def main():
    parser = argparse.ArgumentParser(description="Search for S_3 irreps in activation space")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 10, 16, 20, 21, 22])
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    d = model.cfg.d_model
    print(f"Loaded. {model.cfg.n_layers} layers, d_model={d}")

    for layer in args.layers:
        print(f"\n{'='*80}")
        print(f"LAYER {layer}")
        print(f"{'='*80}")

        operators, prompt_cache, perm_samples = collect_and_fit(model, layer)

        # Search for sign representation
        sign_v, sign_loss = search_sign_representation(operators, d)

        # Search for standard 2D irrep
        irrep_P, irrep_loss = search_standard_irrep(operators, d)


if __name__ == "__main__":
    main()
