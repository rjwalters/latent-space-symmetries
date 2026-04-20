"""Causal intervention via activation patching with learned permutation operators.

After fitting W_π, we validate causally:
1. Take activations h(x) from running prompt x
2. Compute h'(x) = W_π h(x)
3. Splice h'(x) into the forward pass at the target layer
4. Check whether the output shifts toward the completion expected for π·x

This goes beyond correlational alignment to show the operator is
functionally meaningful in the model's computation.
"""

from __future__ import annotations

import torch
import numpy as np
import transformer_lens as tl


def patch_with_operator(
    model: tl.HookedTransformer,
    prompt: str,
    W_pi: np.ndarray,
    layer: int,
    hook_point: str = "resid_post",
    token_positions: list[int] | None = None,
) -> torch.Tensor:
    """Run prompt with learned operator applied at a specific layer.

    Args:
        model: The HookedTransformer.
        prompt: Input text.
        W_pi: (d_model, d_model) learned operator to apply.
        layer: Which layer to intervene at.
        hook_point: Which hook point type (resid_pre, resid_mid, resid_post).
        token_positions: Which token positions to modify. None = all.

    Returns:
        Modified logits tensor (seq_len, vocab_size).
    """
    W_tensor = torch.tensor(W_pi, dtype=torch.float32, device=model.cfg.device)

    def hook_fn(activation, hook):
        # activation shape: (batch, seq_len, d_model)
        if token_positions is not None:
            for pos in token_positions:
                activation[:, pos, :] = (W_tensor @ activation[:, pos, :].T).T
        else:
            # Apply to all positions
            activation = torch.einsum("ij,bsj->bsi", W_tensor, activation)
        return activation

    hook_name = f"blocks.{layer}.hook_{hook_point}"
    tokens = model.to_tokens(prompt)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    return logits.squeeze(0)


def measure_intervention_effect(
    model: tl.HookedTransformer,
    prompt_original: str,
    prompt_permuted: str,
    W_pi: np.ndarray,
    layer: int,
    hook_point: str = "resid_post",
    token_positions: list[int] | None = None,
    top_k: int = 10,
) -> dict:
    """Measure whether applying W_π shifts output toward the permuted completion.

    Compares:
    - Baseline: logits from running original prompt unmodified
    - Target: logits from running permuted prompt (ground truth)
    - Patched: logits from running original prompt with W_π applied

    Returns dict with KL divergences and top-token overlap metrics.
    """
    tokens_orig = model.to_tokens(prompt_original)
    tokens_perm = model.to_tokens(prompt_permuted)

    # Baseline
    logits_baseline = model(tokens_orig).squeeze(0)
    # Target
    logits_target = model(tokens_perm).squeeze(0)
    # Patched
    logits_patched = patch_with_operator(
        model, prompt_original, W_pi, layer, hook_point, token_positions
    )

    # Compare final token logits
    def softmax(logits):
        return torch.softmax(logits[-1], dim=-1)

    p_baseline = softmax(logits_baseline)
    p_target = softmax(logits_target)
    p_patched = softmax(logits_patched)

    # KL(patched || target) vs KL(baseline || target)
    def kl_div(p, q):
        return float(torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))))

    kl_baseline_target = kl_div(p_baseline, p_target)
    kl_patched_target = kl_div(p_patched, p_target)

    # Top-k overlap
    top_k_target = set(torch.topk(p_target, top_k).indices.tolist())
    top_k_patched = set(torch.topk(p_patched, top_k).indices.tolist())
    top_k_baseline = set(torch.topk(p_baseline, top_k).indices.tolist())

    return {
        "kl_baseline_to_target": kl_baseline_target,
        "kl_patched_to_target": kl_patched_target,
        "kl_reduction": kl_baseline_target - kl_patched_target,
        "top_k_overlap_patched": len(top_k_target & top_k_patched) / top_k,
        "top_k_overlap_baseline": len(top_k_target & top_k_baseline) / top_k,
    }
