"""TransformerLens-based activation capture for Qwen2.5-1.5B.

Handles model loading, activation caching across all layers, and
structured storage of activation tensors paired with prompt metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import transformer_lens as tl

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# Qwen2.5-1.5B architecture constants
N_LAYERS = 28
D_MODEL = 1536
N_HEADS = 12
N_KV_HEADS = 2


def load_model(device: str = "cpu") -> tl.HookedTransformer:
    """Load Qwen2.5-1.5B into TransformerLens."""
    model = tl.HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    return model


HookPoint = Literal["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]


@dataclass
class CachedActivations:
    """Activations from a single forward pass, organized by hook point."""

    prompt: str
    tokens: torch.Tensor  # (seq_len,)
    activations: dict[str, torch.Tensor]  # hook_name -> (n_layers, seq_len, d_model)
    logits: torch.Tensor  # (seq_len, vocab_size)


def cache_activations(
    model: tl.HookedTransformer,
    prompt: str,
    hook_points: list[HookPoint] | None = None,
    layers: list[int] | None = None,
) -> CachedActivations:
    """Run a prompt through the model and cache activations at specified hook points.

    Args:
        model: A loaded HookedTransformer.
        prompt: The input text.
        hook_points: Which activation types to cache.
            Defaults to ["resid_pre", "resid_mid", "resid_post"].
        layers: Which layers to cache. Defaults to all layers.

    Returns:
        CachedActivations with structured access to all cached tensors.
    """
    if hook_points is None:
        hook_points = ["resid_pre", "resid_mid", "resid_post"]
    if layers is None:
        layers = list(range(N_LAYERS))

    # Build the list of hook names TransformerLens expects
    names_filter = []
    for hp in hook_points:
        for layer in layers:
            if hp in ("resid_pre", "resid_mid", "resid_post"):
                names_filter.append(f"blocks.{layer}.hook_{hp}")
            elif hp == "attn_out":
                names_filter.append(f"blocks.{layer}.attn.hook_result")
            elif hp == "mlp_out":
                names_filter.append(f"blocks.{layer}.hook_mlp_out")

    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens, names_filter=names_filter)

    # Organize by hook point type
    activations = {}
    for hp in hook_points:
        layer_acts = []
        for layer in layers:
            if hp in ("resid_pre", "resid_mid", "resid_post"):
                key = f"blocks.{layer}.hook_{hp}"
            elif hp == "attn_out":
                key = f"blocks.{layer}.attn.hook_result"
            elif hp == "mlp_out":
                key = f"blocks.{layer}.hook_mlp_out"
            layer_acts.append(cache[key].squeeze(0))  # remove batch dim
        activations[hp] = torch.stack(layer_acts)  # (n_layers, seq_len, d_model)

    return CachedActivations(
        prompt=prompt,
        tokens=tokens.squeeze(0),
        activations=activations,
        logits=logits.squeeze(0),
    )


def cache_prompt_pair(
    model: tl.HookedTransformer,
    prompt_original: str,
    prompt_permuted: str,
    **kwargs,
) -> tuple[CachedActivations, CachedActivations]:
    """Cache activations for an original/permuted prompt pair."""
    act_orig = cache_activations(model, prompt_original, **kwargs)
    act_perm = cache_activations(model, prompt_permuted, **kwargs)
    return act_orig, act_perm
