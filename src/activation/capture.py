"""TransformerLens-based activation capture for Qwen2.5 models.

Handles model loading, activation caching across all layers, and
structured storage of activation tensors paired with prompt metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import transformer_lens as tl

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"

# Architecture constants per model variant
MODEL_CONFIGS = {
    "Qwen/Qwen2.5-0.5B": {"n_layers": 24, "d_model": 896, "n_heads": 14, "n_kv_heads": 2},
    "Qwen/Qwen2.5-1.5B": {"n_layers": 28, "d_model": 1536, "n_heads": 12, "n_kv_heads": 2},
}


def load_model(
    model_name: str = DEFAULT_MODEL, device: str = "cpu"
) -> tl.HookedTransformer:
    """Load a Qwen2.5 model into TransformerLens."""
    model = tl.HookedTransformer.from_pretrained(model_name, device=device)
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
        layers = list(range(model.cfg.n_layers))

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
