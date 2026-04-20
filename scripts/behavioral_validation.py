"""Phase 3: Behavioral validation of permutation symmetry.

Tests whether model outputs are consistent under permutation of entities.
For each prompt family, generate all S_n permutations and compare the model's
next-token predictions across variants.

Key questions:
- Do permuted prompts produce permuted completions?
- Is the model's confidence stable across permutations?
- Are there asymmetries correlated with parity (even vs odd)?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation.capture import load_model, DEFAULT_MODEL
from src.datasets.prompts import (
    ALL_FAMILIES,
    PromptFamily,
    generate_permuted_prompts,
)


def get_top_completions(
    model, prompt: str, k: int = 10
) -> list[tuple[str, float]]:
    """Return top-k next-token completions with probabilities."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    last_logits = logits[0, -1, :]  # (vocab_size,)
    probs = F.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, k)
    results = []
    for idx, prob in zip(topk.indices, topk.values):
        token_str = model.to_string(idx.unsqueeze(0))
        results.append((token_str, prob.item()))
    return results


def compute_kl_divergence(model, prompt_a: str, prompt_b: str) -> float:
    """KL(P_a || P_b) over full next-token distributions."""
    tokens_a = model.to_tokens(prompt_a)
    tokens_b = model.to_tokens(prompt_b)
    with torch.no_grad():
        logits_a = model(tokens_a)[0, -1, :]
        logits_b = model(tokens_b)[0, -1, :]
    log_p_a = F.log_softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)
    return F.kl_div(F.log_softmax(logits_b, dim=-1), F.softmax(logits_a, dim=-1),
                    reduction="sum").item()


def validate_family(model, family: PromptFamily, verbose: bool = True) -> dict:
    """Run behavioral validation for one prompt family."""
    variants = generate_permuted_prompts(family)
    base = next(v for v in variants if v["is_identity"])
    base_prompt = base["prompt"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Family: {family.name} (S_{family.n_entities})")
        print(f"Base:   {base_prompt}")
        print(f"{'='*60}")

    # Get base completions
    base_completions = get_top_completions(model, base_prompt)
    if verbose:
        print(f"\nBase top-5 completions:")
        for tok, prob in base_completions[:5]:
            print(f"  {tok!r:>15s}  {prob:.4f}")

    results = []
    for variant in variants:
        perm = variant["permutation"]
        prompt = variant["prompt"]
        completions = get_top_completions(model, prompt)

        # Compare top-1 tokens
        top1_tok, top1_prob = completions[0]
        base_top1_tok, base_top1_prob = base_completions[0]

        # Compute KL divergence from base
        tokens_base = model.to_tokens(base_prompt)
        tokens_var = model.to_tokens(prompt)
        with torch.no_grad():
            logits_base = model(tokens_base)[0, -1, :]
            logits_var = model(tokens_var)[0, -1, :]
        log_p_base = F.log_softmax(logits_base, dim=-1)
        log_p_var = F.log_softmax(logits_var, dim=-1)
        p_base = F.softmax(logits_base, dim=-1)
        p_var = F.softmax(logits_var, dim=-1)
        kl = F.kl_div(log_p_var, p_base, reduction="sum").item()

        # Top-k overlap
        base_top_tokens = {t for t, _ in base_completions}
        var_top_tokens = {t for t, _ in completions}
        overlap = len(base_top_tokens & var_top_tokens)

        result = {
            "permutation": str(perm),
            "parity": perm.parity,
            "is_identity": perm.is_identity,
            "prompt": prompt,
            "top1_token": top1_tok,
            "top1_prob": top1_prob,
            "kl_from_base": kl,
            "top10_overlap": overlap,
        }
        results.append(result)

        if verbose:
            parity_str = "even" if perm.parity == 0 else "odd "
            print(f"\n  {str(perm):>15s}  ({parity_str})  KL={kl:.4f}  "
                  f"top1={top1_tok!r} ({top1_prob:.4f})  "
                  f"top10_overlap={overlap}/10")

    # Summary statistics
    non_identity = [r for r in results if not r["is_identity"]]
    even = [r for r in non_identity if r["parity"] == 0]
    odd = [r for r in non_identity if r["parity"] == 1]

    summary = {
        "family": family.name,
        "n_entities": family.n_entities,
        "n_permutations": len(results),
        "mean_kl": sum(r["kl_from_base"] for r in non_identity) / max(len(non_identity), 1),
        "mean_kl_even": sum(r["kl_from_base"] for r in even) / max(len(even), 1) if even else None,
        "mean_kl_odd": sum(r["kl_from_base"] for r in odd) / max(len(odd), 1) if odd else None,
        "mean_top10_overlap": sum(r["top10_overlap"] for r in non_identity) / max(len(non_identity), 1),
    }

    if verbose:
        print(f"\n  Summary:")
        print(f"    Mean KL from base:  {summary['mean_kl']:.4f}")
        if summary["mean_kl_even"] is not None:
            print(f"    Mean KL (even):     {summary['mean_kl_even']:.4f}")
            print(f"    Mean KL (odd):      {summary['mean_kl_odd']:.4f}")
        print(f"    Mean top-10 overlap: {summary['mean_top10_overlap']:.1f}/10")

    return {"variants": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Behavioral validation of permutation symmetry")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--families", nargs="*", default=None,
                        help="Which families to test (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    print(f"Loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    families = ALL_FAMILIES
    if args.families:
        families = [f for f in ALL_FAMILIES if f.name in args.families]

    all_results = {}
    for family in families:
        result = validate_family(model, family, verbose=not args.quiet)
        all_results[family.name] = result

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-serializable fields
        serializable = {
            name: {
                "summary": data["summary"],
                "variants": [
                    {k: v for k, v in var.items()}
                    for var in data["variants"]
                ],
            }
            for name, data in all_results.items()
        }
        output_path.write_text(json.dumps(serializable, indent=2))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
