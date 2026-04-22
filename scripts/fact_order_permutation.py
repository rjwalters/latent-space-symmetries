"""Fact-order permutation experiment.

Tests whether the model treats different orderings of the same facts as
equivalent (S_n-invariant) or distinct (ordering encodes information).

This is the key experiment for the synthetic data critique: if the model
has learned that fact ordering encodes salience/priority/narrative frame,
then scrambling fact order in synthetic training data destroys information.

Levels of analysis:
1. Activation distance: how different are representations of the same facts
   in different orders? Compare to representations of genuinely different facts.
2. Behavioral: does the model's next-token prediction change with fact order?
3. Group structure: do fact-reordering operators compose as S_n?
   (We predict they won't, same as entity permutations.)

Prompt design:
- 2-fact sequences (S_2: two orderings)
- 3-fact sequences (S_3: six orderings)
- Facts are semantically independent to isolate ordering effects
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from itertools import permutations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation.capture import load_model, cache_activations, DEFAULT_MODEL
from src.datasets.permutations import Permutation, PermutationGroup
from src.operators.fitting import fit_operator
from src.operators.group_tests import (
    composition_error,
    s3_relation_errors,
)


# --- Fact sets ---
# Each entry is a list of independent facts. We'll test all orderings.

FACT_SETS_2 = [
    ["Alice loves Bob.", "Carol loves David."],
    ["The sky is blue.", "Water is wet."],
    ["Paris is in France.", "Tokyo is in Japan."],
    ["Dogs can swim.", "Birds can fly."],
    ["The book is red.", "The pen is black."],
    ["Monday comes before Tuesday.", "January comes before February."],
    ["Iron is a metal.", "Oxygen is a gas."],
    ["The cat sat on the mat.", "The dog lay by the fire."],
    ["She plays piano.", "He plays guitar."],
    ["The train was late.", "The bus was early."],
    ["Roses are red.", "Violets are blue."],
    ["The movie was long.", "The popcorn was stale."],
    ["Math requires logic.", "Art requires creativity."],
    ["The lake is deep.", "The mountain is tall."],
    ["Coffee is bitter.", "Honey is sweet."],
]

FACT_SETS_3 = [
    ["Alice loves Bob.", "Carol loves David.", "Eve loves Frank."],
    ["The sky is blue.", "Water is wet.", "Fire is hot."],
    ["Paris is in France.", "Tokyo is in Japan.", "Cairo is in Egypt."],
    ["Dogs can swim.", "Birds can fly.", "Fish can breathe underwater."],
    ["The book is red.", "The pen is black.", "The notebook is green."],
    ["Iron is a metal.", "Oxygen is a gas.", "Diamond is a crystal."],
    ["She plays piano.", "He plays guitar.", "They play drums."],
    ["The train was late.", "The bus was early.", "The taxi was on time."],
    ["Roses are red.", "Violets are blue.", "Sunflowers are yellow."],
    ["Coffee is bitter.", "Honey is sweet.", "Lemon is sour."],
    ["The room is on fire.", "There is a chair in the room.", "I am working on training LLMs in my room."],
    ["The cat is sleeping.", "The dishes need washing.", "A storm is approaching."],
    ["She won the race.", "He finished his thesis.", "They launched the satellite."],
    ["The bridge collapsed.", "Traffic was diverted.", "Repairs will take months."],
    ["The experiment failed.", "The grant was approved.", "A new student joined the lab."],
]


def generate_ordered_prompts(fact_set: list[str], n: int) -> list[dict]:
    """Generate all n! orderings of a fact set with permutation labels."""
    results = []
    identity = tuple(range(n))
    for perm_tuple in permutations(range(n)):
        ordered_facts = [fact_set[i] for i in perm_tuple]
        prompt = " ".join(ordered_facts)
        perm = Permutation(perm_tuple)
        results.append({
            "prompt": prompt,
            "permutation": perm,
            "perm_tuple": perm_tuple,
            "facts": ordered_facts,
            "is_identity": perm_tuple == identity,
        })
    return results


def measure_activation_distances(
    model, fact_sets: list[list[str]], layers: list[int], verbose=True,
) -> dict:
    """Measure how much activations change when facts are reordered.

    For each fact set, compute pairwise distances between all orderings
    at each layer. Compare to distances between DIFFERENT fact sets.
    """
    n = len(fact_sets[0])
    all_orderings = []
    for fs in fact_sets:
        orderings = generate_ordered_prompts(fs, n)
        all_orderings.append(orderings)

    # Cache all activations
    all_prompts = set()
    for orderings in all_orderings:
        for o in orderings:
            all_prompts.add(o["prompt"])

    if verbose:
        print(f"Caching {len(all_prompts)} prompts ({len(fact_sets)} fact sets × "
              f"{len(all_orderings[0])} orderings)...")
    prompt_cache = {}
    for prompt in tqdm(sorted(all_prompts), disable=not verbose, desc="Caching"):
        cached = cache_activations(
            model, prompt, hook_points=["resid_post"], layers=layers
        )
        resid = cached.activations["resid_post"]
        prompt_cache[prompt] = {
            layers[i]: resid[i, -1, :].detach().cpu().numpy()
            for i in range(len(layers))
        }

    results = {}
    for layer in layers:
        # Within-set distances (same facts, different order)
        within_distances = []
        for orderings in all_orderings:
            acts = [prompt_cache[o["prompt"]][layer] for o in orderings]
            for i in range(len(acts)):
                for j in range(i + 1, len(acts)):
                    dist = np.linalg.norm(acts[i] - acts[j])
                    norm = (np.linalg.norm(acts[i]) + np.linalg.norm(acts[j])) / 2
                    within_distances.append(dist / (norm + 1e-10))

        # Between-set distances (different facts)
        between_distances = []
        for i in range(len(all_orderings)):
            for j in range(i + 1, len(all_orderings)):
                # Compare identity orderings of different fact sets
                act_i = prompt_cache[all_orderings[i][0]["prompt"]][layer]
                act_j = prompt_cache[all_orderings[j][0]["prompt"]][layer]
                dist = np.linalg.norm(act_i - act_j)
                norm = (np.linalg.norm(act_i) + np.linalg.norm(act_j)) / 2
                between_distances.append(dist / (norm + 1e-10))

        within_mean = float(np.mean(within_distances))
        between_mean = float(np.mean(between_distances))
        ratio = within_mean / (between_mean + 1e-10)

        results[layer] = {
            "within_set_mean_rel_dist": within_mean,
            "within_set_std": float(np.std(within_distances)),
            "between_set_mean_rel_dist": between_mean,
            "between_set_std": float(np.std(between_distances)),
            "within_over_between_ratio": ratio,
        }

        if verbose:
            print(f"  Layer {layer:>3d}: within={within_mean:.4f} ± {np.std(within_distances):.4f}, "
                  f"between={between_mean:.4f} ± {np.std(between_distances):.4f}, "
                  f"ratio={ratio:.4f}")

    return results, all_orderings, prompt_cache


def measure_behavioral_change(
    model, fact_sets: list[list[str]], verbose=True,
) -> dict:
    """Measure how next-token predictions change with fact order.

    For each fact set, compare KL divergence between different orderings.
    """
    n = len(fact_sets[0])
    results = []

    for fs_idx, fs in enumerate(fact_sets):
        orderings = generate_ordered_prompts(fs, n)
        identity_prompt = orderings[0]["prompt"]

        # Get logits for identity ordering
        tokens_id = model.to_tokens(identity_prompt)
        with torch.no_grad():
            logits_id = model(tokens_id)[0, -1, :]

        for ordering in orderings[1:]:
            prompt = ordering["prompt"]
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model(tokens)[0, -1, :]

            # KL divergence
            log_p_id = F.log_softmax(logits_id, dim=-1)
            p_perm = F.softmax(logits, dim=-1)
            kl = F.kl_div(F.log_softmax(logits, dim=-1),
                          F.softmax(logits_id, dim=-1),
                          reduction="sum").item()

            # Top-1 agreement
            top1_id = torch.argmax(logits_id).item()
            top1_perm = torch.argmax(logits).item()

            results.append({
                "fact_set_idx": fs_idx,
                "permutation": str(ordering["permutation"]),
                "parity": ordering["permutation"].parity,
                "kl_divergence": kl,
                "top1_agree": top1_id == top1_perm,
            })

    # Summarize
    kls = [r["kl_divergence"] for r in results]
    top1_agree = [r["top1_agree"] for r in results]

    summary = {
        "mean_kl": float(np.mean(kls)),
        "std_kl": float(np.std(kls)),
        "median_kl": float(np.median(kls)),
        "top1_agreement_rate": float(np.mean(top1_agree)),
        "n_comparisons": len(results),
    }

    if n == 3:
        even_kls = [r["kl_divergence"] for r in results if r["parity"] == 0]
        odd_kls = [r["kl_divergence"] for r in results if r["parity"] == 1]
        summary["mean_kl_even"] = float(np.mean(even_kls)) if even_kls else None
        summary["mean_kl_odd"] = float(np.mean(odd_kls)) if odd_kls else None

    if verbose:
        print(f"\n  Mean KL divergence: {summary['mean_kl']:.4f} ± {summary['std_kl']:.4f}")
        print(f"  Median KL: {summary['median_kl']:.4f}")
        print(f"  Top-1 agreement rate: {summary['top1_agreement_rate']:.2%}")
        if "mean_kl_even" in summary and summary["mean_kl_even"] is not None:
            print(f"  Mean KL (even perms): {summary['mean_kl_even']:.4f}")
            print(f"  Mean KL (odd perms): {summary['mean_kl_odd']:.4f}")

    return summary, results


def test_fact_order_group_structure(
    model, fact_sets_3: list[list[str]], layers: list[int], verbose=True,
) -> dict:
    """Fit fact-reordering operators and test S_3 group relations.

    Same methodology as entity permutation experiments, but operating
    on fact-level reorderings instead of entity-level swaps.
    """
    group = PermutationGroup(3)
    gens = group.generators()
    s_perm = gens["s"]
    r_perm = gens["r"]

    # Generate all orderings for all fact sets
    all_orderings_by_set = []
    for fs in fact_sets_3:
        orderings = generate_ordered_prompts(fs, 3)
        all_orderings_by_set.append({o["permutation"]: o for o in orderings})

    # Cache activations
    all_prompts = set()
    for orderings in all_orderings_by_set:
        for o in orderings.values():
            all_prompts.add(o["prompt"])

    if verbose:
        print(f"\nCaching {len(all_prompts)} prompts for group structure test...")
    prompt_cache = {}
    for prompt in tqdm(sorted(all_prompts), disable=not verbose, desc="Caching"):
        cached = cache_activations(
            model, prompt, hook_points=["resid_post"], layers=layers
        )
        resid = cached.activations["resid_post"]
        prompt_cache[prompt] = {
            layers[i]: resid[i, -1, :].detach().cpu().numpy()
            for i in range(len(layers))
        }

    results_by_layer = {}
    for layer in layers:
        # For each non-identity permutation, collect (identity, permuted) pairs
        perm_pairs = defaultdict(lambda: {"base": [], "perm": []})

        for orderings in all_orderings_by_set:
            identity_prompt = orderings[Permutation.identity(3)]["prompt"]
            base_act = prompt_cache[identity_prompt][layer]

            for perm, o in orderings.items():
                if perm.is_identity:
                    continue
                perm_act = prompt_cache[o["prompt"]][layer]
                perm_pairs[perm]["base"].append(base_act)
                perm_pairs[perm]["perm"].append(perm_act)

        # Fit operators
        operators = {Permutation.identity(3): np.eye(model.cfg.d_model)}
        fit_info = {}

        for perm, data in perm_pairs.items():
            X = np.stack(data["base"])
            Y = np.stack(data["perm"])
            fit = fit_operator(torch.tensor(X), torch.tensor(Y))
            operators[perm] = fit.W
            fit_info[str(perm)] = {
                "test_error": fit.test_relative_error,
                "test_cosine": fit.test_cosine_similarity,
                "parity": perm.parity,
            }

        # Test group relations
        if s_perm in operators and r_perm in operators:
            s3_rels = s3_relation_errors(operators[s_perm], operators[r_perm])
        else:
            s3_rels = {}

        # Composition errors
        comp_errors = []
        for p1 in group.elements:
            for p2 in group.elements:
                if p1.is_identity or p2.is_identity:
                    continue
                composed = p1 * p2
                if all(p in operators for p in [p1, p2, composed]):
                    err = composition_error(operators[p1], operators[p2], operators[composed])
                    comp_errors.append(err.relative_error)

        mean_comp = float(np.mean(comp_errors)) if comp_errors else None

        results_by_layer[layer] = {
            "fit_quality": fit_info,
            "s3_relations": {
                name: {"relative": r.relative_error, "frobenius": r.frobenius_error}
                for name, r in s3_rels.items()
            } if s3_rels else {},
            "mean_composition_error": mean_comp,
        }

        if verbose:
            mean_test = np.mean([v["test_error"] for v in fit_info.values()
                                 if v["test_error"] is not None])
            print(f"\n  Layer {layer}:")
            print(f"    Mean test error: {mean_test:.4f}")
            if s3_rels:
                for name, r in s3_rels.items():
                    print(f"    {name}: {r.relative_error:.4f}")
            print(f"    Mean composition error: {mean_comp:.4f}" if mean_comp else "")

    return results_by_layer


def main():
    parser = argparse.ArgumentParser(description="Fact-order permutation experiment")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 10, 16, 21, 23])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(model_name=args.model, device=args.device)
    print(f"Loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    all_results = {}

    # --- 2-fact experiments ---
    print(f"\n{'#'*70}")
    print(f"# 2-FACT SEQUENCES (S_2 permutations)")
    print(f"{'#'*70}")

    print(f"\n--- Activation Distances ---")
    dist_results_2, _, _ = measure_activation_distances(
        model, FACT_SETS_2, args.layers, verbose=not args.quiet
    )
    all_results["s2_distances"] = {str(k): v for k, v in dist_results_2.items()}

    print(f"\n--- Behavioral Change ---")
    behav_2, behav_details_2 = measure_behavioral_change(
        model, FACT_SETS_2, verbose=not args.quiet
    )
    all_results["s2_behavioral"] = behav_2

    # --- 3-fact experiments ---
    print(f"\n{'#'*70}")
    print(f"# 3-FACT SEQUENCES (S_3 permutations)")
    print(f"{'#'*70}")

    print(f"\n--- Activation Distances ---")
    dist_results_3, _, _ = measure_activation_distances(
        model, FACT_SETS_3, args.layers, verbose=not args.quiet
    )
    all_results["s3_distances"] = {str(k): v for k, v in dist_results_3.items()}

    print(f"\n--- Behavioral Change ---")
    behav_3, behav_details_3 = measure_behavioral_change(
        model, FACT_SETS_3, verbose=not args.quiet
    )
    all_results["s3_behavioral"] = behav_3

    print(f"\n--- Group Structure Test ---")
    group_results = test_fact_order_group_structure(
        model, FACT_SETS_3, args.layers, verbose=not args.quiet
    )
    all_results["s3_group_structure"] = {str(k): v for k, v in group_results.items()}

    # --- Summary ---
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\nActivation distance: within-set (same facts, different order)")
    print(f"  vs between-set (different facts entirely)")
    print(f"\n{'':>8s}  {'Within':>10s}  {'Between':>10s}  {'Ratio':>8s}")
    print(f"  S_2:")
    for layer in args.layers:
        r = dist_results_2[layer]
        print(f"  L{layer:>3d}:  {r['within_set_mean_rel_dist']:>10.4f}  "
              f"{r['between_set_mean_rel_dist']:>10.4f}  "
              f"{r['within_over_between_ratio']:>8.4f}")
    print(f"  S_3:")
    for layer in args.layers:
        r = dist_results_3[layer]
        print(f"  L{layer:>3d}:  {r['within_set_mean_rel_dist']:>10.4f}  "
              f"{r['between_set_mean_rel_dist']:>10.4f}  "
              f"{r['within_over_between_ratio']:>8.4f}")

    print(f"\nBehavioral change from fact reordering:")
    print(f"  S_2: KL={behav_2['mean_kl']:.4f}, top-1 agreement={behav_2['top1_agreement_rate']:.2%}")
    print(f"  S_3: KL={behav_3['mean_kl']:.4f}, top-1 agreement={behav_3['top1_agreement_rate']:.2%}")

    if group_results:
        print(f"\nGroup structure (S_3 fact reordering):")
        print(f"  {'Layer':>6s}  {'TestErr':>8s}  {'s²=e':>8s}  {'r³=e':>8s}  {'Comp':>8s}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for layer in args.layers:
            r = group_results[layer]
            mean_test = np.mean([v["test_error"] for v in r["fit_quality"].values()
                                 if v["test_error"] is not None])
            s3 = r["s3_relations"]
            s2e = s3.get("s^2 = e", {}).get("relative", float("nan"))
            r3e = s3.get("r^3 = e", {}).get("relative", float("nan"))
            comp = r["mean_composition_error"] or float("nan")
            print(f"  {layer:>6d}  {mean_test:>8.4f}  {s2e:>8.4f}  {r3e:>8.4f}  {comp:>8.4f}")

    print(f"\nInterpretation:")
    print(f"  Ratio close to 0: reordering barely changes representations (S_n invariance)")
    print(f"  Ratio close to 1: reordering changes representations as much as different facts")
    print(f"  Ratio >> 0 + no group structure: ordering encodes information non-algebraically")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def ser(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        output_path.write_text(json.dumps(all_results, indent=2, default=ser))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
