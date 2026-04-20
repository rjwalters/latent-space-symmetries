# Latent Space Symmetries

## Project Overview

This is a mechanistic interpretability research project investigating whether transformer latent representations exhibit approximate permutation group symmetry (S_n or A_n). The target model is **Qwen/Qwen2.5-1.5B** (base, not instruct).

## Central Research Question

Is there a subspace of activations, or a feature basis, on which permutations of a set of entities act approximately like a representation of S_n? Do even permutations act more cleanly, suggesting A_n-like structure?

## Architecture

The codebase is organized into focused modules:

- `src/datasets/` — Prompt generation with controlled permutation structure (S_2, S_3, S_4)
- `src/activation/` — TransformerLens-based activation caching across layers
- `src/operators/` — Fitting and testing permutation operators W_π with group-relation metrics
- `src/sae/` — SAE training and feature-space analysis
- `src/interventions/` — Causal validation via activation patching
- `notebooks/` — Exploratory analysis, one per milestone
- `tests/` — Unit tests, especially for group-theory utilities and dataset correctness

## Key Technical Details

**Model**: Qwen2.5-1.5B Base — 28 layers, hidden size 1536, 12 query heads / 2 KV heads, RoPE, SwiGLU, RMSNorm, tied embeddings.

**Stack**: TransformerLens (activation capture/patching), SAELens (sparse feature decomposition), PyTorch, NumPy/SciPy for linear algebra.

**Important confounders to always account for**:
- RoPE causes positional entanglement — permutation symmetry tests must factor out position
- Tokenizer artifacts — use entity names with matched tokenization patterns
- Lexical asymmetry — some names have different pretraining frequencies

## Research Phases (in order)

1. Generate symmetry-bearing prompt datasets (Phase 1)
2. Behavioral validation — confirm model outputs respect permutation symmetry (Phase 3)
3. Activation capture and linear alignment search across layers (Phase 4)
4. Operator fitting and group-relation testing for S_2, S_3 (Phase 5)
5. SAE training on promising layers, feature-space operator fitting (Phase 6)
6. A_n vs S_n parity analysis (Phase 7)
7. Causal intervention validation (Phase 8)
8. Circuit localization (Phase 9)

## Milestones

1. Behavioral S_2/S_3 permutation task consistency
2. Find layers with significantly linear swap-aligned activations
3. SAE feature-space operators fit better than raw hidden-space
4. Approximate group composition for S_3 operators
5. Evidence for full S_n, A_n-cleaner-than-S_n, or sign-representation component
6. Causal steering via learned permutation operators

## Conventions

- Always use the base model (`Qwen/Qwen2.5-1.5B`), never instruct
- Progress S_2 → S_3 → S_4; don't skip ahead
- Validate behaviorally before examining internals
- Correlation is not causation — always follow operator fitting with causal interventions
- Save large artifacts (activations, model weights, SAE checkpoints) under `data/` (gitignored)
- Notebooks are exploratory; reusable logic goes into `src/`

## References

- [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [Group Crosscoders (arXiv 2410.24184)](https://arxiv.org/abs/2410.24184) — symmetry-oriented mechanistic interpretability
- [TransformerLens](https://transformerlensorg.github.io/TransformerLens/)
- [SAELens](https://github.com/jbloomAus/SAELens)
