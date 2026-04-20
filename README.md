# Approximate Permutation Representations in Transformer Latent Space

Investigating whether transformer latent representations exhibit approximate S_n or A_n group symmetry, using [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) as the target model.

## Research Question

> Is there a subspace of activations, or a feature basis, on which permutations of a set of entities act approximately like a representation of S_n, and do even permutations act more cleanly, suggesting A_n-like structure?

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
src/
  datasets/      # Symmetry-bearing prompt generation
  activation/    # TransformerLens activation capture
  operators/     # Permutation operator fitting & group-relation testing
  sae/           # Sparse autoencoder training & feature analysis
  interventions/ # Causal validation via activation patching
notebooks/       # Exploratory analysis (one per milestone)
tests/           # Unit tests
data/            # Artifacts (gitignored)
```

## Milestones

1. Behavioral S_2/S_3 permutation task consistency
2. Linear swap-aligned activations at specific layers
3. SAE feature-space operators outperform raw hidden-space operators
4. Approximate group composition for S_3
5. S_n vs A_n parity structure evidence
6. Causal steering via learned permutation operators

## References

- [Group Crosscoders (arXiv 2410.24184)](https://arxiv.org/abs/2410.24184) — symmetry-oriented mechanistic interpretability
- [TransformerLens](https://transformerlensorg.github.io/TransformerLens/)
- [SAELens](https://github.com/jbloomAus/SAELens)
