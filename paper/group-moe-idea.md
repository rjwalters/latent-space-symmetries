# Group-Structured Mixture of Experts: Symmetry-Aware Routing for Efficient Representation

## Core Idea

A mixture-of-experts architecture where each expert module implements a specific group representation, and the router learns to detect which symmetry applies to the current input. Instead of every layer being equivariant (rigid) or no layer being equivariant (current LLMs), the model *chooses when and which* symmetry to apply.

## Motivation

Our findings in "Ordering Is Not Invariant" show that:

1. **LLMs learn functional equivariance without structural equivariance.** The model produces correct outputs under permutation but uses independent linear maps (a lookup table) rather than group representations (an algebra).

2. **This is wasteful.** A general linear operator for each permutation requires $d^2$ parameters. A group-constrained representation in the irrep basis requires $O(k)$ parameters where $k$ is the dimension of the active irreps — potentially a 100,000x compression.

3. **This limits compositional generalization.** Independent operators can't compose correctly. A model that learns $W_{(0,1)}$ and $W_{(1,2)}$ separately has no guarantee that $W_{(0,1)} W_{(1,2)} \approx W_{(0,1,2)}$. Group representations compose by construction.

The question becomes: can we design an architecture that *earns* group structure where it exists, without forcing it where it doesn't?

## Architecture Sketch

```
Input tokens
    │
    ▼
┌─────────────────────────────┐
│  Standard Transformer Layers │  (unchanged — handles general computation)
│  (attention + MLP)           │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Symmetry Router            │  Detects which group action (if any) applies
│  (lightweight classifier)    │  to the current representation
│                              │
│  Outputs:                    │
│    - group_id ∈ {none, S₂,  │
│      S₃, Z₂, D₄, ...}      │
│    - element_id within group │
│    - confidence score        │
└──────────┬──────────────────┘
           │
     ┌─────┴──────┐
     │             │
     ▼             ▼
┌─────────┐  ┌──────────────────┐
│ Pass-   │  │ Group Expert      │
│ through │  │ Module            │
│ (no     │  │                   │
│ symmetry│  │ Implements R(g)   │
│ needed) │  │ in irrep basis:   │
│         │  │  ┌───┐            │
│         │  │  │ ρ₁│ trivial    │
│         │  │  ├───┤            │
│         │  │  │ ρ₂│ sign       │
│         │  │  ├───┤            │
│         │  │  │ ρ₃│ standard   │
│         │  │  └───┘            │
│         │  │                   │
│         │  │ Block-diagonal    │
│         │  │ application in    │
│         │  │ learned subspace  │
└────┬────┘  └───────┬──────────┘
     │               │
     └───────┬───────┘
             │
             ▼
┌─────────────────────────────┐
│  Standard Transformer Layers │  (continues as normal)
└─────────────────────────────┘
```

## Key Design Decisions

### 1. The Router

The router is a small network that takes the current residual stream representation and outputs:
- **Which group** (if any) is active in this context
- **Which element** of that group corresponds to the current transformation
- **A confidence** that determines how much to blend the group expert's output with the pass-through

This is analogous to MoE routing but the expert selection is semantically meaningful — the router is learning to detect symmetry in the data, not just routing for capacity.

### 2. The Group Expert Modules

Each group expert contains:
- A **learned projection** $P: \mathbb{R}^d \to \mathbb{R}^k$ into the symmetry-active subspace (the dimensions where the group action matters)
- **Irrep matrices** for each group element, stored in block-diagonal form
- A **learned injection** $P^\dagger: \mathbb{R}^k \to \mathbb{R}^d$ back to the full space

For $S_3$, the irrep basis is:
- Trivial (1D): invariant content — doesn't change under permutation
- Sign (1D): parity-sensitive content — flips sign under odd permutations
- Standard (2D): the relational structure that permutes

The expert applies $R(g) = P^\dagger \cdot \text{diag}(\rho_1(g), \rho_2(g), \rho_3(g)) \cdot P$

Total parameters per group element: $2dk + k^2$ where $k \ll d$. For $S_3$ with $k = 4$ and $d = 896$: about 7,200 parameters vs 802,816 for a full $d \times d$ matrix.

### 3. Why This Could Be Efficient

| Approach | Parameters per transform | Composition | Generalization |
|----------|------------------------|-------------|----------------|
| Full linear map | $d^2 = 802,816$ | Not guaranteed | Overfits to seen permutations |
| Group expert (irrep) | $2dk + k^2 \approx 7,200$ | Exact by construction | Generalizes to unseen compositions |
| Equivariant layer | 0 (hard-coded) | Exact | Perfect but rigid |

The group-MoE sits in the sweet spot: much more efficient than unconstrained maps, compositionally correct, but flexible about *when* to apply symmetry.

### 4. Training Signal

The group experts are more parameter-efficient than general MLPs. If the data has genuine symmetry, routing through the group expert achieves lower loss with fewer parameters. This creates a natural incentive: the router learns to detect symmetry because it's *cheaper* to process symmetric inputs through the group pathway.

This is the key insight: **you don't need to label which inputs have symmetry. The efficiency advantage of the group expert creates a self-supervised signal for symmetry detection.**

### 5. What Groups to Include

Start small and grow:
- $S_2$ (pairwise swaps): ubiquitous in language (subject-object reversal, comparison)
- $\mathbb{Z}_2$ (binary contrast): negation, antonymy, true/false
- $S_3$ (three-entity permutations): multi-entity reasoning
- $D_4$ (square symmetries): spatial/visual layout reasoning
- Continuous: $SO(2)$ for angular/periodic concepts (time of day, compass directions)

Each group expert is small (a few thousand parameters). You can include many without significant cost.

## Connection to Our Paper

Our paper shows that current LLMs are at the "full linear map" row of the table above — they learn independent operators that work but don't compose. The Group-MoE architecture would move them to the "group expert" row, gaining:

1. **100x parameter reduction** for symmetry-bearing transformations
2. **Compositional generalization** — $W_{(0,1)} W_{(1,2)} = W_{(0,1,2)}$ by construction
3. **Interpretable structure** — you can inspect which group the router detected and which irrep is active
4. **Selective application** — symmetry is used when the data has it, bypassed when it doesn't

The fact that our paper shows models *don't* learn group structure spontaneously is actually the motivation: if you want it, you have to build it in. The Group-MoE does this without the rigidity of fully equivariant architectures.

## Open Questions

1. **Does the router actually learn to detect symmetry?** Or does it degenerate to using group experts as generic additional capacity? Need careful experiments with controlled symmetry in training data.

2. **How to handle approximate symmetry?** Real data has approximate, not exact, symmetry. The confidence score from the router helps (blend with pass-through), but the irreps themselves are exact. Need soft irrep decomposition?

3. **Interaction between group experts.** If the input has both $S_2$ symmetry (entity swap) and $\mathbb{Z}_2$ symmetry (negation), how do the experts interact? Product groups? Sequential application?

4. **Does this actually improve downstream tasks?** The efficiency argument is clear but the generalization argument needs empirical validation. Train a Group-MoE on compositional reasoning benchmarks and compare.

5. **Scale.** Does this help at 7B+ parameter scale, or is it only relevant for smaller models where parameter efficiency matters more?
