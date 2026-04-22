# Ordering Is Not Invariant: The Absence of Permutation Group Structure in Language Model Representations

## Abstract

We investigate whether transformer language models develop internal representations with permutation group symmetry. Using operator fitting, group relation testing, and representation-theoretic analysis across two scales of Qwen2.5 (0.5B, 1.5B), we find that models are strongly permutation-sensitive — linear operators mapping between permuted representations fit 21x better than chance — but exhibit zero group-algebraic structure. Composition errors for S_3 operators are indistinguishable from random baselines. This null result holds across all layers, in PCA subspaces, in SAE feature bases, and at both model scales. A positive control on synthetic data with exact group structure confirms the pipeline would detect it if present. We extend these findings to fact-level permutations, showing that reordering the same facts changes model representations 35-60% as much as presenting entirely different facts, with next-token predictions changing 30% of the time. We argue this reflects a fundamental property: local optimization under next-token prediction learns functional equivariance (correct outputs under transformation) without structural equivariance (group representations in latent space). This has implications for synthetic data generation, where the implicit assumption that semantic content is order-invariant may destroy information the model has learned about salience and discourse structure.

---

## 1. Introduction

- Symmetry as a lens for understanding neural representations
  - Physics: global symmetries discovered by stepping outside local optimization
  - Vision: Group Crosscoders find approximate D_32 in CNNs where physical symmetry exists
  - Language: does S_n permutation symmetry emerge in LLMs?
- The practical motivation: synthetic data generation
  - Simula and related approaches decompose data into independent factors
  - Implicit assumption: reordering facts/concepts is semantically neutral
  - If models have learned that ordering encodes information, scrambling it destroys signal
- Our contribution: comprehensive negative result + explanation of why it's expected + evidence that ordering encodes information
- Preview of key finding: permutation-sensitive, not permutation-equivariant

## 2. Background

### 2.1 Group Representations in Neural Networks
- Definitions: group, representation, equivariance, invariance
- Irreducible representations of S_n (trivial, sign, standard 2D for S_3)
- The distinction between functional equivariance (f(g·x) = g·f(x) at the output level) and structural equivariance (internal representations carry group structure)

### 2.2 Prior Work
- Group Crosscoders (arXiv 2410.24184): D_32 equivariance in InceptionV1 early vision
  - Key contrast: they study a physical symmetry (rotation) in a CNN with built-in translational equivariance
- Geometric deep learning and equivariant architectures
- Mechanistic interpretability: TransformerLens, SAEs, activation patching
- Sparse autoencoders and feature decomposition

### 2.3 Synthetic Data Generation
- Simula framework: domain decomposition into taxonomy products T_0 x T_1 x ... x T_K
- The product-space assumption: factors are independent and freely combinable
- Broader practice: fact shuffling, context reordering in RAG, data augmentation
- What's at stake: if ordering encodes information, scrambling it is lossy

## 3. Methods

### 3.1 Experimental Setup
- Models: Qwen2.5-0.5B (24 layers, d=896) and Qwen2.5-1.5B (28 layers, d=1536)
- Activation capture via TransformerLens (residual stream, last token position)
- Two levels of permutation:
  - Entity-level: permuting named entities within a fixed template (S_2, S_3)
  - Fact-level: reordering independent sentences/facts (S_2, S_3)

### 3.2 Operator Fitting
- For each permutation pi, fit linear operator W_pi via least-squares: h(pi·x) ≈ W_pi h(x)
- Train/test split for generalization metrics
- Applied at every layer to find where permutation structure is most linear

### 3.3 Group Relation Testing
- S_3 presentation: s^2 = e, r^3 = e, srs = r^{-1}
- Composition closure: W_{pi1} W_{pi2} ≈ W_{pi1 o pi2} for all pairs
- Inverse: W_pi W_{pi^{-1}} ≈ I
- Relative error metric: ||LHS - RHS||_F / ||RHS||_F

### 3.4 Controls
- Positive control: synthetic activations with exact S_3 group structure at calibrated noise levels
- Random baseline: operators fit on shuffled (unrelated) prompt pairs
- Comparison to establish that (a) the pipeline works and (b) real operators are better than chance

### 3.5 Additional Analyses
- PCA subspace search across dimensions 3-200
- Irreducible representation search (sign rep, standard 2D irrep)
- SAE feature-space operator fitting
- S_2 subgroup eigenvalue analysis
- Fact-order activation distance and behavioral impact

## 4. Results

### 4.1 Entity-Level Permutations

#### 4.1.1 Behavioral Validation (Phase 3)
- Model reliably tracks entity permutations in completions
- KL divergence 0.15-2.1 depending on template
- Confirms the model "knows" about permutations at the output level

#### 4.1.2 Activation Alignment (Phase 4)
- U-shaped layer profile: best fit at layers 0-2 (test error 0.08), worst at layers 10-16 (0.43), secondary peak at 20-22 (0.26)
- Same pattern at 1.5B scale (best layer 0, test error 0.12)
- Permutation signal is linearly predictable — operators fit well

#### 4.1.3 Group Relations (Phase 5) — The Null Result
- All presentation relation errors ≈ 1.0 in full space
- s^2 = e: 1.02, r^3 = e: 1.09, srs = r^{-1}: 0.52
- Composition errors ~1.0, indistinguishable from random baseline
- **Same result at 1.5B**: scaling doesn't help

#### 4.1.4 Subspace and Basis Analysis
- PCA subspace: group relations improve slightly at very low dimensions (3-5D) but r^3 = e still fails (0.73)
- Irrep search: no sign representation or standard 2D irrep found
- SAE feature space: group relations worse than raw space (errors 1-34x)
- S_2 subgroup: transpositions are NOT involutions in full space (eigenvalues near +1, none near -1, det ≈ 0)

### 4.2 Controls

#### 4.2.1 Positive Control
- Synthetic S_3 structure in d=10 space: relation errors → 0 at noise=0, scale linearly with noise
- At noise comparable to real model fit quality: relation errors ~0.05 (20x below real model's ~1.0)
- Proves pipeline would detect structure if present

#### 4.2.2 Random Baseline
- Shuffled-pair operators fit 2.5-21x worse than real pairs
- But group relation errors from shuffled pairs are comparable to real pairs
- Confirms: the permutation signal is real, but the algebraic structure is not

### 4.3 Fact-Level Permutations

#### 4.3.1 Activation Distance
- Reordering same facts changes activations 35-60% as much as different facts
- Ratio stable across layers, consistent for 2-fact and 3-fact sequences
- The model does NOT treat fact-sets as permutation-invariant

#### 4.3.2 Behavioral Impact
- Top-1 next-token prediction changes 30-33% of the time from fact reordering
- KL divergence 0.28-0.35 — comparable to entity-level permutations
- Even permutations (3-cycles) produce larger KL than odd (transpositions)

#### 4.3.3 Group Structure
- Same null result as entity permutations: operators fit well (test error 0.13 at layer 0), group relations ~1.0
- Confirms the pattern generalizes beyond entity-swapping

#### 4.3.4 Case Study: "The Room Is On Fire"
- KL divergence matrix for 6 orderings of {fire, research, Wednesday}
- Model cares most about last-mentioned fact (recency)
- Some orderings nearly equivalent (KL=0.026), others very different (KL=0.35)
- Illustrates that ordering encodes salience, not group structure

## 5. Analysis and Discussion

### 5.1 Why No Group Structure?
- Three candidate explanations:
  - (A) The territory: idea space doesn't have S_n symmetry — ordering genuinely encodes information
  - (B) The map: transformers can't learn group structure even if it exists
  - (C) The reward: next-token prediction doesn't incentivize algebraic coherence
- Evidence for A: fact reordering changes predictions; linguistics confirms ordering carries meaning
- Evidence against pure B: Group Crosscoders show vision models learn D_32 when physical symmetry exists
- Likely answer: combination of A and C — the symmetry isn't there AND the objective wouldn't reward learning it even if it were

### 5.2 Functional vs Structural Equivariance
- The model achieves functional equivariance: swap Alice and Bob → the completion swaps accordingly
- But not structural equivariance: internal representations don't carry group structure
- A lookup table, not an algebra
- This is sufficient for next-token prediction and for survival (biological analogy)
- Discovering that functional equivariance arises from structural symmetry is what physics/science does — not something local optimizers are pressured to learn

### 5.3 The Product-Space Representation
- The model represents permutations as independent points in a high-dimensional space
- This is the same geometry as Simula's taxonomy product T_0 x T_1 x ... x T_K
- Both the model and the data generation framework converge on product-space representations because neither has an incentive to learn group structure
- The unconstrained representation has MORE capacity than the group-constrained one

### 5.4 Implications for Synthetic Data Generation
- Fact ordering is not symmetry: scrambling it destroys learned information
- The model has learned that ordering encodes salience, topic, narrative frame
- Synthetic data approaches that treat fact-sets as unordered risk:
  - Washing out ordering information the model has already learned
  - Imposing a false S_n invariance on a space that doesn't have it
  - Generating training data where "the room is on fire" and "it is Wednesday" have equal narrative weight regardless of position
- This doesn't mean all reordering is harmful — it means ordering should be treated as a meaningful dimension, not a nuisance variable

### 5.5 When SHOULD We Expect Group Structure?
- Physical symmetries in vision (rotation, reflection): yes, and Group Crosscoders confirm
- Equivariant architectures (GNNs, equivariant transformers): by construction
- Tasks requiring compositional generalization over group orbits: potentially
- General-purpose language models on natural text: no, because the data + objective don't demand it

## 6. Limitations
- Only tested S_2, S_3 (small groups) — larger groups may differ
- Only tested Qwen2.5 family — architecture-specific effects possible
- Prompt templates may not capture all permutation-relevant tasks
- Fact-level permutations tested on relatively simple multi-sentence sequences
- Did not test whether training with permutation-augmented data would induce group structure
- Did not directly compare model performance on permuted vs ordered synthetic training data

## 7. Future Work
- Train models on permutation-augmented data and measure downstream impact
- Test whether explicitly equivariant architectures develop group representations for linguistic permutations
- Biological analogy: do fMRI visual cortex representations show rotational group structure (cf. Group Crosscoders)?
- Extend to other symmetries: negation (Z_2), tense shifts, etc.
- Direct weight matrix analysis for emergent group structure
- Test whether curriculum ordering effects in training are explained by our ordering-encodes-information finding

## 8. Conclusion

Transformer language models develop permutation-sensitive but not permutation-equivariant representations. This is not a failure of the model — it reflects the structure of the task and the data. Idea space does not have S_n symmetry; fact ordering encodes salience, priority, and narrative structure. Local optimization under next-token prediction correctly learns this asymmetry without discovering the group-theoretic abstractions that would be needed for compositional generalization over permutations. For synthetic data generation, this means that treating semantic content as order-invariant is an empirical assumption that may not hold, and that ordering should be treated as a meaningful dimension of the data rather than a nuisance variable to be averaged out.

---

## Appendices

### A. Prompt Templates and Entity Sets
- Full list of S_2 and S_3 prompt families
- Entity sets and tokenization considerations

### B. Fact Sets for Fact-Order Experiments
- All 15 two-fact and 15 three-fact sets

### C. Positive Control Technical Details
- Exact S_3 representation construction
- Embedding in high-dimensional space
- Noise calibration curves

### D. Complete Layer-by-Layer Results
- Full tables for both model scales
- Per-permutation fit quality
