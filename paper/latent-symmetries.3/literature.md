# Literature Review — Ordering Is Not Invariant

## Positioning

This paper sits at the intersection of three research areas: (1) mechanistic interpretability of language models, (2) group-theoretic approaches to neural network representations, and (3) synthetic data generation. While each area has substantial prior work, their intersection — specifically, whether LLM latent representations exhibit permutation group symmetry and what this implies for synthetic data design — is essentially unaddressed.

The closest prior work is Group Crosscoders (Bhaskar et al., 2024), which finds approximate dihedral group (D_32) equivariance in a vision model (InceptionV1). Our work provides a complementary negative result for language models, establishing that the presence or absence of group structure in representations reflects the presence or absence of symmetry in the data domain, not a universal property of neural networks. This contrast is the central insight of the paper.

The synthetic data connection is novel. While the Simula framework (Davidson et al., 2026) and related work has formalized synthetic data generation as mechanism design over product spaces of independent factors, no prior work has empirically tested whether LLM representations treat semantic content as permutation-invariant — the implicit assumption underlying such approaches.

## Key Related Work

### Group Theory and Neural Network Representations
- **Group Crosscoders** (Bhaskar et al., arXiv 2410.24184, 2024): Found approximate D_32 (dihedral group, rotations + reflections) equivariance in InceptionV1 mixed3b layer using dictionary learning on concatenated activations from transformed inputs. Key finding: curve detectors show full 360° equivariance, line detectors show 180° invariance. Important contrast with our work: they study a physical symmetry (rotation) in a CNN with architectural translation equivariance. We study a linguistic "symmetry" (entity permutation) in a transformer without architectural equivariance constraints.
- **Geometric Deep Learning** (Bronstein et al., 2021, arXiv 2104.13478): Comprehensive survey establishing that symmetry and equivariance are organizing principles for neural network architecture design. Key framework: grids→groups→graphs→geodesics→gauges. Our work asks: when symmetry is NOT built into the architecture, does it emerge from data?
- **Group equivariant CNNs** (Cohen & Welling, 2016): Showed that incorporating group symmetry into CNN architectures improves sample efficiency and generalization. Establishes that explicit architectural equivariance works; our result shows implicit equivariance from data alone does not emerge in transformers.
- **Permutation Equivariance of Transformers** (Xu et al., CVPR 2024, arXiv 2304.07735): Proved that vanilla transformer architectures possess inter- and intra-token permutation equivariance in both forward and backward propagation. Important distinction: this is about the architecture's theoretical equivariance, not about learned representations. Our work tests whether this architectural property translates to group-structured internal representations.

### Mechanistic Interpretability
- **TransformerLens** (Nanda et al., 2022): Framework for activation capture and intervention in transformer models. We use this for all activation-level experiments.
- **SAELens / Sparse Autoencoders** (Bricken et al., 2023; Cunningham et al., 2023): Feature decomposition in transformer residual streams. We train SAEs and test whether permutation operators have cleaner group structure in the sparse feature basis. (They don't.)
- **Linear Representation Hypothesis** (Park, Choe & Veitch, ICML 2024, arXiv 2311.03658): Formalizes the idea that high-level concepts are represented as linear directions in representation space using counterfactual semantics. Our operator fitting approach is consistent with this framework — we test whether permutation-induced changes are linear and find they are, but without algebraic structure.
- **Circuit Tracing** (Anthropic, 2025): Cross-layer transcoders producing interpretable "replacement models." Represents the state of the art in mechanistic interpretability methods. Our work uses simpler methods (linear operators) but asks a more targeted algebraic question.

### Synthetic Data Generation
- **Simula** (Davidson et al., TMLR 2026, arXiv 2603.29791): Reasoning-driven framework decomposing data generation into coverage, complexity, and quality axes via hierarchical taxonomies. Key assumption: domain space decomposes as product T_0 × T_1 × ... × T_K with factors sampled independently. Our work tests the downstream assumption that permuting/reordering content is semantically neutral.
- **Synthetic data scaling laws** (various, 2025-2026): Emerging consensus that "better data scales better" — data quality and composition matter more than volume. Our finding that fact ordering encodes information suggests an underexplored quality dimension.

### Information Structure and Discourse
- **Information Structure** (Krifka, 2008; Lambrecht, 1994): Linguistic framework establishing that sentence and discourse ordering encode topic-focus structure, givenness, and information status. The convention of given-before-new, topic-before-comment is well-established in pragmatics.
- **Order effects in discourse** (van Dijk, 1977; Downing & Noonan, 1995): Propositions closest to the topic are uttered first; ordering reflects the speaker's model of the hearer's mental state. Our empirical finding that LLMs are sensitive to fact ordering is consistent with these pragmatic principles.

## Gap Analysis

No prior work has:
1. Systematically tested whether LLM latent representations carry permutation group structure (S_n representations, irreps, composition closure)
2. Established positive and negative controls for group structure detection in neural network activations
3. Connected the absence of group structure to implications for synthetic data generation
4. Empirically measured the representational impact of fact-level reordering in LLMs
5. Distinguished functional equivariance (correct outputs) from structural equivariance (internal group representations) as an empirical rather than theoretical matter

## Search Methodology

Searched: arXiv (cs.LG, cs.CL, cs.AI), Google Scholar, Semantic Scholar, ACL Anthology
Terms: "permutation equivariance transformer", "group representations neural networks", "mechanistic interpretability group theory", "synthetic data ordering", "information structure LLM", "linear representation hypothesis", "geometric deep learning equivariant", "Group Crosscoders"
Date range: primarily 2020-2026, key foundational works from earlier
