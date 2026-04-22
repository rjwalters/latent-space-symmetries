# Review: latent-symmetries.1

**Reviewer:** Claude (automated paper review)
**Date:** 2026-04-22
**Paper reviewed:** `paper/latent-symmetries.1/paper.tex`

---

## Overall Assessment: NEEDS WORK

**Score: 27/40**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Technical Soundness | 3/5 | Claim that relation errors are "indistinguishable from random" is overstated; random baseline errors are 37-158x, real are ~1.0 |
| Novelty & Contribution | 4/5 | Genuine contribution connecting mech interp, group theory, and synthetic data; functional/structural distinction is valuable |
| Experimental Rigor | 3/5 | No error bars, confidence intervals, or multiple-seed runs; small sample sizes for fact-order experiments |
| Clarity & Writing | 4/5 | Well-written and clearly structured; some notation introduced without definition |
| Related Work Coverage | 3/5 | Missing "Lost in the Middle" (Liu et al., 2024) and grokking/modular arithmetic literature |
| Figures & Tables | 2/5 | Zero figures in a 9-page paper; tables alone cannot convey the U-shaped layer profile or KL matrix |
| Reproducibility | 3/5 | Good method description but missing: hardware, random seeds, SAE hyperparameter sensitivity, code availability |
| Presentation & Structure | 4/5 | Good flow; Discussion section is the strongest part of the paper |

---

## Critical Issues (must fix)

1. **No figures** (Dimension: Figures & Tables)
   - Problem: A 9-page conference paper with 4 tables and 0 figures. The U-shaped layer profile, the positive control calibration curve, the fact-order KL matrix, and the three-way comparison (positive control / real model / random baseline) all need visual treatment.
   - Impact: Reviewers will find the paper hard to parse. The key results are buried in tables that require careful reading.
   - Recommendation: Add at minimum: (1) layer-profile plot showing test error and relation error across layers for both models, (2) positive control noise-vs-error calibration curve with the real model's position marked, (3) KL divergence heatmap for the "room is on fire" case study, (4) bar chart comparing entity-level vs fact-level within/between distance ratios.

2. **Overstated claim about random baseline comparison** (Dimension: Technical Soundness)
   - Problem: The abstract says composition errors are "indistinguishable from random baselines." But Table 1 shows real relation errors ~1.0 while shuffled errors are 37-158. These are distinguishable — the real model's errors are ~1.0, not ~100. The correct claim is that real errors are ~1.0 (the "no structure" regime), not that they equal random.
   - Impact: A careful reviewer will catch this and question the paper's precision.
   - Recommendation: Revise to: "Composition errors for $S_3$ operators fall in the no-structure regime ($\approx 1.0$), far from the $< 0.1$ expected under true group structure." Drop the "indistinguishable from random" phrasing.

3. **No statistical analysis** (Dimension: Experimental Rigor)
   - Problem: No error bars, confidence intervals, bootstrap estimates, or multiple random seeds anywhere in the paper. The fact-order experiment uses only 15 fact sets — are the within/between ratios stable? The positive control uses a single random seed.
   - Impact: A statistics-minded reviewer will question whether the reported numbers are robust.
   - Recommendation: At minimum, report standard deviations for all aggregate metrics (many are already computed in the data files). Run the positive control with 5 different seeds and report variance. For the fact-order experiment, report per-fact-set variation. Consider a permutation test for the null hypothesis that within-set distance = between-set distance.

---

## Important Issues (should fix)

1. **Missing key related work: "Lost in the Middle"** (Dimension: Related Work)
   - Problem: Liu et al. (2024, TACL) showed that LLM performance follows a U-shaped curve with position — best for information at the beginning and end, worst for the middle. This is directly relevant to the fact-ordering results and the recency effect in the case study.
   - Recommendation: Cite and discuss. The "lost in the middle" finding provides independent evidence that ordering matters to LLMs and offers a mechanistic explanation (attention bias) for the recency effect you observe.

2. **Missing key related work: Grokking and modular arithmetic** (Dimension: Related Work)
   - Problem: Nanda et al. (2023, ICLR) showed that small transformers trained on modular addition learn representations with explicit group structure — embedding inputs as rotations on a circle. This is a case where group structure DOES emerge in a transformer, contradicting the paper's implicit suggestion that transformers can't learn group structure.
   - Recommendation: Cite and discuss as a positive case. The key distinction: modular arithmetic is a task where the group structure IS the task. Next-token prediction on natural text is not. This strengthens your argument about when group structure emerges.

3. **Table 1 mixes two model scales awkwardly** (Dimension: Presentation)
   - Problem: Table 1 shows 0.5B and 1.5B side-by-side, but the layers don't align (layer 21 in the 0.5B is a different relative depth than layer 21 in the 1.5B). The positive control and random baseline rows at the bottom reference only 0.5B.
   - Recommendation: Either normalize by relative depth (layer/n_layers) or present the two scales in separate tables/figure panels. Add 1.5B random baseline for completeness.

4. **The Simula critique needs careful hedging** (Dimension: Technical Soundness)
   - Problem: Section 5.3 claims Simula's product-space decomposition "misses" correlations. But Simula doesn't literally permute facts — it decomposes conceptual categories. The paper should be precise about what exactly the Simula framework assumes vs. what the broader field assumes.
   - Recommendation: The current text actually handles this reasonably, but the Level 2 implication ("Product-space decomposition has blind spots") could use a sentence acknowledging that Simula's mixing strategies partially address this. The critique is about the general product-space assumption, not about Simula specifically.

5. **Positive control dimensionality gap** (Dimension: Technical Soundness)
   - Problem: The positive control runs in $d = 10$ but the real model has $d = 896$. The paper acknowledges this in Limitations but doesn't adequately explain why the $d = 10$ result is still informative. A reviewer will ask: "what if the pipeline fails at $d = 896$ even with real group structure?"
   - Recommendation: Add a sentence explaining that at $d = 10$ with $n = 200$ samples, the system is well-conditioned (n > d). The real model at $d = 896$ with $n = 100$ is underdetermined, but the real model's test errors of 0.08 demonstrate the system IS well-conditioned for the actual data (because neural activations are low-rank). The positive control proves the *group relation testing* is sound; the operator fitting's success on real data proves the *linear algebra* is sound.

---

## Suggestions (nice to have)

1. The functional/structural equivariance distinction (Section 5.2) is the most original conceptual contribution. Consider promoting it earlier — perhaps framing it in the Introduction as the paper's central conceptual insight, not just a post-hoc interpretation.

2. The human cognition parallel in Section 5.2 is evocative but brief. Consider cutting it to one sentence or expanding it with a citation — currently it's in an awkward middle ground.

3. Consider adding an "Experiment Overview" figure or table at the start of Section 4 summarizing all 10+ experiments with their results in one glance. The current structure requires reading the entire Results section to see the pattern.

4. The title is strong. Consider whether "The Absence of" could be shortened to just "No Permutation Group Structure in Language Model Representations" for punchiness.

5. The Even KL / Odd KL column in Table 4 (0.366 / 0.231) is interesting but unremarked upon in the text. The paper should either discuss it (even permutations = 3-cycles produce larger KL) or remove the column.

---

## Missing Related Work

- **"Lost in the Middle"** (Liu et al., TACL 2024, arXiv 2307.03172)
  - Relevance: Shows LLMs have strong position bias — performance depends on where information appears in context. Directly relevant to fact-ordering results.
  - Recommendation: Cite and discuss in Section 2.2 (Prior Work) and connect to Section 4.3 (case study showing recency effect).

- **"Progress Measures for Grokking via Mechanistic Interpretability"** (Nanda et al., ICLR 2023, arXiv 2301.05217)
  - Relevance: Transformers trained on modular arithmetic learn explicit group structure (circular representations). Important positive case showing transformers CAN learn group representations when the task demands it.
  - Recommendation: Cite in Section 2.2 and discuss in Section 5.1 as evidence for explanation (C) — the training objective matters.

- **Bricken et al., "Towards Monosemanticity" (Anthropic, 2023)** and **Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Directions" (2023)**
  - Relevance: The SAE methodology references in literature.md are not cited in the paper itself.
  - Recommendation: Add citations when describing SAE training in Section 3.5.

---

## Next Step

Run `/pub-revise latent-symmetries.1` to create version 2 incorporating this review.
