# Review: latent-symmetries.2

**Reviewer:** Claude (automated paper review)
**Date:** 2026-04-22
**Paper reviewed:** `paper/latent-symmetries.2/paper.tex`

---

## Overall Assessment: STRONG

**Score: 34/40**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Technical Soundness | 4/5 | Claims now precisely stated; positive control argument about dimensionality is adequate but could be stronger |
| Novelty & Contribution | 4/5 | Functional/structural equivariance distinction is valuable; fact-order findings are actionable |
| Experimental Rigor | 4/5 | Standard deviations added; still single random seed for main experiments |
| Clarity & Writing | 5/5 | Excellent prose; abstract is accurate; notation consistent throughout |
| Related Work Coverage | 4/5 | All major gaps from v1 addressed; good integration of grokking and Lost in the Middle |
| Figures & Tables | 4/5 | Four strong figures; all key results now have visual treatment |
| Reproducibility | 4/5 | Hardware, seeds, SAE hyperparams now specified; code availability not mentioned |
| Presentation & Structure | 5/5 | Introduction effectively previews the functional/structural distinction; Discussion is compelling |

---

## Previous Review Issues: Verification

All 3 critical issues from v1 review addressed:
- [x] **Figures**: 4 figures added (layer profile, positive control, KL heatmap, fact distances)
- [x] **Overstated claim**: "indistinguishable from random" replaced with "no-structure regime (~1.0)" throughout
- [x] **Statistical analysis**: Standard deviations added to fact-order tables and behavioral metrics

All 5 important issues from v1 review addressed:
- [x] **Lost in the Middle**: Cited in Prior Work and connected to case study recency effect
- [x] **Grokking**: Cited in Introduction and Prior Work; used in Discussion to strengthen explanation (C)
- [x] **Table normalization**: Layers now indexed by relative depth with matched positions
- [x] **Simula hedging**: "While Simula's mixing strategies partially address..." added
- [x] **Positive control dimensionality**: Paragraph explaining well-conditioned system and two-part validation argument

All 5 suggestions from v1 addressed:
- [x] Functional/structural distinction promoted to Introduction paragraph 4
- [x] Human cognition parallel trimmed to one sentence
- [x] Even/Odd KL discussed in behavioral impact paragraph
- [x] Title kept as-is (reviewer agreed it was strong)
- [x] Experiment overview suggestion not implemented (reasonable given space; figures now do the job)

No revision-introduced issues detected. No stale cross-references. No placeholder text.

---

## Critical Issues (must fix)

None.

---

## Important Issues (should fix)

1. **Code availability not mentioned** (Dimension: Reproducibility)
   - Problem: The paper describes extensive experiments with custom code but never mentions whether the code or data will be released.
   - Recommendation: Add a sentence in Methods or after Conclusion: "Code and experimental scripts are available at [URL]." Even "Code available upon request" is better than silence.

2. **Table 1 missing 1.5B test errors** (Dimension: Figures & Tables)
   - Problem: The revised Table 1 shows test errors only for the 0.5B model. The 1.5B columns show relation errors but not fit quality, making it hard to compare across scales.
   - Recommendation: Add a 1.5B test error column, or note in the caption that 1.5B test errors are shown in Figure 1a.

---

## Suggestions (nice to have)

1. The abstract is 171 words — well within the 150-250 target and reads well. No changes needed.

2. The paper is 11 pages — slightly over the 8-10 target. The most compressible sections are the "robust" bullet list in 4.1 (could become a single sentence referencing an appendix) and the Future Work section (could be trimmed to 2-3 sentences integrated into Conclusion).

3. Consider adding an author affiliation or institution, even if independent. "Independent researcher" is acceptable and looks better than no affiliation.

4. The bibliography uses a mix of arXiv preprints and published venues. For Bronstein et al. (2021), this was published as a journal article — update the citation if targeting a venue that requires published versions.

---

## Missing Related Work

No significant missing references identified. The revision successfully addressed all gaps from v1. The coverage of group theory + neural networks (Group Crosscoders, grokking, geometric deep learning), mechanistic interpretability (TransformerLens, SAEs, linear representation hypothesis), position sensitivity (Lost in the Middle), synthetic data (Simula), and linguistics (information structure) is now comprehensive for the scope of this paper.

---

## Next Step

Score 34/40 with 0 critical issues meets the convergence criterion (>= 32/40, 0 critical).

The paper is **ready for submission** after addressing the two important issues (code availability, 1.5B test errors in table).

Optionally, run `/pub-audit latent-symmetries.2` for a final fact-check before submission.
