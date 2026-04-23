# Review: latent-symmetries.3

**Reviewer:** Claude (automated paper review)
**Date:** 2026-04-23
**Paper reviewed:** `paper/latent-symmetries.3/paper.tex`

---

## Overall Assessment: STRONG

**Score: 36/40**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Technical Soundness | 5/5 | Claims precisely stated, positive control well-argued, audit-verified citations |
| Novelty & Contribution | 4/5 | Solid contribution; could strengthen by connecting to compositional generalization literature |
| Experimental Rigor | 4/5 | Comprehensive experiments with controls; single seed remains a minor weakness |
| Clarity & Writing | 5/5 | Exceptionally clear; functional/structural distinction introduced naturally |
| Related Work Coverage | 5/5 | All gaps addressed; grokking + Lost in the Middle well-integrated |
| Figures & Tables | 4/5 | Four effective figures; one minor figure-text mismatch |
| Reproducibility | 5/5 | Hardware, seeds, hyperparams, code URL all present |
| Presentation & Structure | 4/5 | Slightly long at 11 pages; code availability awkwardly placed in Limitations |

---

## Previous Review Issues: Verification

v2 review (34/40) had 2 important issues:
- [x] **Code availability**: Added as footnote in Section 6
- [x] **Table 1 missing 1.5B test errors**: 1.5B test error column added

Audit (v3) had 2 critical issues:
- [x] **Group Crosscoders wrong authors**: Fixed — now correctly cites Gorton
- [x] **Permutation equivariance wrong authors**: Fixed — now correctly cites H. Xu et al.

No regressions detected. All prior issues fully resolved.

---

## Critical Issues (must fix)

None.

---

## Important Issues (should fix)

1. **Code availability footnote is misplaced** (Dimension: Presentation)
   - Problem: The code availability statement (`\footnote{\url{...}}`) is the opening sentence of the Limitations section. This is unconventional — code availability is not a limitation. It reads as if the code being public is something to apologize for.
   - Recommendation: Move to a standalone paragraph after Conclusion, before the bibliography. A simple `\paragraph{Code availability.} All code and experimental scripts are available at \url{...}.` placed between Conclusion and the bibliography is the standard convention.

2. **Figure 1b y-axis range hides the story** (Dimension: Figures & Tables)
   - Problem: Figure 1b (group relation errors) uses a y-axis range of 0.8--1.2, making the ~1.0 errors look like they have meaningful variation when they're essentially flat noise around 1.0. The tight range could mislead a reader into thinking there are layer-dependent trends in the relation errors.
   - Recommendation: Either widen the y-axis to 0--1.2 (to show the contrast with the positive control's <0.1) or add a horizontal annotation/arrow showing where "group structure detected" would appear (<0.1). The current range makes 0.5B layer 0's r³=e=1.094 look like a spike when it's just noise.

3. **Discussion paragraph (C) capacity argument needs tightening** (Dimension: Technical Soundness)
   - Problem: Section 5.1(C) claims "A per-permutation linear map is sufficient and has strictly more capacity than a group-constrained representation ($n!$ independent matrices vs. matrices satisfying $O(n!^2)$ algebraic constraints)." The constraint count $O(n!^2)$ is imprecise — composition closure gives $O(n!^2)$ constraints but the presentation relations give far fewer (3 for $S_3$). More importantly, the argument conflates degrees of freedom (which favor unconstrained) with generalization (which favors constrained), without acknowledging this tension.
   - Recommendation: Revise to something like: "A per-permutation linear map has more degrees of freedom than a group-constrained representation, making it easier to fit from limited data — though it sacrifices the compositional generalization that group structure would provide. Since next-token prediction never tests compositional generalization over permutations, the unconstrained solution dominates."

---

## Suggestions (nice to have)

1. The curriculum learning literature (2025-2026) provides additional evidence that data ordering affects LLM training, with some studies showing 18-45% faster convergence. A brief mention in Section 5.3 would strengthen the "ordering is not a nuisance variable" claim with training-time evidence to complement your inference-time evidence.

2. The paper at 11 pages is slightly long. The most compressible material is Table 2 (positive control), which largely duplicates Figure 2. Consider removing the table and keeping only the figure with a more detailed caption, saving ~0.5 page.

3. Consider adding an author email or affiliation. Single-author papers without affiliations can raise flags at some venues.

---

## Missing Related Work

No significant gaps. The paper now covers all major relevant threads comprehensively. One optional addition:

- **Compositional generalization circuits** (e.g., arXiv:2502.15801, 2025): Recent work mechanistically identifies transformer circuits for compositional generalization, finding they use "disentangled representation of token position and identity" rather than algebraic structure. This is consistent with your findings but from the circuits perspective.
  - Recommendation: Optional. Would strengthen Section 5.1(C) but is not essential.

---

## Next Step

Score 36/40 with 0 critical issues. Well above convergence threshold.

The paper is ready for submission. The 3 important issues are polish-level — addressing them would bring the score to 38-39/40.

If desired, run `/pub-revise latent-symmetries.3` for a final polish pass, then submit.
