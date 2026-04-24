# Review: latent-symmetries.4

**Reviewer:** Claude (automated paper review)
**Date:** 2026-04-24
**Paper reviewed:** `paper/latent-symmetries.4/paper.tex`

---

## Overall Assessment: STRONG

**Score: 38/40**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Technical Soundness | 5/5 | All claims audit-verified; capacity argument now well-stated |
| Novelty & Contribution | 5/5 | Functional/structural equivariance distinction, fact-order experiments, and synthetic data implications form a compelling trifecta |
| Experimental Rigor | 4/5 | Comprehensive with controls; single seed remains but is acknowledged |
| Clarity & Writing | 5/5 | Excellent throughout; the "lookup table, not an algebra" line is memorable |
| Related Work Coverage | 5/5 | Comprehensive; no significant gaps identified |
| Figures & Tables | 5/5 | Figure 1b now shows full 0--1.2 range with green zone — visually unambiguous |
| Reproducibility | 5/5 | Hardware, seeds, hyperparams, code URL, SAE details all present |
| Presentation & Structure | 4/5 | Two minor structural issues remain |

---

## Previous Review Issues: Verification

v3 review (36/40) had 3 important issues:
- [x] **Code availability misplaced**: Moved to standalone paragraph after Conclusion — correct placement
- [x] **Figure 1b y-axis misleading**: Widened to 0--1.2 with green "group structure detected" zone — much improved
- [x] **Capacity argument imprecise**: Revised to discuss degrees of freedom vs. generalization tradeoff — now sound

No regressions. No stale references. No placeholder text. All citations verified by prior audit.

---

## Critical Issues (must fix)

None.

---

## Important Issues (should fix)

1. **Figure 1 caption doesn't mention the green zone** (Dimension: Figures & Tables)
   - Problem: The Figure 1 caption describes the gray dotted line at 1.0 as "the error expected from unrelated operators" but doesn't mention the green shaded region at the bottom labeled "Group structure detected." Captions should be self-contained — a reader looking at the figure without reading the text should understand what the green zone means.
   - Recommendation: Add to caption: "The green region ($<0.1$) indicates where errors would fall if true group structure were present, as validated by our positive control (Figure~2)."

2. **The paper would benefit from a brief discussion of superposition** (Dimension: Novelty & Contribution)
   - Problem: The paper argues that the model uses a "lookup table" of independent per-permutation operators. An alternative explanation is that the model represents permutations in superposition (Elhage et al., 2022) — compressing more features than dimensions allows, which would break algebraic structure. This is not a contradiction of the paper's thesis but an additional mechanistic explanation that a reviewer familiar with mech interp will look for.
   - Recommendation: Add 1-2 sentences in Section 5.1(C) or 5.2: "An additional factor may be superposition \cite{elhage2022superposition}: if the model represents permutation information in superposition with other features, the resulting interference would prevent operators from satisfying exact algebraic relations even if the underlying computation has group structure." This strengthens rather than weakens the argument.

---

## Suggestions (nice to have)

1. The paper is now 11 pages with well-used space. If a venue requires 10 pages, the most natural cut is Table 2 (positive control), which is redundant with Figure 2. The figure alone with an expanded caption would save ~0.5 pages.

2. Consider submitting to the Mechanistic Interpretability Workshop at ICML 2026 (mechinterpworkshop.com) — the functional/structural equivariance distinction and the positive control methodology would be of direct interest to that community.

3. The last sentence of the Conclusion ("Symmetry in representations is earned by symmetry in the world") is an excellent closing line. No change needed.

---

## Missing Related Work

- **"Toy Models of Superposition"** (Elhage et al., arXiv 2209.10652, 2022)
  - Relevance: Establishes that neural networks store more features than dimensions via superposition, which would prevent clean algebraic structure in shared representational space. Relevant to explaining WHY the operators don't compose.
  - Recommendation: Optional but would strengthen Section 5. A 1-sentence cite would suffice.

No other significant gaps identified.

---

## Next Step

Score 38/40 with 0 critical issues. Well above convergence threshold.

The paper is ready for submission. The 2 remaining important issues are minor polish. A final `/pub-revise` could address them, or the paper can be submitted as-is.
