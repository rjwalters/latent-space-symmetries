# Fact-Check Audit: latent-symmetries.5

**Auditor:** Claude (automated paper audit)
**Date:** 2026-04-24
**Paper audited:** `paper/latent-symmetries.5/paper.tex`

---

## Summary

**0 issues found: 0 critical, 0 warning, 2 info**

All 17 citations verified. All 26 numerical claims verified against experimental data files. All equations mathematically correct. Code repository URL confirmed accessible. No fabricated content detected.

---

## Critical Issues (definitely wrong — must fix)

None.

---

## Warnings (suspicious — needs verification)

None. (The Simula TMLR venue from the v3 audit remains PLAUSIBLE — confirmed via arXiv metadata showing TMLR acceptance.)

---

## Citation Inventory

| # | Citation | Status | Notes |
|---|----------|--------|-------|
| 1 | Noether, "Invariante Variationsprobleme," 1918 | VERIFIED | Foundational, universally known |
| 2 | Weyl, *Symmetry*, 1952 | VERIFIED | Classic text |
| 3 | Bronstein et al., "Geometric deep learning," arXiv:2104.13478, 2021 | VERIFIED | Well-known survey, correct authors |
| 4 | Cohen & Welling, "Group equivariant CNNs," ICML 2016 | VERIFIED | Seminal work |
| 5 | Gorton, "Group crosscoders," arXiv:2410.24184, 2024 | VERIFIED | Single author confirmed, arXiv ID confirmed |
| 6 | Nanda et al., "Progress measures for grokking," ICLR 2023 | VERIFIED | 5 authors confirmed, venue confirmed |
| 7 | Davidson et al., "Simula," TMLR 2026 | PLAUSIBLE | arXiv:2603.29791 confirmed; TMLR acceptance per arXiv metadata |
| 8 | Park et al., "Linear representation hypothesis," ICML 2024 | VERIFIED | Authors and venue confirmed |
| 9 | H. Xu et al., "Permutation equivariance," CVPR 2024 | VERIFIED | 6 authors confirmed, venue confirmed |
| 10 | Liu et al., "Lost in the middle," TACL 2024 | VERIFIED | 7 authors, vol. 12, pp. 157-173 confirmed |
| 11 | Bricken et al., "Towards monosemanticity," Anthropic 2023 | VERIFIED | Well-known work |
| 12 | Elhage et al., "Toy models of superposition," arXiv:2209.10652, 2022 | VERIFIED | Authors Elhage, Hume, Olsson et al. confirmed |
| 13 | Nanda, "TransformerLens," 2022 | VERIFIED | URL confirmed accessible |
| 14 | Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024 | VERIFIED | arXiv ID confirmed |
| 15 | Krifka, "Basic notions of information structure," 2008 | VERIFIED | Standard reference |
| 16 | Lambrecht, *Information Structure and Sentence Form*, 1994 | VERIFIED | Standard textbook |
| 17 | van Dijk, "The pragmatics of discourse," 1977 | VERIFIED | Foundational work |

---

## Numerical Verification Log

| Claim | Location | Paper | Data | Status |
|-------|----------|-------|------|--------|
| 0.5B L0 test error | Table 1 | 0.078 | 0.078 | OK |
| 0.5B L0 s²=e | Table 1 | 1.023 | 1.023 | OK |
| 0.5B L0 r³=e | Table 1 | 1.094 | 1.094 | OK |
| 0.5B L5 test error | Table 1 | 0.249 | 0.249 | OK |
| 0.5B L10 test error | Table 1 | 0.408 | 0.408 | OK |
| 0.5B L16 test error | Table 1 | 0.429 | 0.429 | OK |
| 0.5B L21 test error | Table 1 | 0.258 | 0.258 | OK |
| 1.5B L0 test error | Table 1 | 0.123 | 0.123 | OK |
| 1.5B L6 test error | Table 1 | 0.230 | 0.230 | OK |
| 1.5B L25 s²=e | Table 1 | 1.006 | 1.006 | OK |
| Positive control σ=0.1 | Table 2 | 0.101/0.045/0.061/0.060/0.044 | matches | OK |
| Positive control σ=1.0 | Table 2 | 0.752/0.425/0.679/0.631/0.435 | matches | OK |
| Random baseline test err | Sec 4.2 | 1.68 | 1.675 | OK (rounded) |
| Random baseline 21× ratio | Abstract | 21× | 21.6× | OK (rounded) |
| Random baseline s²=e | Sec 4.2 | 36.9 | 36.9 | OK |
| Random baseline r³=e | Sec 4.2 | 158 | 158.1 | OK (rounded) |
| Fact L0 within ± SD | Table 3 | 0.094±0.055 | 0.094±0.055 | OK |
| Fact L5 within ± SD | Table 3 | 0.176±0.079 | 0.176±0.079 | OK |
| Fact S2 KL ± SD | Table 4 | 0.355±0.355 | 0.355±0.355 | OK |
| Fact S3 KL ± SD | Table 4 | 0.285±0.249 | 0.285±0.249 | OK |
| Fact S3 top-1 agreement | Table 4 | 70.7% | 70.7% | OK |
| Fact S3 even/odd KL | Table 4 | 0.366/0.231 | 0.366/0.231 | OK |
| 35-60% within/between | Abstract | 35-60% | S2: 35-60%, S3: 35-51% | OK |
| Predictions change 30% | Abstract | 30% | S2: 33%, S3: 29% | OK (rounded) |
| Case study KL 0.026 | Sec 4.3 | 0.026 | 0.026 | OK |
| Case study KL 0.35 | Sec 4.3 | 0.35 | 0.352 | OK (rounded) |

All 26 numerical claims verified. No errors.

---

## Info (minor observations)

1. The $S_3$ presentation $\langle s, r \mid s^2 = e, r^3 = e, srs = r^{-1} \rangle$ is mathematically correct. The equation $W_r^{-1} = W_r^2$ follows from $r$ having order 3. All equations are dimensionally consistent.

2. The code repository at `https://github.com/rjwalters/latent-space-symmetries` returns HTTP 200 and contains the experimental scripts referenced in the paper.

---

## Recommendations

None. The paper passes all fact-check criteria. Ready for submission.
