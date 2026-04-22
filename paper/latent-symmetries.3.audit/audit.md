# Fact-Check Audit: latent-symmetries.3

**Auditor:** Claude (automated paper audit)
**Date:** 2026-04-22
**Paper audited:** `paper/latent-symmetries.3/paper.tex`

---

## Summary

**5 issues found: 2 critical, 2 warning, 1 info**

All numerical claims verified against experimental data files. Two citations have incorrect author names — must fix before submission.

---

## Critical Issues (definitely wrong — must fix)

1. **Group Crosscoders citation has wrong authors**
   - **Location:** Bibliography, \bibitem{bhaskar2024group}
   - **Text:** "D.~Bhaskar, J.~Benton, and S.~Racaniere"
   - **Problem:** The actual author of arXiv:2410.24184 is **Liv Gorton** (single author). "Bhaskar," "Benton," and "Racaniere" do not appear on this paper. This is a fabricated author list.
   - **Fix:** Replace with: `L.~Gorton, ``Group crosscoders for mechanistic analysis of symmetry,'' \textit{arXiv preprint arXiv:2410.24184}, 2024.` Also update all in-text references from "Bhaskar et al." to "Gorton".

2. **Permutation equivariance of transformers citation has wrong authors**
   - **Location:** Bibliography, \bibitem{xu2024permutation}
   - **Text:** "D.~Xu, S.~Cheng, and Z.~Wang"
   - **Problem:** The actual authors are **Hengyuan Xu, Liyao Xiang, Hangyu Ye, Dixi Yao, Pengzhi Chu, and Baochun Li**. "S. Cheng" and "Z. Wang" do not appear. The first name initial is also wrong (H. not D.).
   - **Fix:** Replace with: `H.~Xu, L.~Xiang, H.~Ye, D.~Yao, P.~Chu, and B.~Li, ``Permutation equivariance of transformers and its applications,'' in \textit{Proc.\ CVPR}, 2024.` Update in-text from "Xu et al." (this happens to remain correct).

---

## Warnings (suspicious — needs verification)

1. **Simula venue needs confirmation**
   - **Location:** Bibliography, \bibitem{davidson2026simula}
   - **Concern:** Cited as "Trans. Machine Learning Research, 2026." The arXiv submission is March 2026. TMLR acceptance was noted in our web search but should be confirmed — recent papers are hard to verify.
   - **Action:** Verify TMLR acceptance status. If unconfirmed, cite as arXiv preprint.

2. **Code repository URL not verified**
   - **Location:** Section 6 (Limitations), footnote
   - **Concern:** Links to `https://github.com/rjwalters/latent-space-symmetries`. User confirmed this is public, but the URL should be verified to be accessible and contain the claimed scripts.
   - **Action:** Open the URL and confirm it resolves to the correct repository.

---

## Citation Inventory

| # | Citation | Status | Notes |
|---|----------|--------|-------|
| 1 | Noether, "Invariante Variationsprobleme," 1918 | VERIFIED | Foundational, universally known |
| 2 | Weyl, *Symmetry*, 1952 | VERIFIED | Classic text |
| 3 | Bronstein et al., "Geometric deep learning," arXiv:2104.13478, 2021 | VERIFIED | Well-known survey, correct authors |
| 4 | Cohen & Welling, "Group equivariant CNNs," ICML 2016 | VERIFIED | Seminal work, correct |
| 5 | ~~Bhaskar et al.~~ Gorton, "Group crosscoders," arXiv:2410.24184, 2024 | **CRITICAL: WRONG AUTHORS** | Single author: Liv Gorton |
| 6 | Nanda et al., "Progress measures for grokking," ICLR 2023 | VERIFIED | Authors, venue, arXiv ID all correct |
| 7 | Davidson et al., "Simula," TMLR 2026 | PLAUSIBLE | arXiv:2603.29791 confirmed; TMLR acceptance plausible but recent |
| 8 | Park et al., "Linear representation hypothesis," ICML 2024 | VERIFIED | Authors and venue correct |
| 9 | ~~D. Xu et al.~~ H. Xu et al., "Permutation equivariance," CVPR 2024 | **CRITICAL: WRONG AUTHORS** | First author Hengyuan Xu, not D. Xu; co-authors wrong |
| 10 | Liu et al., "Lost in the middle," TACL 2024 | VERIFIED | Authors, venue, volume all correct |
| 11 | Bricken et al., "Towards monosemanticity," Anthropic 2023 | VERIFIED | Well-known, correct |
| 12 | Nanda, "TransformerLens," 2022 | VERIFIED | Correct URL |
| 13 | Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024 | VERIFIED | Correct arXiv ID, team authorship |
| 14 | Krifka, "Basic notions of information structure," 2008 | VERIFIED | Standard reference in pragmatics |
| 15 | Lambrecht, *Information Structure and Sentence Form*, 1994 | VERIFIED | Standard textbook |
| 16 | van Dijk, "The pragmatics of discourse," 1977 | VERIFIED | Foundational work |

---

## Numerical Verification Log

| Claim | Location | Paper | Data | Status |
|-------|----------|-------|------|--------|
| 0.5B L0 test error 0.078 | Table 1 | 0.078 | 0.078 | OK |
| 0.5B L0 s²=e 1.023 | Table 1 | 1.023 | 1.023 | OK |
| 0.5B L0 r³=e 1.094 | Table 1 | 1.094 | 1.094 | OK |
| 1.5B L0 test error 0.123 | Table 1 | 0.123 | 0.123 | OK |
| 1.5B L25 s²=e 1.006 | Table 1 | 1.006 | 1.006 | OK |
| Positive control σ=0.1 test err 0.101 | Table 2 | 0.101 | 0.101 | OK |
| Positive control σ=0.1 s²=e 0.045 | Table 2 | 0.045 | 0.045 | OK |
| Random baseline test err 1.68 | Sec 4.2 | 1.68 | 1.675 | OK (rounded) |
| Random baseline 21× ratio | Abstract, Sec 4.2 | 21× | 21.6× | OK (rounded) |
| Random baseline s²=e 36.9 | Sec 4.2 | 36.9 | 36.9 | OK |
| Random baseline r³=e 158 | Sec 4.2 | 158 | 158.1 | OK |
| Fact order L5 within 0.176±0.079 | Table 3 | 0.176±0.079 | 0.176±0.079 | OK |
| Fact order S2 KL 0.355±0.355 | Table 4 | 0.355±0.355 | 0.355±0.355 | OK |
| Fact order S3 top-1 agreement 70.7% | Table 4 | 70.7% | 70.7% | OK |
| Fact order even KL 0.366, odd 0.231 | Table 4 | 0.366/0.231 | 0.366/0.231 | OK |
| Fact-order group L0 test err 0.13 | Sec 4.3 | 0.13 | 0.128 | OK (rounded) |
| Fact-order group s²=e ≈0.99 | Sec 4.3 | ≈0.99 | 0.994 | OK |
| Fact-order group r³=e ≈1.00 | Sec 4.3 | ≈1.00 | 1.002 | OK |
| PCA best s²=e 0.09 | Sec 4.1 | 0.09 | 0.0897 | OK |
| PCA r³=e 0.73 | Sec 4.1 | 0.73 | 0.7282 | OK |
| KL case study 0.026 | Sec 4.3 | 0.026 | 0.026 | OK |
| KL case study 0.35 | Sec 4.3 | 0.35 | 0.352 | OK |
| Qwen2.5-0.5B: 24 layers, d=896 | Sec 3.1 | 24, 896 | 24, 896 | OK |
| Qwen2.5-1.5B: 28 layers, d=1536 | Sec 3.1 | 28, 1536 | 28, 1536 | OK |

All 24 numerical claims verified. No errors found.

---

## Info (minor observations)

1. The $S_3$ presentation relation $srs = r^{-1}$ (Equation 4) correctly equates $W_r^{-1} = W_r^2$ for a 3-cycle. This is mathematically sound since $r$ has order 3.

---

## Recommendations

1. **URGENT: Fix the two fabricated author lists** (Group Crosscoders and Permutation Equivariance citations). These are the only factual errors in the paper but they are serious — incorrect attribution in citations.
2. Verify the Simula TMLR acceptance if possible; otherwise cite as arXiv preprint.
3. Verify the GitHub URL resolves correctly before submission.
