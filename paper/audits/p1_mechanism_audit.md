# P1 mechanism (i)/(ii) referent invariant

**Date:** 2026-05-02
**Branch:** final-stretch/p1-assumption
**Audit performed during:** Phase 1 P1 sprint, pre-push verification

## Invariant (DO NOT VIOLATE)

| Label | Mechanism name | Model | Empirical signature |
|---|---|---|---|
| **(i)** | Geometric | Llama-3.1-8B | $\|\hat P \mathbf h\|$ **decreases** under prefill on 89.5% of pairs (n=1000) |
| **(ii)** | Distribution-shift | Mistral-7B | $\|\hat P \mathbf h\|$ **increases** under prefill on 92.1% of pairs (n=1000); $\tilde\Delta$ misaims |

A swap inverts §5.2.1's evidence attribution. Reviewer-facing claim "Llama supports the mechanism, Mistral falsifies it" depends on this mapping holding across 11 paper locations + figure caption + table.

## Source data

- `results/final_stretch/p1_assumption_verification/{mistral-7b,llama3-1-8b}/summary.json`
- `results/final_stretch/p1_assumption_verification/{mistral-7b,llama3-1-8b}/per_pair.jsonl` (n=1000 each)
- Frozen snapshot: `_dev/reviews/02MAY/p1_evidence_snapshot_20260502_1304/`

## Locations (paper/main.tex, 2026-05-02 state)

11 occurrences of `mechanism (i)` and/or `(ii)` form across body + appendix:

| # | Line | Section | Occurrence | Verified consistent |
|---|---|---|---|---|
| 1 | 54 | §1 Contrib 1 | "geometric mechanism" → Llama; "distribution shift" → Mistral | ✓ |
| 2 | 167 | §5.2 Proposition title | "heterogeneous mechanisms across models, one verified, one inferred" | ✓ |
| 3 | 167 | §5.2 Proposition body | "(i) Geometric (Llama)" + "(ii) Distribution-shift (Mistral)" | ✓ |
| 4 | 167 | §5.2 footnote 1 | "(i) verified" → Llama; "(ii) implied as residual" → Mistral | ✓ |
| 5 | 169 | §5.2 Corollary Sketch | "by mechanism (i) if h is geometrically close" / "by (ii) if learned correction misaims" | ✓ |
| 6 | 175 | §5.2.1 Llama paragraph | "Llama supports mechanism (i)" → 89.5% ↓ | ✓ |
| 7 | 177 | §5.2.1 Mistral paragraph | "Mistral falsifies mechanism (i)" → 92.1% ↑ ; residual is (ii) | ✓ |
| 8 | 269 | §7 Conclusion | "geometric on Llama, distribution-shift on Mistral" | ✓ |
| 9 | 639 | App. T figure caption | Llama supporting (i); Mistral falsifying (i), residual (ii) | ✓ |
| 10 | 643 | App. T cosine prose | "Llama where ‖P̂h‖ confirms mechanism (i)" | ✓ |
| 11 | 648 | App. T table caption | "predicted shift under mechanism (i): increase" | ✓ |

Each location carries a TeX comment `% Mechanism (i)/(ii) referent fixed: see paper/audits/p1_mechanism_audit.md` — search the .tex for those comments to find the referent-locked clauses.

## What to check before any future edit

If you edit any paragraph carrying the audit comment:

1. Confirm Llama is associated with (i)=Geometric=‖P̂h‖↓
2. Confirm Mistral is associated with (ii)=Distribution-shift=‖P̂h‖↑/Δ̃-misaim
3. If you swap models or swap (i)/(ii), update ALL 11 locations
4. The figure caption (line 639) and table caption (line 648) are also referent-bearing — don't miss them

## Why this matters

§5.2.1 is the new contribution from the P1 sprint. The disaggregation finding ("Llama supports, Mistral falsifies") rests on a single consistent labeling. A typo that swaps "(i)" for "(ii)" in any location creates an internal inconsistency a reviewer will catch. The audit verifies all 11 locations were correct as of 2026-05-02; the inline comments make the invariant visible to future maintainers without requiring re-audit.
