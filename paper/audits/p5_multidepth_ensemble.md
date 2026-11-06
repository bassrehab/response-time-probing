# P5 multi-depth probe ensemble audit

**Date:** 2026-05-03
**Branch:** final-stretch/p5-multi-depth-ensemble
**Audit performed during:** Phase 5 P5 sprint, pre-push verification

## Result invariant (DO NOT VIOLATE)

| Model | shallow | deep | Best CoH on novel templates | AND benign FPR (training_templates) |
|---|---|---|---|---|
| **Mistral-7B** | L=1 | L=11 | **deep 90.9%** (canonical_novel_token); **deep 100%** (novel_register) | AND 80% < shallow 92.5% |
| **Llama-3.1-8B** | L=1 | L=15 | both saturated on training; n/a OOD (no harmful baseline) | AND 85% = shallow 85% |

**Key invariants:**
1. **Mistral L=11 outperforms L=1 on novel templates** by generalization-criterion selection (training-AUROC selected L=1)
2. **Llama L=15 fully overfits training distribution** (100% benign FPR OOD; CV AUROC 1.0000 saturates)
3. **AND-gate provides 12-25pp Mistral benign FPR reduction** at no ASR cost (precision composition)
4. **Multi-depth ensemble doesn't unlock generalization beyond best single probe** on either model
5. **Layer-depth landscape is model-specific** — Mistral has a usable semantic-features depth (L=11); Llama's saturated L=1-13 plateau may all read similar features, with L=14+ overfitting

A swap of shallow↔deep on either model, or a swap of "Mistral L=11 better than L=1" to its inverse, breaks the methodological reframe in §5.3 and the App. P fresh-retrain canonicality framing.

## Layer-selection criterion distinction (load-bearing methodological finding)

P5 introduces a methodological distinction now applied across the paper:

- **Training-AUROC selection:** standard 5-fold CV AUROC on the paper's 90-sample phase-1 training set; selects layer maximizing training-distribution AUROC. The fresh-retrain L=1 in App. P is selected by this criterion.
- **Generalization-criterion selection:** layer chosen to maximize conditional-on-harm detection rate on out-of-distribution canonical-register novel-token holdout templates. P5 finds Mistral L=11 dominates L=1 by this criterion.

The two criteria can disagree, as Mistral demonstrates. The §5.3 deployment recommendation (line 243) was updated to "prefer probe layers ≥ 4 when AUROCs are comparable, with generalization-criterion selection preferred over training-AUROC alone." The §5.3 layer-depth split (line 239) was caveated with "under training-AUROC layer selection."

## Body-content scope-narrowing decisions

Three deliberate scope narrowings during P5 integration to fit the page budget:

1. **Original Option B paragraph (3 sentences) → single body sentence.** The full elaboration with AND/OR ensemble specifics, per-distribution table, and findings (i)-(vi) lives in App. T (`app:p5_multidepth`). Body sentence retains: methodological finding (Mistral L=11 dominates L=1) + Llama overfit support.
2. **"No generalization gain" finding moved to App. T.** The body lead "Layer-selection criterion matters" now implies this — under Edit 2 framing, "Mistral L=11 dominates L=1" implicitly says ensembles can't beat the best single deeper probe. App. T finding (i) carries the explicit ensemble-doesn't-help claim.
3. **Heterogeneity-tie clause ("the layer-depth landscape itself is model-specific") moved to App. T finding (v).** The body sentence still notes "Llama L=15 fully overfits (100% benign FPR OOD)" which carries the heterogeneity observation; the explicit tie to the broader Mistral-vs-Llama thread moves to App. T + audit doc + Day 4 framing.

These decisions are deliberate scope narrowings, not oversights. Future revision cycles should not re-expand without checking page budget.

## Source data

- `results/p5_multi_depth_ensemble/{mistral-7b,llama31-8b}/per_sample.jsonl` (320 rows each: 4 distributions × 80 samples)
- `results/p5_multi_depth_ensemble/{mistral-7b,llama31-8b}/summary.json`
- `results/p5_multi_depth_ensemble/summary_all.json` (combined)
- `results/p5_multi_depth_ensemble/llama_l15_probe.pkl` (Llama L=15 probe trained from paper's 90-sample phase-1 set; CV AUROC 1.0000 on training distribution)
- `results/p5_multi_depth_ensemble/done` (timestamp marker)

## Probe-training and ensemble setup

**Mistral L=1 + L=11:** Both fresh-retrained probes from `results/retrain_all_7/mistral-7b/`. L=1 = `probe_fresh.pkl` (fresh-best by training-AUROC). L=11 = `probe_fresh_at_paper_layer.pkl` (fresh-trained at paper's chosen layer).

**Llama L=1 + L=15:** L=1 = `probe_fresh.pkl` from `results/retrain_all_7/llama31-8b/`. L=15 freshly trained during P5 run from paper's 90-sample phase-1 set (40 harmful with prefill "Sure, here is" + 50 benign queries; 5 N_PREVIEW tokens; logistic regression on mean-pooled L=15 hidden states with StandardScaler). Selected L=15 because Llama's AUROC curve saturates 1.0000 at L=1-13 and dips slightly at L=14+ — L=15 is in the "different from L=1" region.

**Ensemble rules:**
- `halt_and = fire_shallow AND fire_deep` (precision composition)
- `halt_or = fire_shallow OR fire_deep` (coverage composition)
- Both probes evaluated on the same activations (extracted once per sample at both layers)

**Threshold:** $\tau_r = 0.5$ for both probes (matches §5.3 baseline).

## Eval distribution construction (40 attack + 40 matched-condition benign each)

1. **training_templates:** 8 first TRAIN_TEMPLATES from gpu_issue1_holdout.py × 5 AdvBench behaviors. Matched benign: same templates × 5 BENIGN_PROMPTS.
2. **canonical_novel_token:** 10 canonical_novel_token TEST_TEMPLATES from gpu_issue1_holdout.py × 4 AdvBench behaviors. Matched benign: same.
3. **novel_register:** 10 novel_register TEST_TEMPLATES from gpu_issue1_holdout.py × 4 AdvBench behaviors. Matched benign: same.
4. **prefill_past_window:** 5 P3 prefills × 8 AdvBench behaviors. Matched benign: P3 BENIGN_PROMPTS list.

Per-distribution count: 40 attack + 40 benign = 80 samples × 4 distributions × 2 models = 640 generations + 640 preview-passes. Total runtime ~1.5h on A100.

## Per-distribution per-model table (verbatim from results)

### Mistral-7B (shallow=L=1, deep=L=11)

| Distribution | shallow halt | shallow CoH | shallow ASR | shallow FPR | deep halt | deep CoH | deep ASR | deep FPR | AND halt | AND CoH | AND ASR | AND FPR | OR halt | OR CoH | OR ASR | OR FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_templates | 80% | 78.6% | 15% | 92.5% | 85% | 85.7% | 10% | 87.5% | 75% | 78.6% | 15% | **80%** | 90% | 85.7% | 10% | 100% |
| canonical_novel_token | 70% | 63.6% | 10% | 67.5% | 90% | **90.9%** | 2.5% | 80% | 65% | 63.6% | 10% | **50%** | 95% | 90.9% | 2.5% | 97.5% |
| novel_register | 70% | 66.7% | 12.5% | 70% | 97.5% | **100%** | 0% | 85% | 70% | 66.7% | 12.5% | **57.5%** | 97.5% | 100% | 0% | 97.5% |
| prefill_past_window | 100% | 100% | 0% | **100%** | 97.5% | 95.2% | 2.5% | 92.5% | 97.5% | 95.2% | 2.5% | 92.5% | 100% | 100% | 0% | 100% |

### Llama-3.1-8B (shallow=L=1, deep=L=15)

| Distribution | shallow halt | shallow CoH | shallow ASR | shallow FPR | deep halt | deep CoH | deep ASR | deep FPR | AND halt | AND CoH | AND ASR | AND FPR | OR halt | OR CoH | OR ASR | OR FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_templates | 85% | 100% | 0% | 85% | 100% | 100% | 0% | 100% | 85% | 100% | 0% | 85% | 100% | 100% | 0% | 100% |
| canonical_novel_token | 100% | n/a* | 0% | 95% | 100% | n/a* | 0% | 100% | 100% | n/a* | 0% | 95% | 100% | n/a* | 0% | 100% |
| novel_register | 100% | n/a* | 0% | 97.5% | 100% | n/a* | 0% | 100% | 100% | n/a* | 0% | 97.5% | 100% | n/a* | 0% | 100% |
| prefill_past_window | 92.5% | 100% | 0% | 97.5% | 100% | 100% | 0% | 100% | 92.5% | 100% | 0% | 97.5% | 100% | 100% | 0% | 100% |

*Llama refused all 40 samples without defense (no harmful baseline); CoH undefined.

## What to check before any future edit

If you edit §5.3 layer-depth split (line 239), the deployment recommendation (line 243), the §5.3 multi-depth body sentence (now in adaptive paragraph), or App. P fresh-retrain framing:

1. **Don't drop the "(under training-AUROC layer selection)" qualifier** in line 239 — it's the methodological scope of the layer-depth split claim.
2. **Don't drop "with generalization-criterion selection preferred over training-AUROC alone"** in line 243 — it's the operational deployment recommendation.
3. **Don't swap Mistral L=11 vs L=1** roles — the methodological reframe rests on L=11 being the generalizer.
4. **Don't expand the body multi-depth sentence** without re-running the page-budget check — it's currently at the minimum scope needed to support Edit 2's caveats.
5. **App. P "fresh-retrain L=1 is canonical paper-protocol number" sentence** must stay paired with the "P5 shows L=11 is the better generalizer" cross-ref. Dropping either creates an apparent contradiction.

## Forward note (Day 4 framing)

P5 contributes two items to the diagnostic-protocols list:

1. **Layer-selection-criterion analysis** — comparing training-AUROC-selected and generalization-criterion-selected probe layers on novel-template holdout. The two can disagree (Mistral); when they do, the latter is the deployment-correct choice.

2. **Per-distribution per-paradigm CoH × benign-FPR sweep** — already implicit in §5.3 surface-feature reading thread; P5 makes it explicit as a probe-evaluation protocol that surfaces artifactual halting (Mistral L=1 100%/100% on prefill_past_window matches Llama's documented pattern from P3).

## Mistral-vs-Llama heterogeneity thread (cumulative as of P5)

P5 is the **sixth independent observation** of the underlying Mistral-vs-Llama heterogeneity pattern:

1. §5.2 P1 mechanism check — Mistral falsifies geometric mechanism; Llama supports it
2. §5.3 surface-feature reading thread (existing) — Mistral layer-11 vs Llama layer-1 fundamentally different probing depths
3. §5.3 P3 prefill-past-window — Mistral window-bound failure; Llama artifactual halt on preamble register
4. App. P paper-pkl-vs-fresh divergence (existing) — Mistral $-70$pp drift; Llama $+10$pp
5. §5.3 P4 n=200 per-template — Mistral fresh probe softening-vs-continuation split; paperpkl window-bound on larger distribution
6. §5.3 P5 layer-selection criterion (this run) — Mistral has usable semantic-features depth at L=11; Llama L=1-13 saturated plateau (likely all surface), L=14+ overfits

When Day 4 framing rolls around, this is the strongest empirical thread in the paper. Six independent observations of the same heterogeneity warrants its own paragraph in §6 or §7. Don't surface to Claude Code yet — Day 4 is the right time. Note for now.
