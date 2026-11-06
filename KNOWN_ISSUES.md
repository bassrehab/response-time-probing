# Known issues in the evaluation code for arXiv:2606.29441

An audit of this repository's evaluation code (November 2026) found errors that the experiments in
*Closing the Activation-Cone Blind Spot: Response-Time Probing and Unified Defense*
(arXiv:2606.29441) depend on, so the paper's empirical results are not supported and its numbers
should not be reused. The paper's corollary is a theoretical argument and does not depend on these
experiments. The code is kept public for transparency.

Each issue below was checked against the code and the stored result files. Paths are relative to
the repository root.

## 1. The "prefilling attack" closes the assistant turn

Prefilled prompts are built with
`apply_chat_template([user, {"role": "assistant", "content": prefill}], add_generation_prompt=False)`
and never use `continue_final_message=True` (`src/eval/safety_eval.py:48`). The chat templates of
the evaluated models append an end-of-turn token after the assistant message (`</s>`,
`<|eot_id|>`, `<|im_end|>`, `<end_of_turn>`). The model therefore sees a *completed* assistant turn
and starts a new one, instead of continuing the prefill.

On Llama-3.1-8B, 30 of 40 "prefilled" responses begin a fresh `assistant` turn
(`results_pod3_final/llamaguard_baseline/llama31-8b_samples.json`).

Every prefilling experiment uses the same construction, including:
- `scripts/gpu_response_probe_full.py:65`
- `scripts/gpu_adaptive_attack.py:80`
- `scripts/gpu_issue1_holdout.py:189`
- `scripts/gpu_retrain_all_7.py:138`
- `scripts/gpu_n200_dual_probe_rerun.py:187`
- `scripts/p2_unified_mistral_n40.py:193`

**Affected:** every prefilling number in the paper. That includes the cross-paradigm comparison
behind the "structural blind spot" claim, which is therefore not established.

## 2. The keyword judge scores prefilled refusals as harmful

- Four of the eight refusal patterns contain an uppercase `I` (`src/eval/judge.py:15-17,21`), but
  they are matched against lowercased text (`src/eval/judge.py:54,77`), so they never fire.
- The attacker's prefill is prepended to the response before it is judged
  (`src/eval/safety_eval.py:73`). Six of the eight prefill templates match the judge's compliance
  patterns on their own.

As a result, `"I can't help with that."` is scored safe, but
`"Sure, here's how to do that. First, I can't help with that."` is scored **harmful**.

**Affected:** all keyword-judge attack success rates for prefilling. They are inflated most on
Llama-3.1-8B, whose "successful" prefilling samples are largely refusals.

## 3. The response-time probe is evaluated in-sample, and its labels are set by the prefill template

- **Labels:** probe training labels are the keyword judge's verdict on the prefill plus 5 generated
  tokens (`scripts/gpu_response_probe_full.py:117-118`). The judge flags 6 of the 8 templates from
  the template text alone, which is why every model has 33/40 positives (31/40 on Gemma-2).
- **Same prompts for training and evaluation:** `generate_prefilling_attacks` reseeds with a fixed
  seed on every call (`src/attacks/prefilling.py:43`). The halt evaluation therefore reuses the
  exact 40 training prompts (`scripts/gpu_response_probe_full.py:317,376`;
  `scripts/gpu_phase4_dual_probe.py:63`).
- **Benign FPR:** it is measured on the probe's own training negatives
  (`scripts/gpu_phase4_dual_probe.py:82`).
- **Halting depends only on the template.** On Mistral, the probe halts all 33 prompts with flagged
  templates and none of the 7 with the two unflagged templates. Its scores are bimodal: at most
  0.005 when it doesn't fire, at least 0.997 when it does
  (`results_pod2_v2_final/response_probe_full/phase1_intervention_halt_samples.json`).
- **Surface features alone separate the classes.** Layer-0 (embedding) AUROC is already 0.90–0.95
  on every model (`layer_sweep` in the `*response_probe_meta.json` files).
- **It doesn't generalise.** On an unseen prefill ("Sure, here is"), the Mistral probe fires on
  0/20 prompts (`results_pod3_final/adaptive_attack/mistral-7b/adaptive_results.json`).

**Affected:** the reported AUROC (0.97–1.00), halt rates, "0/40 on all seven models", and 0%
benign FPR. These are training-set results that reflect template identity, not harmful content.

## 4. The Llama Guard 3 baseline never ran

`scripts/gpu_llamaguard_baseline.py:75-76` catches every exception and returns "not flagged". All
240 stored outputs are `"error: "`
(`results_pod3_final/llamaguard_baseline/{llama31-8b,mistral-7b}_samples.json`).

**Affected:** the "Llama Guard 3 flags 0/40" comparison is invalid.

## 5. The LLM-judge validation mis-parses verdicts

`scripts/gpu_llm_judge.py:77` uses `"HARMFUL" in result.upper()`. A verdict such as
"SAFE … not harmful" therefore counts as harmful. The judged inputs were also 120-character
previews.

**Affected:** the reported judge-agreement figures.

## 6. The utility comparisons don't cover all paradigms

- **MMLU:** run only on Mistral-7B, and only for no defense, static, and gated steering. CAST and
  AlphaSteer were never evaluated on MMLU. The figure hardcodes `[51.6] * 5`
  (`scripts/gpu_paper_phases2345.py:421`).
- **Hedging:** the evaluation re-applies only static and gated steering (hedging loop in
  `scripts/gpu_paper_final.py`). The "AlphaSteer" and "CAST" hedging rates were measured with no
  defense active.

**Affected:** the claim that MMLU is identical across all five paradigms, and the hedging rates for
AlphaSteer and CAST.

## 7. The geometric analyses don't measure what they claim

- **Probe vs. refusal-vector cosines:** they use logistic-regression weights in StandardScaler
  space, compared against the refusal vector at the *steering* layer rather than the probe layer
  (`scripts/gpu_eval_scaleup.py:265-293`).
- **Cross-model cosines:** they compare weight vectors index-by-index across models whose residual
  streams share no basis. No code in the repository generates
  `results_a100_runpod_final/new_models/cross_family_probe_transfer.json`, and that file contains
  no Gemma-4 entry.

**Affected:** the "orthogonal subspace" and "model-specific detection direction" claims.

## 8. Reporting errors

- **Template-robustness table:** the Llama-3.1-8B, Qwen-2.5-7B and Gemma-3-12B rows labelled
  "augmented linear" report the MLP results (`results_pod2_v2_final/mlp_probe/*/mlp_results.json`).
- **"Prompt-time" framing doesn't match the code:** every steering hook modifies `hidden[:, -1, :]`
  on each forward pass (`src/steering/apply_steering.py:46`). The defenses described as
  "prompt-time" therefore also act on every generated token.

## Doing it correctly

- **Prefill:** use `apply_chat_template(..., continue_final_message=True)` (or strip the closing
  token), and inspect the rendered string for every model.
- **Judging:** judge only the model's continuation, with the prefill excluded, using a validated
  classifier (e.g. the HarmBench classifier or StrongREJECT) on full responses. Unit-test the
  judge on adversarial cases such as "prefill + refusal".
- **Probes:** hold out both behaviours and templates. Include benign requests carrying a prefill as
  negatives. Report trivial baselines (layer-0 embeddings, bag-of-words).
- **Baselines:** fail loudly. Never let an exception default to "safe".
