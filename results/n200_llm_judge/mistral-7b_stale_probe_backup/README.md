# Mistral n=200 dual-probe rerun — stale-probe artefact

Produced 2026-04-19 on Pod B via `scripts/gpu_n200_dual_probe_rerun.py` using the
paper's saved `results/response_probe_full/mistral_response_probe.pkl`.

Under transformers 5.5.4 + torch 2.4.1 + CUDA 12.4, this run produced DSR=0.442,
prefilling ASR=50%, GCG ASR=67.5% — substantially worse than paper Table 1
Mistral dual-probe n=40 numbers (DSR=0.706, prefilling=0%).

Subsequent retrain-from-scratch test (same environment) demonstrated the paper's
saved probe pkl drifted: a fresh Mistral probe trained on the paper's original
90-sample training set reproduces paper phase4's halt rate at 32/40 = 80% (vs
paper's 82%). Therefore **the numbers in this directory are artifacts of the
stale pkl under current environment**, not the method's actual performance.

Retained for the appendix reproducibility table comparing (paper pkl under
current env) × (fresh-retrained pkl). See `_dev/hardening/REVISION_STATUS.md`
entry E8 for the full rule-10 triage.
