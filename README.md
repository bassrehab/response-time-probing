# response-time-probing (archived)

> **Status: archived. The empirical results in arXiv:2606.29441 are not supported.**
> Errors found in the evaluation code after publication affect the paper's experiments:
> - prefilling attacks closed the assistant turn;
> - a keyword judge scored prefilled refusals as harmful;
> - the response-time probe was evaluated on its own training prompts;
> - the Llama Guard baseline never ran.
>
> See **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)**. arXiv v2 (November 2026) carries a correction notice pointing here.

This repository holds the code and raw outputs behind *Closing the Activation-Cone Blind Spot: Response-Time Probing and Unified Defense* (Subhadip Mitra, arXiv:2606.29441, June 2026). It is kept public for transparency and is not maintained.

**Content warning:** `results/` and `results_*/` contain model outputs to harmful requests (some with step-by-step harmful content) and adversarial GCG suffixes.

## What's here

| Path | Contents |
|---|---|
| `src/` | Model loading, activation hooks, linear probes, steering (static, probe-gated, CAST-style, AlphaSteer wrapper), attacks, evaluation |
| `scripts/` | Experiment scripts. `02`–`11` are the original pipeline. The `gpu_*` and `p*_` scripts produced the paper's tables, and the remaining scripts cover post-submission retraining, holdouts and judge re-scoring |
| `results/`, `results_*/` | Raw outputs from the GPU runs. Large logs and probe weights are stored with Git LFS |
| `arxiv_submission/` | LaTeX source as posted to arXiv (v1, tag `arxiv-v1`) |
| `paper/` | Working LaTeX source (NeurIPS 2026 submission and revisions) |
| `paper_workshop/` | Workshop version, *High AUROC, Wrong Feature* (not accepted) |
| `closing-the-activation-cone-blind-spot-v2.pdf` | The paper, with the correction notice |
| `KNOWN_ISSUES.md` | The audit: what is wrong, where in the code, and which claims it affects |

## What may still be useful

These components may be reusable, with the caveats in `KNOWN_ISSUES.md`:
- The activation hooks and steering utilities (`src/models/hooks.py`, `src/steering/`).
- The AlphaSteer wrapper (`src/baselines/alphasteer/`).
- The GCG suffixes in `results_*/gcg_suffixes/`.

**Do not reuse** `src/eval/judge.py`, or the prefilled-prompt construction in `src/eval/safety_eval.py`.

## Setup

```bash
pip install -e ".[dev]"
pytest tests/
```

AlphaSteer requires `git clone https://github.com/AlphaLab-USTC/AlphaSteer external/AlphaSteer`. The GPU experiments assumed an A100 80GB.

## License

Code is released under the MIT License (see `LICENSE`). Third-party datasets and baselines (AdvBench, AlphaSteer, the intent-laundering set) keep their own licenses.
