#!/usr/bin/env python3
"""Public-facing wrapper around `scripts/gpu_retrain_all_7.py`.

Regenerates response-probe weights from scratch for all seven models, using
the same 90-sample phase-1 training set that produced the paper's reported
halt rates. Recommended for reviewers who want to reproduce paper Table~2
numbers under their own PyTorch / Transformers / HF-snapshot versions: the
Mistral paper-shipped pkl drifts on later library stacks (see §5.3 in the
paper), while fresh retraining reproduces the paper's 82% halt rate within
$\pm 2$pp across all seven models.

Requirements:
  - NVIDIA GPU with $\geq$62 GB VRAM (Gemma-4-31B is the largest; A100 80GB
    recommended; H100 80GB also fine).
  - HuggingFace token in $\sim$/.cache/huggingface/token for gated weights
    (Llama-3.1-8B-Instruct, Gemma family).
  - `pip install -e .` from repo root.

Outputs:
  results/retrain_all_7/{model}/{probe_fresh.pkl, retrain_report.json, ...}
  results/retrain_all_7/summary.json  (cross-model discriminator table)

Invocation:
  python scripts/retrain_response_probes.py              # all 7 models
  python scripts/retrain_response_probes.py --model mistral-7b

Budget (on A100 80GB): ~30--45 min per model, ~3 h total for all 7.
"""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    target = root / "scripts" / "gpu_retrain_all_7.py"
    if not target.exists():
        print(f"ERROR: {target} not found. Expected repo layout: scripts/gpu_retrain_all_7.py")
        sys.exit(1)
    # Pass through any CLI args
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    print(f"[retrain_response_probes.py] Invoking: {' '.join(cmd)}", flush=True)
    print("[retrain_response_probes.py] This script is a public-facing wrapper;")
    print("[retrain_response_probes.py] the actual retraining logic is in gpu_retrain_all_7.py.")
    print("[retrain_response_probes.py] See the paper's Appendix on probe reproducibility for the")
    print("[retrain_response_probes.py] comparison table of paper-shipped vs fresh-retrained halt rates.")
    subprocess.check_call(cmd, cwd=root)


if __name__ == "__main__":
    main()
