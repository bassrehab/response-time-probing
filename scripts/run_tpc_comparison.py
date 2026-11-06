#!/usr/bin/env python3
"""Compare linear probe vs TPC (Truncated Polynomial Classifier).

Preempts reviewer objection: "Why linear probes when TPCs exist?"

Trains both at the best jailbreak probe layer for Mistral-7B and compares
AUROC. If TPC improvement is <2%, we justify "linear probes suffice."

Usage:
    python scripts/run_tpc_comparison.py --model mistral-7b
"""

import argparse
import json
import time

import numpy as np
import torch

from src.models.configs import MODEL_CONFIGS, get_config
from src.models.hooks import MultiLayerHook
from src.models.loader import get_device, load_model
from src.probes.train_tpc import compare_linear_vs_tpc
from src.utils.paths import contrast_pairs_dir, probe_results_dir, RESULTS_DIR
from src.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Compare linear probe vs TPC")
    parser.add_argument("--model", type=str, default="mistral-7b")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = args.device or get_device()
    model_name = args.model

    # Load best probe layer
    probe_dir = probe_results_dir(model_name, "jailbreak")
    with open(probe_dir / "best_layer.json") as f:
        best_layer = json.load(f)["layer"]

    print(f"Comparing linear vs TPC at layer {best_layer} for {model_name}")

    # Load contrast pairs
    pairs_path = contrast_pairs_dir() / "jailbreak_pairs.json"
    with open(pairs_path) as f:
        pairs = json.load(f)

    prompts = [s["prompt"] for s in pairs]
    labels = np.array([s["label"] for s in pairs])

    # Load model and extract activations at best layer
    print(f"\nLoading {model_name}...")
    model, tokenizer, config = load_model(model_name, device=device)

    print(f"Extracting activations at layer {best_layer}...")
    hook = MultiLayerHook(model, [best_layer])
    activations_list = []

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        act = hook.get_all_last()[best_layer].cpu().float().numpy()
        activations_list.append(act)
        hook.clear()

    hook.remove()
    del model
    if device == "mps":
        torch.mps.empty_cache()

    activations = np.concatenate(activations_list, axis=0)
    print(f"Activations shape: {activations.shape}")

    # Compare
    print(f"\nRunning comparison (5-fold CV)...")
    results = compare_linear_vs_tpc(
        activations, labels,
        ranks=[16, 32, 64, 128],
        n_folds=5,
        seed=args.seed,
    )

    # Save
    output_dir = RESULTS_DIR / "tpc_comparison" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Report
    summary = results["summary"]
    print(f"\n{'='*60}")
    print(f"RESULT: {summary['recommendation']}")
    print(f"  Best TPC: {summary['best_tpc']}")
    print(f"  Improvement: {summary['improvement_pp']:+.2f} percentage points AUROC")
    if summary["recommendation"] == "linear_sufficient":
        print(f"  → Linear probes suffice. Report in paper: \"TPC provides")
        print(f"    marginal improvement, justifying our use of linear probes")
        print(f"    for simplicity and interpretability.\"")
    elif summary["recommendation"] == "upgrade_to_tpc":
        print(f"  → Significant TPC improvement. Upgrade probe and re-run experiments.")
    print(f"{'='*60}")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
