#!/usr/bin/env python3
"""TPC vs linear probe: OOD FPR comparison on Mistral-7B.

The in-distribution AUROC comparison was uninformative (both 1.000).
Instead, compare OOD false positive rate on 100 diverse open-ended
benign queries. Does TPC achieve lower FPR than linear?

Usage:
    python scripts/run_tpc_ood_comparison.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.eval.openended_eval import get_all_queries
from src.models.configs import get_config
from src.models.hooks import ActivationHook
from src.models.loader import get_device, load_model
from src.probes.train_tpc import train_tpc_at_layer
from src.probes.train_probe import train_probe_at_layer
from src.utils.paths import contrast_pairs_dir, probe_results_dir, RESULTS_DIR
from src.utils.seeding import seed_everything


def extract_single_layer(model, tokenizer, config, prompts, layer, device):
    hook = ActivationHook(model, layer)
    acts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=config.max_seq_len).to(device)
        with torch.no_grad():
            model(**inputs)
        acts.append(hook.get_last().cpu().float().numpy())
        hook.clear()
    hook.remove()
    return np.concatenate(acts, axis=0)


def main():
    seed_everything(42)
    device = get_device()
    model_name = "mistral-7b"

    # Load best probe layer
    probe_dir = probe_results_dir(model_name, "jailbreak")
    with open(probe_dir / "best_layer.json") as f:
        best_layer = json.load(f)["layer"]

    # Load diverse contrast pairs
    with open(contrast_pairs_dir() / "jailbreak_pairs.json") as f:
        pairs = json.load(f)
    train_prompts = [s["prompt"] for s in pairs]
    train_labels = np.array([s["label"] for s in pairs])

    # OOD test queries
    ood_queries = [q["query"] for q in get_all_queries()]

    print(f"Model: {model_name}, probe layer: {best_layer}")
    print(f"Training: {len(train_prompts)} prompts ({train_labels.sum()} pos, {(1-train_labels).sum()} neg)")
    print(f"OOD test: {len(ood_queries)} benign queries")

    # Load model
    print(f"\nLoading {model_name}...")
    model, tokenizer, config = load_model(model_name, device=device)

    # Extract training activations
    print(f"Extracting training activations at layer {best_layer}...")
    train_acts = extract_single_layer(model, tokenizer, config, train_prompts, best_layer, device)

    # Extract OOD activations
    print(f"Extracting OOD activations ({len(ood_queries)} queries)...")
    ood_acts = extract_single_layer(model, tokenizer, config, ood_queries, best_layer, device)

    del model
    if device == "mps":
        torch.mps.empty_cache()

    # Train linear probe
    print(f"\nTraining linear probe...")
    linear_probe, linear_scaler, auroc, acc, fpr, fold_aurocs = train_probe_at_layer(
        train_acts, train_labels, n_folds=5
    )
    print(f"  Linear: AUROC={auroc:.4f}, in-dist FPR={fpr:.4f}")

    # Compute OOD FPR for linear
    ood_scaled = linear_scaler.transform(ood_acts)
    linear_ood_scores = linear_probe.predict_proba(ood_scaled)[:, 1]

    # Train TPC probes at various ranks
    print(f"\nTraining TPC probes...")
    tpc_results = {}
    for rank in [16, 32, 64, 128]:
        tpc_probe, tpc_scaler, U, tpc_result = train_tpc_at_layer(
            train_acts, train_labels, degree=2, rank=rank, n_folds=5
        )
        print(f"  TPC r={rank}: AUROC={tpc_result.auroc:.4f}, in-dist FPR={tpc_result.fpr:.4f}")

        # Compute OOD features and FPR
        from src.probes.train_tpc import _polynomial_features_cp
        ood_features = _polynomial_features_cp(ood_acts, rank, seed=42)
        ood_tpc_scaled = tpc_scaler.transform(ood_features)
        tpc_ood_scores = tpc_probe.predict_proba(ood_tpc_scaled)[:, 1]

        tpc_results[rank] = {
            "auroc": tpc_result.auroc,
            "id_fpr": tpc_result.fpr,
            "ood_scores": tpc_ood_scores.tolist(),
        }

    # Compare OOD FPR at various thresholds
    print(f"\n{'='*70}")
    print(f"OOD FPR COMPARISON (100 benign open-ended queries)")
    print(f"{'='*70}")
    for threshold in [0.3, 0.5, 0.7]:
        linear_fpr = (linear_ood_scores >= threshold).mean()
        print(f"\n  Threshold = {threshold}")
        print(f"    Linear:     FPR = {linear_fpr:.1%} (mean score: {linear_ood_scores.mean():.3f})")
        for rank in [16, 32, 64, 128]:
            scores = np.array(tpc_results[rank]["ood_scores"])
            tpc_fpr = (scores >= threshold).mean()
            print(f"    TPC r={rank:3d}: FPR = {tpc_fpr:.1%} (mean score: {scores.mean():.3f})")

    # Save
    output_dir = RESULTS_DIR / "tpc_ood_comparison" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "linear": {
            "auroc": auroc,
            "id_fpr": fpr,
            "ood_scores_mean": float(linear_ood_scores.mean()),
            "ood_scores_std": float(linear_ood_scores.std()),
            "ood_fpr_t0.3": float((linear_ood_scores >= 0.3).mean()),
            "ood_fpr_t0.5": float((linear_ood_scores >= 0.5).mean()),
        },
    }
    for rank in [16, 32, 64, 128]:
        scores = np.array(tpc_results[rank]["ood_scores"])
        results[f"tpc_r{rank}"] = {
            "auroc": tpc_results[rank]["auroc"],
            "id_fpr": tpc_results[rank]["id_fpr"],
            "ood_scores_mean": float(scores.mean()),
            "ood_scores_std": float(scores.std()),
            "ood_fpr_t0.3": float((scores >= 0.3).mean()),
            "ood_fpr_t0.5": float((scores >= 0.5).mean()),
        }

    with open(output_dir / "ood_fpr_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
