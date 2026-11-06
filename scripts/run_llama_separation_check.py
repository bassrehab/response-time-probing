#!/usr/bin/env python3
"""Llama-3.1-8B: probe + vector separation check.

4th data point for the vector separation → Pareto advantage claim.
Quick check: train probe, extract vector, report separation.
If separation > 0.4, run full evaluation. If < 0.4, just report.

Usage:
    python scripts/run_llama_separation_check.py
"""

import json
import time

import numpy as np
import torch

from src.models.configs import get_config
from src.models.hooks import ActivationHook, MultiLayerHook
from src.models.loader import get_device, load_model
from src.probes.train_probe import layer_sweep, save_probe_results
from src.steering.extract_vectors import compute_caa_vector, save_steering_vector
from src.utils.paths import contrast_pairs_dir, probe_results_dir, vector_results_dir, RESULTS_DIR
from src.utils.seeding import seed_everything


def main():
    seed_everything(42)
    device = get_device()
    model_name = "llama31-8b"

    # Load contrast pairs
    with open(contrast_pairs_dir() / "jailbreak_pairs.json") as f:
        pairs = json.load(f)
    prompts = [s["prompt"] for s in pairs]
    labels = np.array([s["label"] for s in pairs])

    config = get_config(model_name)

    print(f"{'='*60}")
    print(f"LLAMA-3.1-8B SEPARATION CHECK")
    print(f"Layers: {config.num_layers}, Hidden dim: {config.hidden_dim}")
    print(f"{'='*60}")

    print(f"\nLoading {model_name}...")
    t0 = time.time()
    model, tokenizer, config = load_model(model_name, device=device)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Train probe with chunked extraction
    print(f"\nStep 1: Train jailbreak probe (full layer sweep)")
    layer_indices = list(range(config.num_layers))
    all_acts = {}
    chunk_size = 8
    for cs in range(0, len(layer_indices), chunk_size):
        chunk = layer_indices[cs:cs + chunk_size]
        print(f"  Extracting layers {chunk[0]}-{chunk[-1]}...")
        hook = MultiLayerHook(model, chunk)
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=config.max_seq_len).to(device)
            with torch.no_grad():
                model(**inputs)
            last = hook.get_all_last()
            for l, act in last.items():
                all_acts.setdefault(l, []).append(act.cpu().float().numpy())
            hook.clear()
        hook.remove()
    all_activations = {l: np.concatenate(acts, axis=0) for l, acts in all_acts.items()}

    results, best_layer, best_probe, best_scaler = layer_sweep(all_activations, labels, n_folds=5)
    probe_dir = probe_results_dir(model_name, "jailbreak")
    save_probe_results(probe_dir, model_name, "jailbreak", results, best_layer, best_probe, best_scaler)

    best_result = results[best_layer]
    print(f"\n  Best probe: layer {best_layer}, AUROC={best_result.auroc:.4f}, FPR={best_result.fpr:.4f}")

    # Extract steering vector
    print(f"\nStep 2: Extract refusal steering vector at layer 14")
    steering_layer = config.optimal_steering_layer
    pos_prompts = [s["prompt"] for s in pairs if s["label"] == 1]
    neg_prompts = [s["prompt"] for s in pairs if s["label"] == 0]

    hook = ActivationHook(model, steering_layer)
    pos_acts = []
    for p in pos_prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=config.max_seq_len).to(device)
        with torch.no_grad():
            model(**inputs)
        pos_acts.append(hook.get_last().cpu().float().numpy())
        hook.clear()
    neg_acts = []
    for p in neg_prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=config.max_seq_len).to(device)
        with torch.no_grad():
            model(**inputs)
        neg_acts.append(hook.get_last().cpu().float().numpy())
        hook.clear()
    hook.remove()

    pos_acts = np.concatenate(pos_acts, axis=0)
    neg_acts = np.concatenate(neg_acts, axis=0)

    vector, vec_meta = compute_caa_vector(pos_acts, neg_acts)
    vec_dir = vector_results_dir(model_name, "refusal")
    save_steering_vector(vec_dir, vector, vec_meta, steering_layer)

    separation = vec_meta["separation"]

    del model
    if device == "mps":
        torch.mps.empty_cache()

    # Report
    print(f"\n{'='*60}")
    print(f"LLAMA-3.1-8B RESULTS")
    print(f"{'='*60}")
    print(f"  Probe: layer {best_layer}, AUROC {best_result.auroc:.4f}, FPR {best_result.fpr:.4f}")
    print(f"  Vector: layer {steering_layer}, separation {separation:.4f}")
    print()

    # Cross-model separation comparison
    print(f"  Cross-model separation comparison:")
    print(f"    Mistral-7B:    0.533 → Pareto advantage YES")
    print(f"    Gemma-2-9B:    0.336 → Pareto advantage NO")
    print(f"    Qwen-2.5-7B:   0.276 → Pareto advantage NO")
    print(f"    Llama-3.1-8B:  {separation:.3f} → {'> 0.4 threshold' if separation > 0.4 else '< 0.4 threshold'}")
    print()

    if separation > 0.4:
        print(f"  RESULT: Separation > 0.4 — run full evaluation")
        print(f"  Use: python scripts/run_fixed_probe_validation.py --model llama31-8b")
    else:
        print(f"  RESULT: Separation < 0.4 — consistent with pattern")
        print(f"  Report separation only, no full evaluation needed")

    # Save
    output_dir = RESULTS_DIR / "llama_separation" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "model": model_name,
            "probe_layer": best_layer,
            "probe_auroc": best_result.auroc,
            "probe_fpr": best_result.fpr,
            "vector_layer": steering_layer,
            "separation": separation,
            "above_threshold": separation > 0.4,
        }, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
