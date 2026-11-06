#!/usr/bin/env python3
"""Train probes across models and probe types with full layer sweep.

Ties together: model loading → activation extraction → probe training → saving results.

Usage:
    # Single model, single probe type (smoke test)
    python scripts/03_train_probes.py --model mistral-7b --probe jailbreak

    # All models, all probes
    python scripts/03_train_probes.py --models all --probes all
"""

import argparse
import json
import time

import numpy as np
import torch

from src.models.configs import MODEL_CONFIGS, get_config
from src.models.hooks import MultiLayerHook
from src.models.loader import get_device, load_model
from src.probes.train_probe import layer_sweep, save_probe_results
from src.utils.paths import contrast_pairs_dir, probe_results_dir
from src.utils.seeding import seed_everything


def extract_activations(
    model,
    tokenizer,
    config,
    prompts: list[str],
    layer_indices: list[int],
    device: str,
) -> dict[int, np.ndarray]:
    """Extract activations at the last prompt token for each prompt.

    Uses the model's chat template to format each prompt as a user message
    with generation prompt appended. Captures the last-token activation at
    each hooked layer — this is the model's internal state when deciding
    how to respond, before any generation.

    Returns:
        {layer_idx: np.ndarray of shape (n_samples, hidden_dim)}
    """
    hook = MultiLayerHook(model, layer_indices)
    all_acts: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_len,
        ).to(device)

        with torch.no_grad():
            model(**inputs)

        last_acts = hook.get_all_last()
        for layer_idx, act in last_acts.items():
            all_acts[layer_idx].append(act.cpu().float().numpy())

        hook.clear()

        if (i + 1) % 10 == 0:
            print(f"    Extracted {i + 1}/{len(prompts)} samples")

    hook.remove()

    return {l: np.concatenate(acts, axis=0) for l, acts in all_acts.items()}


def train_probes_for_model(
    model_name: str,
    probe_type: str,
    device: str | None = None,
) -> dict:
    """Full pipeline: load model, extract activations, train probes, save results.

    Returns:
        Best layer info dict.
    """
    device = device or get_device()
    config = get_config(model_name)

    # Load contrast pairs (model-specific, falling back to generic)
    pairs_path = contrast_pairs_dir() / f"{probe_type}_pairs_{model_name}.json"
    if not pairs_path.exists():
        pairs_path = contrast_pairs_dir() / f"{probe_type}_pairs.json"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"No contrast pairs found. "
            f"Run: python scripts/02_generate_contrast_pairs.py --model {model_name} first."
        )
    with open(pairs_path) as f:
        pairs = json.load(f)

    print(f"\n{'='*60}")
    print(f"Training {probe_type} probes for {model_name}")
    print(f"  Device: {device}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Layers: {config.num_layers}")
    print(f"{'='*60}\n")

    # Build prompt and label lists from samples
    prompts = [s["prompt"] for s in pairs]
    labels = np.array([s["label"] for s in pairs])
    print(f"  Total samples: {len(labels)} ({labels.sum()} harmful, {(1-labels).sum()} benign)")

    # Load model
    print(f"\n  Loading model {model_name}...")
    t0 = time.time()
    model, tokenizer, config = load_model(model_name, device=device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Extract activations at all layers
    layer_indices = list(range(config.num_layers))
    print(f"\n  Extracting activations at {len(layer_indices)} layers...")
    t0 = time.time()
    all_activations = extract_activations(
        model, tokenizer, config, prompts, layer_indices, device
    )
    print(f"  Extraction done in {time.time() - t0:.1f}s")

    # Free model memory before probe training
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # Layer sweep
    print(f"\n  Running layer sweep with 5-fold CV...")
    results, best_layer, best_probe, best_scaler = layer_sweep(
        all_activations, labels, n_folds=5
    )

    # Save results
    output_dir = probe_results_dir(model_name, probe_type)
    save_probe_results(
        output_dir, model_name, probe_type, results, best_layer, best_probe, best_scaler
    )

    best_result = results[best_layer]
    print(f"\n  Best layer: {best_layer}")
    print(f"  Best AUROC: {best_result.auroc:.4f} (±{best_result.auroc_std:.4f})")
    print(f"  Accuracy:   {best_result.accuracy:.4f}")
    print(f"  FPR:        {best_result.fpr:.4f}")

    return {
        "model": model_name,
        "probe_type": probe_type,
        "best_layer": best_layer,
        "best_auroc": best_result.auroc,
        "best_accuracy": best_result.accuracy,
        "best_fpr": best_result.fpr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes with full layer sweep")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. mistral-7b). Use --models for multiple.")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--probe", type=str, default=None,
                        help="Probe type (e.g. jailbreak). Use --probes for multiple.")
    parser.add_argument("--probes", type=str, default=None,
                        help="Comma-separated probe types or 'all'")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Resolve models
    if args.models == "all":
        model_names = list(MODEL_CONFIGS.keys())
    elif args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    elif args.model:
        model_names = [args.model]
    else:
        parser.error("Specify --model or --models")

    # Resolve probe types
    all_probe_types = ["jailbreak", "sycophancy"]
    if args.probes == "all":
        probe_types = all_probe_types
    elif args.probes:
        probe_types = [p.strip() for p in args.probes.split(",")]
    elif args.probe:
        probe_types = [args.probe]
    else:
        parser.error("Specify --probe or --probes")

    # Train
    all_results = []
    for model_name in model_names:
        for probe_type in probe_types:
            result = train_probes_for_model(model_name, probe_type, device=args.device)
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['model']:15s} | {r['probe_type']:12s} | "
              f"Layer {r['best_layer']:3d} | AUROC {r['best_auroc']:.4f}")


if __name__ == "__main__":
    main()
