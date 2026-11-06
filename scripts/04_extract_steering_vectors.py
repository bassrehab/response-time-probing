#!/usr/bin/env python3
"""Extract CAA steering vectors for all models and behaviors.

For each (model, behavior):
  1. Load contrast pairs (harmful vs benign prompts)
  2. Run through model, capture activations at the steering layer
  3. Compute CAA vector: mean(harmful) - mean(benign), normalized
  4. Also extract at the probe's best layer for comparison
  5. Save vector + metadata

Usage:
    python scripts/04_extract_steering_vectors.py --model mistral-7b --behavior refusal
    python scripts/04_extract_steering_vectors.py --models all --behaviors all
"""

import argparse
import json
import time

import numpy as np
import torch

from src.models.configs import MODEL_CONFIGS, get_config
from src.models.hooks import ActivationHook
from src.models.loader import get_device, load_model
from src.steering.extract_vectors import compute_caa_vector, save_steering_vector
from src.utils.paths import contrast_pairs_dir, probe_results_dir, vector_results_dir
from src.utils.seeding import seed_everything


def extract_activations_at_layer(
    model,
    tokenizer,
    config,
    prompts: list[str],
    layer: int,
    device: str,
) -> np.ndarray:
    """Extract last-token activations at a single layer for a list of prompts.

    Returns:
        np.ndarray of shape (n_samples, hidden_dim)
    """
    hook = ActivationHook(model, layer)
    activations = []

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

        act = hook.get_last().cpu().float().numpy()
        activations.append(act)
        hook.clear()

        if (i + 1) % 20 == 0:
            print(f"    Extracted {i + 1}/{len(prompts)} samples")

    hook.remove()
    return np.concatenate(activations, axis=0)


def extract_vectors_for_model(
    model_name: str,
    behavior: str,
    device: str | None = None,
) -> dict:
    """Extract steering vectors at the configured steering layer (+ probe layer).

    Returns:
        Metadata dict for the primary steering vector.
    """
    device = device or get_device()
    config = get_config(model_name)

    # Map behaviors to their contrast pair files and probe types
    behavior_to_probe = {
        "refusal": "jailbreak",
        "anti_sycophancy": "sycophancy",
    }
    probe_type = behavior_to_probe.get(behavior, "jailbreak")

    # Load contrast pairs — try behavior-specific file first, then probe type
    pairs_path = contrast_pairs_dir() / f"{probe_type}_pairs.json"
    if not pairs_path.exists():
        pairs_path = contrast_pairs_dir() / "jailbreak_pairs.json"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"No contrast pairs found. Run scripts/02 first."
        )
    with open(pairs_path) as f:
        samples = json.load(f)

    positive_prompts = [s["prompt"] for s in samples if s["label"] == 1]
    negative_prompts = [s["prompt"] for s in samples if s["label"] == 0]

    print(f"\n{'='*60}")
    print(f"Extracting {behavior} steering vectors for {model_name}")
    print(f"  Device: {device}")
    print(f"  Positive prompts: {len(positive_prompts)}")
    print(f"  Negative prompts: {len(negative_prompts)}")
    print(f"  Source: {pairs_path.name}")
    print(f"{'='*60}")

    # Determine layers to extract at
    steering_layer = config.optimal_steering_layer
    layers_to_extract = [steering_layer]

    # Also extract at the probe's best layer if different
    probe_results_path = probe_results_dir(model_name, probe_type) / "best_layer.json"
    probe_layer = None
    if probe_results_path.exists():
        with open(probe_results_path) as f:
            probe_info = json.load(f)
        probe_layer = probe_info["layer"]
        if probe_layer != steering_layer:
            layers_to_extract.append(probe_layer)

    print(f"  Steering layer: {steering_layer}")
    if probe_layer is not None and probe_layer != steering_layer:
        print(f"  Probe best layer: {probe_layer} (also extracting for comparison)")

    # Load model
    print(f"\n  Loading model {model_name}...")
    t0 = time.time()
    model, tokenizer, config = load_model(model_name, device=device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    output_dir = vector_results_dir(model_name, behavior)
    primary_metadata = None

    for layer in layers_to_extract:
        print(f"\n  Extracting activations at layer {layer}...")
        t0 = time.time()

        positive_acts = extract_activations_at_layer(
            model, tokenizer, config, positive_prompts, layer, device
        )
        negative_acts = extract_activations_at_layer(
            model, tokenizer, config, negative_prompts, layer, device
        )

        print(f"  Extraction done in {time.time() - t0:.1f}s")
        print(f"  Positive activations shape: {positive_acts.shape}")
        print(f"  Negative activations shape: {negative_acts.shape}")

        # Compute CAA vector
        vector, metadata = compute_caa_vector(positive_acts, negative_acts)
        metadata["model"] = model_name
        metadata["behavior"] = behavior

        # Save
        save_steering_vector(output_dir, vector, metadata, layer)

        print(f"  Layer {layer} vector:")
        print(f"    Raw norm: {metadata['raw_norm']:.4f}")
        print(f"    Pos cosine: {metadata['pos_cosine_mean']:.4f} ± {metadata['pos_cosine_std']:.4f}")
        print(f"    Neg cosine: {metadata['neg_cosine_mean']:.4f} ± {metadata['neg_cosine_std']:.4f}")
        print(f"    Separation: {metadata['separation']:.4f}")

        if layer == steering_layer:
            primary_metadata = metadata

    # Free model memory
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return primary_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CAA steering vectors")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. mistral-7b)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--behavior", type=str, default=None,
                        help="Behavior (e.g. refusal)")
    parser.add_argument("--behaviors", type=str, default=None,
                        help="Comma-separated behaviors or 'all'")
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

    # Resolve behaviors
    all_behaviors = ["refusal", "anti_sycophancy"]
    if args.behaviors == "all":
        behaviors = all_behaviors
    elif args.behaviors:
        behaviors = [b.strip() for b in args.behaviors.split(",")]
    elif args.behavior:
        behaviors = [args.behavior]
    else:
        parser.error("Specify --behavior or --behaviors")

    # Extract
    all_metadata = []
    for model_name in model_names:
        for behavior in behaviors:
            meta = extract_vectors_for_model(model_name, behavior, device=args.device)
            all_metadata.append(meta)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for m in all_metadata:
        print(f"  {m['model']:15s} | {m['behavior']:10s} | "
              f"norm={m['raw_norm']:.2f} | separation={m['separation']:.4f}")


if __name__ == "__main__":
    main()
