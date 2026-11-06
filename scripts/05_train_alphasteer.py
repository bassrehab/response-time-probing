#!/usr/bin/env python3
"""Train AlphaSteer Δ matrices for all models.

Computes:
  1. Null-space projection P̂ from benign activations
  2. Refusal direction vectors from contrast pairs
  3. Δ̃* matrix via regularized least squares

Requires CUDA (A100) — the matrix computation involves large SVDs
and the activation extraction is faster on GPU.

Usage:
    python scripts/05_train_alphasteer.py --model mistral-7b
    python scripts/05_train_alphasteer.py --models all
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.attacks.datasets.prompts import ADVBENCH_BEHAVIORS, BENIGN_QUERIES, BENIGN_OPENENDED
from src.baselines.alphasteer.wrapper import (
    compute_null_space_projection,
    compute_delta_matrix,
    save_alphasteer_matrices,
)
from src.models.configs import MODEL_CONFIGS, get_config
from src.models.hooks import ActivationHook
from src.models.loader import get_device, load_model
from src.utils.paths import RESULTS_DIR
from src.utils.seeding import seed_everything


def extract_activations(model, tokenizer, config, prompts, layer, device):
    """Extract last-token activations at a single layer."""
    hook = ActivationHook(model, layer)
    acts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True,
            max_length=config.max_seq_len,
        ).to(device)
        with torch.no_grad():
            model(**inputs)
        acts.append(hook.get_last().cpu().float().numpy())
        hook.clear()
    hook.remove()
    return np.concatenate(acts, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Train AlphaSteer Δ matrices")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = args.device or get_device()

    if args.models == "all":
        model_names = list(MODEL_CONFIGS.keys())
    elif args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    elif args.model:
        model_names = [args.model]
    else:
        parser.error("Specify --model or --models")

    # Benign prompts for null-space computation (diverse set)
    benign_prompts = list(BENIGN_QUERIES[:40]) + list(BENIGN_OPENENDED[:60])
    # Malicious prompts
    malicious_prompts = list(ADVBENCH_BEHAVIORS[:40])

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training AlphaSteer for {model_name}")
        print(f"{'='*60}")

        config = get_config(model_name)
        model, tokenizer, config = load_model(model_name, device=device)

        # Steer at the model's optimal steering layer
        steering_layer = config.optimal_steering_layer
        output_dir = RESULTS_DIR / "alphasteer" / model_name

        print(f"  Steering layer: {steering_layer}")
        print(f"  Benign prompts: {len(benign_prompts)}")
        print(f"  Malicious prompts: {len(malicious_prompts)}")

        # Extract benign activations for null-space
        print(f"  Extracting benign activations...")
        t0 = time.time()
        benign_acts = extract_activations(
            model, tokenizer, config, benign_prompts, steering_layer, device
        )
        print(f"  Done in {time.time()-t0:.0f}s, shape: {benign_acts.shape}")

        # Extract malicious activations
        print(f"  Extracting malicious activations...")
        t0 = time.time()
        malicious_acts = extract_activations(
            model, tokenizer, config, malicious_prompts, steering_layer, device
        )
        print(f"  Done in {time.time()-t0:.0f}s, shape: {malicious_acts.shape}")

        # Compute refusal direction: mean(benign) - mean(malicious)
        # (We want to STEER TOWARD refusal, so the direction goes from malicious to benign)
        refusal_dir = benign_acts.mean(axis=0) - malicious_acts.mean(axis=0)
        refusal_dir = refusal_dir / np.linalg.norm(refusal_dir)
        # Broadcast to match malicious_acts shape
        refusal_directions = np.tile(refusal_dir, (len(malicious_acts), 1))

        # Compute null-space projection
        print(f"  Computing null-space projection P̂...")
        t0 = time.time()
        P_hat = compute_null_space_projection(benign_acts, rank_threshold=0.01)
        null_rank = np.linalg.matrix_rank(P_hat, tol=1e-6)
        print(f"  Done in {time.time()-t0:.0f}s, null-space rank: {null_rank}/{config.hidden_dim}")

        # Compute Δ matrix
        print(f"  Computing Δ̃* matrix...")
        t0 = time.time()
        Delta = compute_delta_matrix(malicious_acts, refusal_directions, P_hat, alpha=0.01)
        print(f"  Done in {time.time()-t0:.0f}s, shape: {Delta.shape}")

        # Save
        save_alphasteer_matrices(output_dir, Delta, P_hat, steering_layer)
        print(f"  Saved to {output_dir}")

        # Verify: check that P̂ · h_benign ≈ 0
        benign_proj = (P_hat @ benign_acts.T).T
        benign_proj_norm = np.linalg.norm(benign_proj, axis=1).mean()
        malicious_proj = (P_hat @ malicious_acts.T).T
        malicious_proj_norm = np.linalg.norm(malicious_proj, axis=1).mean()
        print(f"  Verification:")
        print(f"    Mean ||P̂ · h_benign||: {benign_proj_norm:.4f} (should be small)")
        print(f"    Mean ||P̂ · h_malicious||: {malicious_proj_norm:.4f} (should be larger)")
        print(f"    Ratio: {malicious_proj_norm / (benign_proj_norm + 1e-8):.2f}x")

        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
