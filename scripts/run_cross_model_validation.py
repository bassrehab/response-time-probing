#!/usr/bin/env python3
"""Cross-model validation: full pipeline for a single model.

Runs the complete sequence:
  1. Train jailbreak probe with full layer sweep
  2. Extract refusal steering vector at layer 14
  3. Safety eval: 7 defenses × 40 attack prompts (DSR)
  4. Open-ended utility: 7 defenses × 100 benign queries (hedging)
  5. Report Pareto table

Usage:
    python scripts/run_cross_model_validation.py --model gemma2-9b
    python scripts/run_cross_model_validation.py --model qwen25-7b
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.attacks.datasets.prompts import BENIGN_QUERIES
from src.attacks.registry import load_attack_set
from src.baselines.static_steering.infer import StaticSteering
from src.eval.openended_eval import run_openended_eval, save_openended_results
from src.eval.safety_eval import run_safety_eval
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.models.configs import get_config
from src.models.hooks import MultiLayerHook
from src.models.loader import get_device, load_model
from src.probes.train_probe import layer_sweep, save_probe_results
from src.steering.extract_vectors import compute_caa_vector, save_steering_vector
from src.utils.paths import (
    contrast_pairs_dir, probe_results_dir, vector_results_dir, RESULTS_DIR,
)
from src.utils.run_logger import capture_run_metadata, save_run_metadata
from src.utils.seeding import seed_everything


def extract_activations_for_probing(
    model, tokenizer, config, prompts, layer_indices, device,
):
    """Extract activations at all layers for probe training."""
    hook = MultiLayerHook(model, layer_indices)
    all_acts = {l: [] for l in layer_indices}

    for i, prompt in enumerate(prompts):
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

        last_acts = hook.get_all_last()
        for layer_idx, act in last_acts.items():
            all_acts[layer_idx].append(act.cpu().float().numpy())
        hook.clear()

        if (i + 1) % 20 == 0:
            print(f"    Extracted {i + 1}/{len(prompts)}")

    hook.remove()
    return {l: np.concatenate(acts, axis=0) for l, acts in all_acts.items()}


def extract_activations_single_layer(
    model, tokenizer, config, prompts, layer, device,
):
    """Extract activations at a single layer."""
    from src.models.hooks import ActivationHook
    hook = ActivationHook(model, layer)
    activations = []

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

        activations.append(hook.get_last().cpu().float().numpy())
        hook.clear()

    hook.remove()
    return np.concatenate(activations, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Cross-model validation pipeline")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = args.device or get_device()
    model_name = args.model

    output_dir = RESULTS_DIR / "cross_model" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = capture_run_metadata("cross_model_validation", __file__,
                                     args={"model": model_name})
    save_run_metadata(output_dir, metadata)

    model_config = get_config(model_name)

    # Load contrast pairs
    pairs_path = contrast_pairs_dir() / "jailbreak_pairs.json"
    with open(pairs_path) as f:
        pairs = json.load(f)
    prompts = [s["prompt"] for s in pairs]
    labels = np.array([s["label"] for s in pairs])

    print(f"\n{'#'*70}")
    print(f"# CROSS-MODEL VALIDATION: {model_name}")
    print(f"# Layers: {model_config.num_layers}, Hidden dim: {model_config.hidden_dim}")
    print(f"# Device: {device}")
    print(f"{'#'*70}")

    # ================================================================
    # STEP 1: Load model + train jailbreak probe with full layer sweep
    # ================================================================
    print(f"\n{'='*60}")
    print(f"STEP 1: Train jailbreak probe (full layer sweep)")
    print(f"{'='*60}")

    print(f"  Loading {model_name}...")
    t0 = time.time()
    model, tokenizer, model_config = load_model(model_name, device=device)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    layer_indices = list(range(model_config.num_layers))
    # Extract in chunks to avoid OOM on large models (Gemma 42 layers)
    chunk_size = 8
    all_activations = {}
    t0 = time.time()
    for chunk_start in range(0, len(layer_indices), chunk_size):
        chunk = layer_indices[chunk_start:chunk_start + chunk_size]
        print(f"  Extracting layers {chunk[0]}-{chunk[-1]}...")
        chunk_acts = extract_activations_for_probing(
            model, tokenizer, model_config, prompts, chunk, device
        )
        all_activations.update(chunk_acts)
    print(f"  Extraction done in {time.time()-t0:.1f}s ({len(all_activations)} layers)")

    print(f"  Running layer sweep with 5-fold CV...")
    results, best_layer, best_probe, best_scaler = layer_sweep(
        all_activations, labels, n_folds=5
    )

    probe_dir = probe_results_dir(model_name, "jailbreak")
    save_probe_results(
        probe_dir, model_name, "jailbreak", results, best_layer, best_probe, best_scaler
    )

    best_result = results[best_layer]
    print(f"\n  Best probe layer: {best_layer}")
    print(f"  AUROC: {best_result.auroc:.4f} (±{best_result.auroc_std:.4f})")
    print(f"  FPR: {best_result.fpr:.4f}")

    # ================================================================
    # STEP 2: Extract refusal steering vector
    # ================================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: Extract refusal steering vector at layer 14")
    print(f"{'='*60}")

    steering_layer = model_config.optimal_steering_layer
    positive_prompts = [s["prompt"] for s in pairs if s["label"] == 1]
    negative_prompts = [s["prompt"] for s in pairs if s["label"] == 0]

    print(f"  Extracting positive activations ({len(positive_prompts)} prompts)...")
    pos_acts = extract_activations_single_layer(
        model, tokenizer, model_config, positive_prompts, steering_layer, device
    )
    print(f"  Extracting negative activations ({len(negative_prompts)} prompts)...")
    neg_acts = extract_activations_single_layer(
        model, tokenizer, model_config, negative_prompts, steering_layer, device
    )

    vector, vec_metadata = compute_caa_vector(pos_acts, neg_acts)
    vec_dir = vector_results_dir(model_name, "refusal")
    save_steering_vector(vec_dir, vector, vec_metadata, steering_layer)

    print(f"  Steering vector at layer {steering_layer}:")
    print(f"    Norm: {vec_metadata['raw_norm']:.4f}")
    print(f"    Separation: {vec_metadata['separation']:.4f}")

    # ================================================================
    # STEP 3: Safety eval (7 defenses × 40 attacks)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"STEP 3: Safety evaluation")
    print(f"{'='*60}")

    vector_path = vec_dir / f"vector_layer{steering_layer}.pt"

    all_prompts = []
    for attack in ["prefilling", "intent_laundering"]:
        all_prompts.extend(load_attack_set(attack, n_attacks=20, include_benign=False))
    for i in range(min(20, len(BENIGN_QUERIES))):
        all_prompts.append({"prompt": BENIGN_QUERIES[i], "attack_type": "benign"})

    defenses = [
        {"name": "none", "type": "none"},
        {"name": "static_0.5", "type": "static", "strength": 0.5},
        {"name": "static_0.8", "type": "static", "strength": 0.8},
        {"name": "static_1.0", "type": "static", "strength": 1.0},
        {"name": "static_1.3", "type": "static", "strength": 1.3},
        {"name": "gated_bb_1.3_t0.3", "type": "gated", "controller": "bang_bang",
         "strength": 1.3, "threshold": 0.3},
        {"name": "gated_prop_1.0_t0.5", "type": "gated", "controller": "proportional",
         "strength": 1.0, "threshold": 0.5, "gain": 2.0},
    ]

    safety_results = {}
    for defense in defenses:
        name = defense["name"]
        print(f"\n  [{name}] Safety eval...")

        pgs = None
        static = None

        if defense["type"] == "static":
            static = StaticSteering(model, str(vector_path), steering_layer, defense["strength"])
        elif defense["type"] == "gated":
            gate_config = GateConfig(
                probe_path=probe_dir / "best_probe.pkl",
                steering_vector_path=vector_path,
                probe_layer=best_layer,
                steering_layer=steering_layer,
                controller_type=defense["controller"],
                threshold=defense["threshold"],
                max_strength=defense["strength"],
                gain=defense.get("gain", 2.0),
            )
            pgs = ProbeGatedSteering(model, gate_config)

        _, metrics = run_safety_eval(
            model, tokenizer, all_prompts, name, model_name, device, pgs=pgs,
        )

        if pgs:
            pgs.remove()
        if static:
            static.remove()

        safety_results[name] = {"dsr": metrics.dsr, "asr": metrics.asr, "fpr": metrics.fpr}
        print(f"    DSR={metrics.dsr:.3f}  ASR={metrics.asr:.3f}  FPR={metrics.fpr:.3f}")
        for at, asr in metrics.per_attack_asr.items():
            print(f"      {at}: ASR={asr:.3f}")

    # ================================================================
    # STEP 4: Open-ended utility (7 defenses × 100 queries)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"STEP 4: Open-ended utility evaluation")
    print(f"{'='*60}")

    utility_results = {}
    for defense in defenses:
        name = defense["name"]
        print(f"\n  [{name}] Open-ended utility...")

        pgs = None
        static = None

        if defense["type"] == "static":
            static = StaticSteering(model, str(vector_path), steering_layer, defense["strength"])
        elif defense["type"] == "gated":
            gate_config = GateConfig(
                probe_path=probe_dir / "best_probe.pkl",
                steering_vector_path=vector_path,
                probe_layer=best_layer,
                steering_layer=steering_layer,
                controller_type=defense["controller"],
                threshold=defense["threshold"],
                max_strength=defense["strength"],
                gain=defense.get("gain", 2.0),
            )
            pgs = ProbeGatedSteering(model, gate_config)

        result = run_openended_eval(model, tokenizer, device, defense_name=name)
        save_openended_results(output_dir / "openended", result)

        if pgs:
            pgs.remove()
        if static:
            static.remove()

        utility_results[name] = {
            "hedging_rate": result.hedging_rate,
            "over_refusal_rate": result.over_refusal_rate,
            "mean_response_length": result.mean_response_length,
        }
        print(f"    Hedging={result.hedging_rate:.1%}  Refusal={result.over_refusal_rate:.1%}  "
              f"MeanLen={result.mean_response_length:.0f}")

    # Free model
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # ================================================================
    # STEP 5: Report cross-model Pareto table
    # ================================================================
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"Probe: layer {best_layer}, AUROC {best_result.auroc:.4f}, FPR {best_result.fpr:.4f}")
    print(f"Vector: layer {steering_layer}, separation {vec_metadata['separation']:.4f}")
    print()
    print(f"{'Defense':<25} {'DSR':>6} {'Hedge%':>8} {'Dominance':>12}")
    print(f"{'-'*55}")

    gated_bb = safety_results.get("gated_bb_1.3_t0.3", {})
    static_13 = safety_results.get("static_1.3", {})
    gated_hedge = utility_results.get("gated_bb_1.3_t0.3", {}).get("hedging_rate", 0)
    static_hedge = utility_results.get("static_1.3", {}).get("hedging_rate", 0)

    for defense in defenses:
        name = defense["name"]
        dsr = safety_results[name]["dsr"]
        hedge = utility_results[name]["hedging_rate"]

        # Check if dominated by gated BB
        gated_dsr = gated_bb.get("dsr", 0)
        dominance = ""
        if name == "gated_bb_1.3_t0.3":
            dominance = "GATED"
        elif dsr <= gated_dsr and hedge >= gated_hedge and name != "gated_bb_1.3_t0.3":
            if dsr < gated_dsr or hedge > gated_hedge:
                dominance = "dominated"

        print(f"{name:<25} {dsr:>6.3f} {hedge:>7.1%} {dominance:>12}")

    # Critical check
    hedge_gap = static_hedge - gated_hedge
    dsr_diff = gated_bb.get("dsr", 0) - static_13.get("dsr", 0)

    print(f"\nCRITICAL: Gated BB vs Static 1.3")
    print(f"  DSR:     Gated {gated_bb.get('dsr', 0):.3f} vs Static {static_13.get('dsr', 0):.3f} (diff: {dsr_diff:+.3f})")
    print(f"  Hedging: Gated {gated_hedge:.1%} vs Static {static_hedge:.1%} (gap: {hedge_gap:+.1%})")
    if gated_bb.get("dsr", 0) >= static_13.get("dsr", 0) and hedge_gap > 0.05:
        print(f"  VERDICT: PARETO DOMINANCE CONFIRMED for {model_name}")
    elif hedge_gap > 0.05:
        print(f"  VERDICT: Utility advantage confirmed, DSR gap: {dsr_diff:+.3f}")
    else:
        print(f"  VERDICT: NO PARETO DOMINANCE — hedging gap too small ({hedge_gap:+.1%})")

    # Save all results
    combined = {
        "model": model_name,
        "probe": {"best_layer": best_layer, "auroc": best_result.auroc, "fpr": best_result.fpr},
        "vector": {"layer": steering_layer, "separation": vec_metadata["separation"]},
        "safety": safety_results,
        "utility": utility_results,
        "pareto_check": {
            "gated_dsr": gated_bb.get("dsr", 0),
            "static_dsr": static_13.get("dsr", 0),
            "gated_hedging": gated_hedge,
            "static_hedging": static_hedge,
            "hedging_gap": hedge_gap,
            "dominance": gated_bb.get("dsr", 0) >= static_13.get("dsr", 0) and hedge_gap > 0.05,
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
