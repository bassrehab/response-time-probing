#!/usr/bin/env python3
"""Sweep steering strength and threshold on prefilling attacks.

Tests on Mistral-7B:
  - Static steering: strengths [0.5, 0.8, 1.0, 1.3]
  - Probe-gated proportional: strengths [0.5, 0.8, 1.0, 1.3] × thresholds [0.3, 0.5]
  - Probe-gated bang-bang: strengths [0.8, 1.0, 1.3] × thresholds [0.3, 0.5]
  - Ungated baseline (once)

Uses 20 prefilling attacks + 20 benign prompts.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.attacks.registry import load_attack_set
from src.baselines.static_steering.infer import StaticSteering
from src.eval.safety_eval import run_safety_eval
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.models.loader import get_device, load_model
from src.utils.paths import probe_results_dir, vector_results_dir, RESULTS_DIR
from src.utils.run_logger import capture_run_metadata, save_run_metadata
from src.utils.seeding import seed_everything


def main():
    seed_everything(42)
    device = get_device()
    model_name = "mistral-7b"

    print(f"Loading {model_name} on {device}...")
    model, tokenizer, config = load_model(model_name, device=device)

    probe_dir = probe_results_dir(model_name, "jailbreak")
    vector_dir = vector_results_dir(model_name, "refusal")
    vector_path = vector_dir / "vector_layer14.pt"

    attack_prompts = load_attack_set("prefilling", n_attacks=20, include_benign=True, n_benign=20)
    output_base = RESULTS_DIR / "strength_sweep" / model_name

    # Save run metadata
    metadata = capture_run_metadata("strength_sweep", __file__)
    save_run_metadata(output_base, metadata)

    all_results = []

    # --- Ungated baseline ---
    print("\n[1] UNGATED BASELINE")
    _, metrics = run_safety_eval(
        model, tokenizer, attack_prompts, "none", model_name, device
    )
    result = {"defense": "none", "strength": 0, "threshold": 0, "controller": "none",
              "asr": metrics.asr, "dsr": metrics.dsr, "fpr": metrics.fpr,
              "latency": metrics.mean_latency_ms}
    all_results.append(result)
    print(f"  ASR={metrics.asr:.3f}  DSR={metrics.dsr:.3f}  FPR={metrics.fpr:.3f}")

    # --- Static steering sweep ---
    strengths = [0.5, 0.8, 1.0, 1.3]
    for strength in strengths:
        print(f"\n[S] STATIC strength={strength}")
        static = StaticSteering(model, str(vector_path), steering_layer=14, strength=strength)
        _, metrics = run_safety_eval(
            model, tokenizer, attack_prompts, f"static_{strength}", model_name, device
        )
        static.remove()
        result = {"defense": "static", "strength": strength, "threshold": 0,
                  "controller": "static", "asr": metrics.asr, "dsr": metrics.dsr,
                  "fpr": metrics.fpr, "latency": metrics.mean_latency_ms}
        all_results.append(result)
        print(f"  ASR={metrics.asr:.3f}  DSR={metrics.dsr:.3f}  FPR={metrics.fpr:.3f}")

    # --- Probe-gated proportional sweep ---
    thresholds = [0.3, 0.5]
    for strength in strengths:
        for threshold in thresholds:
            print(f"\n[P] PROPORTIONAL strength={strength} threshold={threshold}")
            gate_config = GateConfig(
                probe_path=probe_dir / "best_probe.pkl",
                steering_vector_path=vector_path,
                probe_layer=6, steering_layer=14,
                controller_type="proportional",
                threshold=threshold, max_strength=strength, gain=2.0,
            )
            pgs = ProbeGatedSteering(model, gate_config)
            _, metrics = run_safety_eval(
                model, tokenizer, attack_prompts,
                f"proportional_s{strength}_t{threshold}",
                model_name, device, pgs=pgs,
            )
            pgs.remove()
            result = {"defense": "probe_gated", "strength": strength,
                      "threshold": threshold, "controller": "proportional",
                      "asr": metrics.asr, "dsr": metrics.dsr, "fpr": metrics.fpr,
                      "latency": metrics.mean_latency_ms}
            all_results.append(result)
            print(f"  ASR={metrics.asr:.3f}  DSR={metrics.dsr:.3f}  FPR={metrics.fpr:.3f}")

    # --- Probe-gated bang-bang sweep ---
    bb_strengths = [0.8, 1.0, 1.3]
    for strength in bb_strengths:
        for threshold in thresholds:
            print(f"\n[B] BANG-BANG strength={strength} threshold={threshold}")
            gate_config = GateConfig(
                probe_path=probe_dir / "best_probe.pkl",
                steering_vector_path=vector_path,
                probe_layer=6, steering_layer=14,
                controller_type="bang_bang",
                threshold=threshold, max_strength=strength,
            )
            pgs = ProbeGatedSteering(model, gate_config)
            _, metrics = run_safety_eval(
                model, tokenizer, attack_prompts,
                f"bangbang_s{strength}_t{threshold}",
                model_name, device, pgs=pgs,
            )
            pgs.remove()
            result = {"defense": "probe_gated", "strength": strength,
                      "threshold": threshold, "controller": "bang_bang",
                      "asr": metrics.asr, "dsr": metrics.dsr, "fpr": metrics.fpr,
                      "latency": metrics.mean_latency_ms}
            all_results.append(result)
            print(f"  ASR={metrics.asr:.3f}  DSR={metrics.dsr:.3f}  FPR={metrics.fpr:.3f}")

    # Save all results
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Defense':<15} {'Controller':<14} {'Strength':>8} {'Thresh':>7} "
          f"{'ASR':>6} {'DSR':>6} {'FPR':>6}")
    print(f"{'='*80}")
    for r in all_results:
        print(f"{r['defense']:<15} {r['controller']:<14} {r['strength']:>8.1f} "
              f"{r['threshold']:>7.1f} {r['asr']:>6.3f} {r['dsr']:>6.3f} {r['fpr']:>6.3f}")

    print(f"\nResults saved to {output_base}")


if __name__ == "__main__":
    main()
