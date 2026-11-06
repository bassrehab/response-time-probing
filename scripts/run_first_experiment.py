#!/usr/bin/env python3
"""First real experiment: probe-gated vs ungated on prefilling + intent laundering.

Tests on Mistral-7B with:
  - 20 prefilling attacks + 20 benign prompts
  - 20 intent laundering attacks + 20 benign prompts

Compares: no defense (ungated) vs probe-gated steering.
Reports: ASR, DSR, FPR, per-attack breakdown.
"""

import json
import time
from pathlib import Path

import torch

from src.attacks.registry import load_attack_set
from src.eval.safety_eval import run_safety_eval, save_eval_results
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.models.loader import get_device, load_model
from src.utils.paths import probe_results_dir, vector_results_dir, RESULTS_DIR
from src.utils.seeding import seed_everything


def main():
    seed_everything(42)
    device = get_device()
    model_name = "mistral-7b"

    print(f"Loading {model_name} on {device}...")
    model, tokenizer, config = load_model(model_name, device=device)

    # Set up probe-gated steering
    probe_dir = probe_results_dir(model_name, "jailbreak")
    vector_dir = vector_results_dir(model_name, "refusal")

    gate_config = GateConfig(
        probe_path=probe_dir / "best_probe.pkl",
        steering_vector_path=vector_dir / "vector_layer14.pt",
        probe_layer=6,
        steering_layer=14,
        controller_type="proportional",
        threshold=0.5,
        max_strength=0.8,
        gain=2.0,
    )

    output_base = RESULTS_DIR / "first_experiment" / model_name

    for attack_name in ["prefilling", "intent_laundering"]:
        print(f"\n{'='*60}")
        print(f"ATTACK: {attack_name}")
        print(f"{'='*60}")

        attack_prompts = load_attack_set(
            attack_name, n_attacks=20, include_benign=True, n_benign=20
        )
        n_attack = sum(1 for a in attack_prompts if a["attack_type"] != "benign")
        n_benign = sum(1 for a in attack_prompts if a["attack_type"] == "benign")
        print(f"  Loaded {n_attack} attack + {n_benign} benign prompts")

        # --- Ungated ---
        print(f"\n  Running UNGATED...")
        t0 = time.time()
        samples_ungated, metrics_ungated = run_safety_eval(
            model, tokenizer, attack_prompts,
            defense_type="none",
            model_name=model_name,
            device=device,
            pgs=None,
        )
        print(f"  Done in {time.time() - t0:.1f}s")
        print(f"  ASR={metrics_ungated.asr:.3f}  DSR={metrics_ungated.dsr:.3f}  "
              f"FPR={metrics_ungated.fpr:.3f}  Latency={metrics_ungated.mean_latency_ms:.0f}ms")
        if metrics_ungated.per_attack_asr:
            for at, asr in metrics_ungated.per_attack_asr.items():
                print(f"    {at}: ASR={asr:.3f}")

        output_dir = output_base / attack_name
        save_eval_results(output_dir, samples_ungated, metrics_ungated, "none", model_name)

        # --- Probe-gated ---
        print(f"\n  Running PROBE-GATED...")
        pgs = ProbeGatedSteering(model, gate_config)
        t0 = time.time()
        samples_gated, metrics_gated = run_safety_eval(
            model, tokenizer, attack_prompts,
            defense_type="probe_gated",
            model_name=model_name,
            device=device,
            pgs=pgs,
        )
        pgs.remove()
        print(f"  Done in {time.time() - t0:.1f}s")
        print(f"  ASR={metrics_gated.asr:.3f}  DSR={metrics_gated.dsr:.3f}  "
              f"FPR={metrics_gated.fpr:.3f}  Latency={metrics_gated.mean_latency_ms:.0f}ms")
        if metrics_gated.per_attack_asr:
            for at, asr in metrics_gated.per_attack_asr.items():
                print(f"    {at}: ASR={asr:.3f}")

        save_eval_results(output_dir, samples_gated, metrics_gated, "probe_gated", model_name)

        # --- Comparison ---
        print(f"\n  COMPARISON ({attack_name}):")
        print(f"    {'Metric':<12} {'Ungated':>10} {'Gated':>10} {'Delta':>10}")
        print(f"    {'ASR':<12} {metrics_ungated.asr:>10.3f} {metrics_gated.asr:>10.3f} "
              f"{metrics_gated.asr - metrics_ungated.asr:>+10.3f}")
        print(f"    {'DSR':<12} {metrics_ungated.dsr:>10.3f} {metrics_gated.dsr:>10.3f} "
              f"{metrics_gated.dsr - metrics_ungated.dsr:>+10.3f}")
        print(f"    {'FPR':<12} {metrics_ungated.fpr:>10.3f} {metrics_gated.fpr:>10.3f} "
              f"{metrics_gated.fpr - metrics_ungated.fpr:>+10.3f}")
        print(f"    {'Latency(ms)':<12} {metrics_ungated.mean_latency_ms:>10.0f} "
              f"{metrics_gated.mean_latency_ms:>10.0f} "
              f"{metrics_gated.mean_latency_ms - metrics_ungated.mean_latency_ms:>+10.0f}")

    print(f"\nResults saved to {output_base}")


if __name__ == "__main__":
    main()
