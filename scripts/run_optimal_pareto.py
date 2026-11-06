#!/usr/bin/env python3
"""Pareto sweep with optimal configs from strength sweep.

Tests on Mistral-7B:
  - Ungated baseline
  - Static steering: [0.5, 0.8, 1.0, 1.3]
  - Probe-gated bang-bang s=1.3 t=0.3 (optimal from sweep)
  - Probe-gated proportional s=1.0 t=0.5 (for comparison)

Each defense: safety eval (prefilling + intent laundering + benign) + MMLU utility.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.attacks.datasets.prompts import BENIGN_QUERIES
from src.attacks.registry import load_attack_set
from src.eval.pareto import ParetoPoint, compute_pareto_frontier
from src.eval.safety_eval import run_safety_eval
from src.eval.utility_eval import run_utility_eval
from src.baselines.static_steering.infer import StaticSteering
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

    with open(probe_dir / "best_layer.json") as f:
        probe_layer = json.load(f)["layer"]

    # Build attack set
    all_prompts = []
    for attack in ["prefilling", "intent_laundering"]:
        all_prompts.extend(load_attack_set(attack, n_attacks=20, include_benign=False))
    for i in range(min(20, len(BENIGN_QUERIES))):
        all_prompts.append({"prompt": BENIGN_QUERIES[i], "attack_type": "benign"})

    n_atk = sum(1 for p in all_prompts if p["attack_type"] != "benign")
    n_ben = sum(1 for p in all_prompts if p["attack_type"] == "benign")
    print(f"Prompts: {n_atk} attack + {n_ben} benign = {len(all_prompts)}")

    output_dir = RESULTS_DIR / "pareto_optimal" / model_name
    metadata = capture_run_metadata("pareto_optimal", __file__)
    save_run_metadata(output_dir, metadata)

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

    points = []

    for defense in defenses:
        name = defense["name"]
        print(f"\n{'='*50}")
        print(f"Defense: {name}")
        print(f"{'='*50}")

        pgs = None
        static = None

        if defense["type"] == "static":
            static = StaticSteering(model, str(vector_path), 14, defense["strength"])
        elif defense["type"] == "gated":
            gate_config = GateConfig(
                probe_path=probe_dir / "best_probe.pkl",
                steering_vector_path=vector_path,
                probe_layer=probe_layer,
                steering_layer=14,
                controller_type=defense["controller"],
                threshold=defense["threshold"],
                max_strength=defense["strength"],
                gain=defense.get("gain", 2.0),
            )
            pgs = ProbeGatedSteering(model, gate_config)

        # Safety
        samples, metrics = run_safety_eval(
            model, tokenizer, all_prompts, name, model_name, device, pgs=pgs,
        )

        # Utility
        if pgs:
            pgs.reset()
        utility = run_utility_eval(model, tokenizer, device)

        if pgs:
            pgs.remove()
        if static:
            static.remove()

        point = ParetoPoint(
            model=model_name, defense=name,
            dsr=metrics.dsr, utility=utility.accuracy,
            fpr=metrics.fpr, latency_ms=metrics.mean_latency_ms,
            config=defense,
        )
        points.append(point)

        print(f"  DSR={metrics.dsr:.3f}  MMLU={utility.accuracy:.3f}  FPR={metrics.fpr:.3f}")
        for at, asr in metrics.per_attack_asr.items():
            print(f"    {at}: ASR={asr:.3f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pareto_results.json", "w") as f:
        json.dump([asdict(p) for p in points], f, indent=2)

    # Frontier
    frontier = compute_pareto_frontier(points)

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Defense':<25} {'DSR':>6} {'MMLU':>6} {'FPR':>6} {'Pareto':>8}")
    print(f"{'='*70}")
    frontier_names = {p.defense for p in frontier}
    for p in points:
        on_frontier = "***" if p.defense in frontier_names else ""
        print(f"{p.defense:<25} {p.dsr:>6.3f} {p.utility:>6.3f} {p.fpr:>6.3f} {on_frontier:>8}")

    print(f"\nPareto frontier:")
    for p in frontier:
        print(f"  {p.defense}: DSR={p.dsr:.3f}, MMLU={p.utility:.3f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
