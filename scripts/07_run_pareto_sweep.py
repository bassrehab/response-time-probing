#!/usr/bin/env python3
"""Pareto sweep — THE headline experiment.

For each (model, defense, attack):
  1. Apply defense
  2. Generate responses to attack prompts
  3. Evaluate safety (ASR/DSR) and utility (MMLU)
  4. Compute Pareto frontier

Models:    [mistral-7b, gemma2-9b, qwen25-7b]
Defenses:  [none, prompting, static_steering, probe_gated]
           (AlphaSteer and CAST added when baselines are ready)
Attacks:   [prefilling, intent_laundering, benign]
           (GCG, AutoDAN, PAIR, DeepInception added when implemented)

Usage:
    # Single model, quick test
    python scripts/07_run_pareto_sweep.py --model mistral-7b --quick

    # Full sweep (all models, all implemented defenses/attacks)
    python scripts/07_run_pareto_sweep.py --models all
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.attacks.registry import load_attack_set
from src.baselines.prompting.infer import format_with_system_prompt
from src.baselines.static_steering.infer import StaticSteering
from src.eval.metrics import compute_safety_metrics
from src.eval.pareto import ParetoPoint, compute_pareto_frontier, check_pareto_dominance
from src.eval.safety_eval import run_safety_eval, save_eval_results
from src.eval.utility_eval import run_utility_eval
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.models.configs import MODEL_CONFIGS, get_config
from src.models.loader import get_device, load_model
from src.utils.paths import probe_results_dir, vector_results_dir, RESULTS_DIR
from src.utils.run_logger import capture_run_metadata, save_run_metadata
from src.utils.seeding import seed_everything


def run_pareto_for_model(
    model_name: str,
    device: str,
    attack_names: list[str],
    n_attacks: int = 20,
    n_benign: int = 20,
    quick: bool = False,
) -> list[ParetoPoint]:
    """Run full Pareto sweep for a single model.

    Returns list of ParetoPoints for all defenses.
    """
    config = get_config(model_name)
    probe_dir = probe_results_dir(model_name, "jailbreak")
    vector_dir = vector_results_dir(model_name, "refusal")
    vector_path = vector_dir / "vector_layer14.pt"

    # Check if probe and vector exist
    if not (probe_dir / "best_probe.pkl").exists():
        print(f"  WARNING: No probe for {model_name}, skipping gated defenses")
        has_probe = False
    else:
        has_probe = True

    if not vector_path.exists():
        print(f"  WARNING: No steering vector for {model_name}, skipping steering defenses")
        has_vector = False
    else:
        has_vector = True

    # Load model
    print(f"\n  Loading {model_name}...")
    t0 = time.time()
    model, tokenizer, model_config = load_model(model_name, device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Prepare attack prompts (combined across attack types)
    all_attack_prompts = []
    for attack_name in attack_names:
        attacks = load_attack_set(
            attack_name, n_attacks=n_attacks,
            include_benign=False,
        )
        all_attack_prompts.extend(attacks)

    # Add benign prompts once
    from src.attacks.datasets.prompts import BENIGN_QUERIES
    for i in range(min(n_benign, len(BENIGN_QUERIES))):
        all_attack_prompts.append({
            "prompt": BENIGN_QUERIES[i],
            "attack_type": "benign",
        })

    print(f"  Total prompts: {len(all_attack_prompts)} "
          f"({sum(1 for a in all_attack_prompts if a['attack_type'] != 'benign')} attack, "
          f"{sum(1 for a in all_attack_prompts if a['attack_type'] == 'benign')} benign)")

    output_dir = RESULTS_DIR / "pareto" / model_name
    pareto_points = []

    # Define defenses to test
    defenses = [
        {"name": "none", "type": "none"},
    ]

    if has_vector:
        if quick:
            defenses.append({"name": "static_1.0", "type": "static", "strength": 1.0})
        else:
            for s in [0.5, 0.8, 1.0, 1.3]:
                defenses.append({"name": f"static_{s}", "type": "static", "strength": s})

    if has_probe and has_vector:
        if quick:
            defenses.append({
                "name": "gated_p_s1.0_t0.5", "type": "probe_gated",
                "controller": "proportional", "strength": 1.0,
                "threshold": 0.5, "gain": 2.0,
            })
        else:
            for s in [0.8, 1.0, 1.3]:
                for t in [0.3, 0.5]:
                    defenses.append({
                        "name": f"gated_p_s{s}_t{t}", "type": "probe_gated",
                        "controller": "proportional", "strength": s,
                        "threshold": t, "gain": 2.0,
                    })
                    defenses.append({
                        "name": f"gated_bb_s{s}_t{t}", "type": "probe_gated",
                        "controller": "bang_bang", "strength": s,
                        "threshold": t,
                    })

    for defense in defenses:
        defense_name = defense["name"]
        print(f"\n  --- Defense: {defense_name} ---")

        # Set up defense
        pgs = None
        static = None

        if defense["type"] == "static":
            static = StaticSteering(
                model, str(vector_path),
                steering_layer=14, strength=defense["strength"],
            )
        elif defense["type"] == "probe_gated":
            # Read best probe layer
            with open(probe_dir / "best_layer.json") as f:
                probe_layer = json.load(f)["layer"]

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

        # Safety eval
        samples, metrics = run_safety_eval(
            model, tokenizer, all_attack_prompts,
            defense_type=defense_name,
            model_name=model_name,
            device=device,
            pgs=pgs,
        )
        save_eval_results(output_dir, samples, metrics, defense_name, model_name)

        # Utility eval
        utility = run_utility_eval(model, tokenizer, device)

        # Clean up defense
        if pgs is not None:
            pgs.remove()
        if static is not None:
            static.remove()

        point = ParetoPoint(
            model=model_name,
            defense=defense_name,
            dsr=metrics.dsr,
            utility=utility.accuracy,
            fpr=metrics.fpr,
            latency_ms=metrics.mean_latency_ms,
            config=defense,
        )
        pareto_points.append(point)

        print(f"  DSR={metrics.dsr:.3f}  Utility={utility.accuracy:.3f}  "
              f"FPR={metrics.fpr:.3f}  Latency={metrics.mean_latency_ms:.0f}ms")
        if metrics.per_attack_asr:
            for at, asr in metrics.per_attack_asr.items():
                print(f"    {at}: ASR={asr:.3f}")
        print(f"  MMLU: {utility.n_correct}/{utility.n_total} "
              f"({utility.accuracy:.1%})")

    # Free model
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return pareto_points


def main():
    parser = argparse.ArgumentParser(description="Run Pareto sweep")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer defense configs")
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

    # Currently implemented attacks
    attack_names = ["prefilling", "intent_laundering"]

    metadata = capture_run_metadata(
        "pareto_sweep", __file__,
        args={"models": model_names, "attacks": attack_names, "quick": args.quick},
    )
    save_run_metadata(RESULTS_DIR / "pareto", metadata)

    all_points = []
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        points = run_pareto_for_model(
            model_name, device, attack_names,
            n_attacks=20, n_benign=20, quick=args.quick,
        )
        all_points.extend(points)

    # Compute Pareto frontier per model
    output_dir = RESULTS_DIR / "pareto"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = [asdict(p) for p in all_points]
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_data, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("PARETO SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Defense':<25} {'DSR':>6} {'Utility':>8} {'FPR':>6}")
    print("-" * 60)
    for p in all_points:
        print(f"{p.model:<15} {p.defense:<25} {p.dsr:>6.3f} "
              f"{p.utility:>8.3f} {p.fpr:>6.3f}")

    # Per-model Pareto frontiers
    for model_name in model_names:
        model_points = [p for p in all_points if p.model == model_name]
        frontier = compute_pareto_frontier(model_points)
        print(f"\nPareto frontier for {model_name}:")
        for p in frontier:
            print(f"  {p.defense:<25} DSR={p.dsr:.3f}  Utility={p.utility:.3f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
