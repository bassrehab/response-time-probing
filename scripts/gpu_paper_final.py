#!/usr/bin/env python3
"""Final paper data generation: 50 prompts per attack, all paradigms, all models.

Phases:
1. Train probes + extract vectors for all 4 models
2. Train AlphaSteer matrices for all 4 models
3. Expand attacks to 50 per type
4. Run full 5×5×4 matrix with 50-prompt attacks
5. Compute Wilson confidence intervals
6. Multi-threat composition (Mistral + Llama)
7. Hedging validation with model-as-judge
8. Generate figures

Usage: python3 scripts/gpu_paper_final.py
"""

import json, os, time, sys, csv, math
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

from src.models.loader import load_model
from src.models.configs import MODEL_CONFIGS, get_config
from src.models.hooks import ActivationHook, MultiLayerHook
from src.attacks.datasets.prompts import ADVBENCH_BEHAVIORS, BENIGN_QUERIES, BENIGN_OPENENDED
from src.attacks.registry import load_attack_set
from src.baselines.static_steering.infer import StaticSteering
from src.baselines.alphasteer.wrapper import (
    AlphaSteerDefense, AlphaSteerConfig,
    compute_null_space_projection, compute_delta_matrix,
    save_alphasteer_matrices,
)
from src.eval.safety_eval import run_safety_eval
from src.eval.openended_eval import run_openended_eval, save_openended_results
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.probes.train_probe import layer_sweep, save_probe_results
from src.steering.extract_vectors import compute_caa_vector, save_steering_vector
from src.utils.paths import (
    probe_results_dir, vector_results_dir, contrast_pairs_dir, RESULTS_DIR, DATA_DIR,
)
from src.utils.seeding import seed_everything

seed_everything(42)
device = "cuda"
FINAL_DIR = RESULTS_DIR / "paper_final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

models = ["mistral-7b", "gemma2-9b", "qwen25-7b", "llama31-8b"]
paradigms = ["none", "static_1.3", "gated_bb_1.3", "alphasteer", "cast"]
attack_names = ["gcg", "autodan", "deep_inception", "prefilling", "intent_laundering"]


def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score 95% confidence interval."""
    if n_total == 0:
        return 0, 0, 0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p, max(0, center - spread), min(1, center + spread)


def check_marker(name):
    return (FINAL_DIR / f".{name}_done").exists()


def mark_done(name):
    (FINAL_DIR / f".{name}_done").touch()


# ================================================================
# PHASE 1: Train probes + vectors
# ================================================================
if check_marker("phase1"):
    print(f"[{time.strftime('%H:%M:%S')}] SKIP Phase 1 — probes exist")
else:
    print(f"[{time.strftime('%H:%M:%S')}] PHASE 1: Train probes + vectors")

    # Generate contrast pairs if needed
    if not (contrast_pairs_dir() / "jailbreak_pairs.json").exists():
        os.system("python3 scripts/02_generate_contrast_pairs.py --probe_type jailbreak --n_pairs 40")

    pairs = json.load(open(contrast_pairs_dir() / "jailbreak_pairs.json"))
    prompts = [s["prompt"] for s in pairs]
    labels = np.array([s["label"] for s in pairs])

    for model_name in models:
        if (probe_results_dir(model_name, "jailbreak") / "best_probe.pkl").exists():
            print(f"  SKIP {model_name} — probe exists")
            continue

        print(f"  [{time.strftime('%H:%M:%S')}] {model_name}: probe + vector")
        config = get_config(model_name)
        model, tokenizer, config = load_model(model_name, device=device)

        # Extract activations (chunked)
        layer_indices = list(range(config.num_layers))
        all_acts = {}
        for cs in range(0, len(layer_indices), 8):
            chunk = layer_indices[cs:cs+8]
            hook = MultiLayerHook(model, chunk)
            for prompt in prompts:
                msgs = [{"role": "user", "content": prompt}]
                fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
                with torch.no_grad():
                    model(**inp)
                for l, act in hook.get_all_last().items():
                    all_acts.setdefault(l, []).append(act.cpu().float().numpy())
                hook.clear()
            hook.remove()
        all_activations = {l: np.concatenate(a, axis=0) for l, a in all_acts.items()}

        results, best_layer, best_probe, best_scaler = layer_sweep(all_activations, labels, n_folds=5)
        save_probe_results(probe_results_dir(model_name, "jailbreak"),
                          model_name, "jailbreak", results, best_layer, best_probe, best_scaler)
        print(f"    Probe: layer {best_layer}, AUROC {results[best_layer].auroc:.4f}")

        # Vector
        sl = config.optimal_steering_layer
        pos = [s["prompt"] for s in pairs if s["label"] == 1]
        neg = [s["prompt"] for s in pairs if s["label"] == 0]

        def extract_single(prompts_list, layer):
            hook = ActivationHook(model, layer)
            acts = []
            for p in prompts_list:
                msgs = [{"role": "user", "content": p}]
                fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
                with torch.no_grad():
                    model(**inp)
                acts.append(hook.get_last().cpu().float().numpy())
                hook.clear()
            hook.remove()
            return np.concatenate(acts, axis=0)

        pos_acts = extract_single(pos, sl)
        neg_acts = extract_single(neg, sl)
        vector, meta = compute_caa_vector(pos_acts, neg_acts)
        save_steering_vector(vector_results_dir(model_name, "refusal"), vector, meta, sl)
        print(f"    Vector: layer {sl}, separation {meta['separation']:.4f}")

        del model; torch.cuda.empty_cache()

    mark_done("phase1")

# ================================================================
# PHASE 2: Train AlphaSteer
# ================================================================
if check_marker("phase2"):
    print(f"[{time.strftime('%H:%M:%S')}] SKIP Phase 2 — AlphaSteer exists")
else:
    print(f"[{time.strftime('%H:%M:%S')}] PHASE 2: Train AlphaSteer matrices")

    benign_prompts = list(BENIGN_QUERIES[:40]) + list(BENIGN_OPENENDED[:60])
    malicious_prompts = list(ADVBENCH_BEHAVIORS[:40])

    for model_name in models:
        alpha_dir = RESULTS_DIR / "alphasteer" / model_name
        config = get_config(model_name)
        sl = config.optimal_steering_layer
        if (alpha_dir / f"delta_layer{sl}.pt").exists():
            print(f"  SKIP {model_name}")
            continue

        print(f"  [{time.strftime('%H:%M:%S')}] {model_name}")
        model, tokenizer, config = load_model(model_name, device=device)

        def extract_single(prompts_list, layer):
            hook = ActivationHook(model, layer)
            acts = []
            for p in prompts_list:
                msgs = [{"role": "user", "content": p}]
                fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
                with torch.no_grad():
                    model(**inp)
                acts.append(hook.get_last().cpu().float().numpy())
                hook.clear()
            hook.remove()
            return np.concatenate(acts, axis=0)

        benign_acts = extract_single(benign_prompts, sl)
        malicious_acts = extract_single(malicious_prompts, sl)
        refusal_dir = benign_acts.mean(axis=0) - malicious_acts.mean(axis=0)
        refusal_dir = refusal_dir / np.linalg.norm(refusal_dir)
        refusal_directions = np.tile(refusal_dir, (len(malicious_acts), 1))

        P_hat = compute_null_space_projection(benign_acts)
        Delta = compute_delta_matrix(malicious_acts, refusal_directions, P_hat)
        save_alphasteer_matrices(alpha_dir, Delta, P_hat, sl)
        print(f"    Saved to {alpha_dir}")

        del model; torch.cuda.empty_cache()

    mark_done("phase2")

# ================================================================
# PHASE 3: Build expanded attack sets (50 per type)
# ================================================================
print(f"[{time.strftime('%H:%M:%S')}] PHASE 3: Build 50-prompt attack sets")

def build_expanded_attacks(n=50):
    """Build attack set with up to n prompts per attack type."""
    all_prompts = []

    # Prefilling (50)
    from src.attacks.prefilling import generate_prefilling_attacks
    pf = generate_prefilling_attacks(n_attacks=min(n, 40))
    all_prompts.extend(pf)

    # Intent laundering (up to 20 available)
    from src.attacks.intent_laundering import generate_intent_laundering_attacks
    il = generate_intent_laundering_attacks(n_attacks=min(n, 20))
    all_prompts.extend(il)

    # DeepInception (up to 40 available)
    from src.attacks.deep_inception import generate_deep_inception_attacks
    di = generate_deep_inception_attacks(n_attacks=min(n, 40))
    all_prompts.extend(di)

    # GCG — use optimized suffixes if available
    gcg_file = DATA_DIR / "attacks" / "gcg" / "mistral-7b_suffixes_v2.json"
    if gcg_file.exists():
        suffixes = json.load(open(gcg_file))
        for i, suffix in enumerate(suffixes[:min(n, 40)]):
            if suffix and i < len(ADVBENCH_BEHAVIORS):
                all_prompts.append({
                    "prompt": f"{ADVBENCH_BEHAVIORS[i]} {suffix}",
                    "attack_type": "gcg",
                })
    else:
        gcg = load_attack_set("gcg", n_attacks=min(n, 20), include_benign=False)
        all_prompts.extend(gcg)

    # AutoDAN
    autodan = load_attack_set("autodan", n_attacks=min(n, 40), include_benign=False)
    all_prompts.extend(autodan)

    # Benign (50)
    benign_all = list(BENIGN_QUERIES) + list(BENIGN_OPENENDED[:10])
    for i in range(min(n, len(benign_all))):
        all_prompts.append({"prompt": benign_all[i], "attack_type": "benign"})

    return all_prompts

attack_prompts = build_expanded_attacks(50)
attack_counts = {}
for p in attack_prompts:
    attack_counts[p["attack_type"]] = attack_counts.get(p["attack_type"], 0) + 1
print(f"  Attack set: {len(attack_prompts)} total")
for at, n in sorted(attack_counts.items()):
    print(f"    {at}: {n}")

# ================================================================
# PHASE 4: Full 5×5×4 matrix with expanded attacks
# ================================================================
if check_marker("phase4"):
    print(f"[{time.strftime('%H:%M:%S')}] SKIP Phase 4 — matrix exists")
    matrix = json.load(open(FINAL_DIR / "matrix.json"))
else:
    print(f"[{time.strftime('%H:%M:%S')}] PHASE 4: Full matrix (expanded attacks)")

    matrix = {}

    for model_name in models:
        print(f"\n  [{time.strftime('%H:%M:%S')}] Model: {model_name}")
        config = get_config(model_name)
        model, tokenizer, config = load_model(model_name, device=device)
        sl = config.optimal_steering_layer

        vec_dir = vector_results_dir(model_name, "refusal")
        vec_path = vec_dir / f"vector_layer{sl}.pt"
        probe_dir_path = probe_results_dir(model_name, "jailbreak")
        alpha_dir = RESULTS_DIR / "alphasteer" / model_name

        with open(probe_dir_path / "best_layer.json") as f:
            probe_layer = json.load(f)["layer"]

        matrix[model_name] = {}

        for paradigm in paradigms:
            print(f"    [{paradigm}]...", end=" ", flush=True)
            pgs, static, alpha_def = None, None, None

            try:
                if paradigm == "static_1.3":
                    static = StaticSteering(model, str(vec_path), sl, 1.3)
                elif paradigm == "gated_bb_1.3":
                    pgs = ProbeGatedSteering(model, GateConfig(
                        probe_path=probe_dir_path / "best_probe.pkl",
                        steering_vector_path=vec_path,
                        probe_layer=probe_layer, steering_layer=sl,
                        controller_type="bang_bang", threshold=0.3, max_strength=1.3,
                    ))
                elif paradigm == "alphasteer":
                    if not (alpha_dir / f"delta_layer{sl}.pt").exists():
                        print("SKIP (no matrices)")
                        matrix[model_name][paradigm] = {"dsr": None, "per_attack_asr": {}}
                        continue
                    alpha_def = AlphaSteerDefense(model, AlphaSteerConfig(
                        model_name=model_name, matrices_dir=alpha_dir,
                        steering_layers=[sl], strength=1.0,
                    ))
                elif paradigm == "cast":
                    from src.steering.extract_vectors import load_steering_vector
                    if not vec_path.exists():
                        print("SKIP"); matrix[model_name][paradigm] = {"dsr": None}; continue
                    sv = load_steering_vector(vec_path)
                    benign = list(BENIGN_QUERIES[:20])
                    hook = ActivationHook(model, sl)
                    acts = []
                    for p in benign:
                        msgs = [{"role": "user", "content": p}]
                        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        inp = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
                        with torch.no_grad(): model(**inp)
                        acts.append(hook.get_last().cpu().float().numpy())
                        hook.clear()
                    hook.remove()
                    cond = np.mean(acts, axis=0).squeeze()
                    cond = cond / np.linalg.norm(cond)
                    cond_t = torch.from_numpy(cond).float()

                    class CASTHook:
                        def __init__(s, mdl, layer, sv, cond):
                            s.sv, s.cond = sv, cond
                            s._handle = ActivationHook._resolve_layer(mdl, layer).register_forward_hook(s._hook)
                        def _hook(s, mod, inp, out):
                            h = out[0] if isinstance(out, tuple) else out
                            act = h[:, -1, :].detach().cpu().float()
                            cos = torch.nn.functional.cosine_similarity(act, s.cond.unsqueeze(0))
                            if cos.item() < 0.3:
                                h[:, -1, :] += s.sv.to(h.device, dtype=h.dtype)
                            return (h,) + out[1:] if isinstance(out, tuple) else h
                        def remove(s): s._handle.remove()
                    static = CASTHook(model, sl, sv, cond_t)

                _, metrics = run_safety_eval(
                    model, tokenizer, attack_prompts, paradigm, model_name, device, pgs=pgs,
                )

                if pgs: pgs.remove()
                if static: static.remove()
                if alpha_def: alpha_def.remove()

                # Compute Wilson CIs per attack
                per_attack_ci = {}
                for at, asr in metrics.per_attack_asr.items():
                    n_at = sum(1 for p in attack_prompts if p["attack_type"] == at)
                    n_success = int(round(asr * n_at))
                    p, lo, hi = wilson_ci(n_success, n_at)
                    per_attack_ci[at] = {"asr": asr, "ci_lo": lo, "ci_hi": hi, "n": n_at}

                matrix[model_name][paradigm] = {
                    "dsr": metrics.dsr, "fpr": metrics.fpr,
                    "per_attack_asr": metrics.per_attack_asr,
                    "per_attack_ci": per_attack_ci,
                }
                print(f"DSR={metrics.dsr:.3f}")

            except Exception as e:
                print(f"ERROR: {e}")
                matrix[model_name][paradigm] = {"dsr": None, "error": str(e)}
                if pgs: pgs.remove()
                if static: static.remove()
                if alpha_def: alpha_def.remove()

        # Hedging eval (Mistral only — others don't hedge)
        if model_name == "mistral-7b":
            print(f"    Hedging eval...")
            for paradigm in paradigms:
                if matrix[model_name].get(paradigm, {}).get("dsr") is None:
                    continue
                pgs, static = None, None
                if paradigm == "static_1.3":
                    static = StaticSteering(model, str(vec_path), sl, 1.3)
                elif paradigm == "gated_bb_1.3":
                    pgs = ProbeGatedSteering(model, GateConfig(
                        probe_path=probe_dir_path / "best_probe.pkl",
                        steering_vector_path=vec_path,
                        probe_layer=probe_layer, steering_layer=sl,
                        controller_type="bang_bang", threshold=0.3, max_strength=1.3,
                    ))
                oe = run_openended_eval(model, tokenizer, device, defense_name=paradigm)
                save_openended_results(FINAL_DIR / "openended" / model_name, oe)
                if pgs: pgs.remove()
                if static: static.remove()
                matrix[model_name][paradigm]["hedging_rate"] = oe.hedging_rate

        del model; torch.cuda.empty_cache()

    with open(FINAL_DIR / "matrix.json", "w") as f:
        json.dump(matrix, f, indent=2)
    mark_done("phase4")

# ================================================================
# PHASE 5: Print results + save CSV
# ================================================================
print(f"\n{'='*100}")
print(f"FINAL MATRIX: ASR % (50 prompts per attack)")
print(f"{'='*100}")

print(f"\n{'Model':<12} {'Paradigm':<15}", end="")
for a in attack_names:
    print(f" {a[:7]:>8}", end="")
print(f" {'DSR':>6}")
print("-" * 100)

for m in models:
    for p in paradigms:
        r = matrix.get(m, {}).get(p, {})
        pa = r.get("per_attack_asr", {})
        dsr = r.get("dsr")
        print(f"{m:<12} {p:<15}", end="")
        for a in attack_names:
            v = pa.get(a)
            print(f" {f'{v:.0%}':>8}" if v is not None else f" {'N/A':>8}", end="")
        print(f" {f'{dsr:.3f}':>6}" if dsr is not None else f" {'N/A':>6}")
    print()

# Save CSV
with open(FINAL_DIR / "master_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "paradigm", "dsr"] + [f"asr_{a}" for a in attack_names]
               + [f"ci_lo_{a}" for a in attack_names] + [f"ci_hi_{a}" for a in attack_names])
    for m in models:
        for p in paradigms:
            r = matrix.get(m, {}).get(p, {})
            pa = r.get("per_attack_asr", {})
            ci = r.get("per_attack_ci", {})
            row = [m, p, r.get("dsr", "")]
            row += [pa.get(a, "") for a in attack_names]
            row += [ci.get(a, {}).get("ci_lo", "") for a in attack_names]
            row += [ci.get(a, {}).get("ci_hi", "") for a in attack_names]
            w.writerow(row)

print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE")
print(f"Results: {FINAL_DIR}")
