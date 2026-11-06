#!/usr/bin/env python3
"""Fix all A100 results: re-run with correct probe, proper GCG, full matrix.

Phase 1: Re-run ablation + expanded eval with local probe (layer 6)
Phase 2: GCG optimization (512 steps, 100 suffixes per model)
Phase 3: Full per-attack matrix for AlphaSteer + CAST
Phase 4: Generate master table

Usage: python3 scripts/gpu_fix_all.py
"""

import json
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure HF cache on workspace
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

from src.models.loader import load_model
from src.models.configs import MODEL_CONFIGS, get_config
from src.attacks.datasets.prompts import ADVBENCH_BEHAVIORS, BENIGN_QUERIES
from src.attacks.registry import load_attack_set
from src.baselines.static_steering.infer import StaticSteering
from src.baselines.alphasteer.wrapper import (
    AlphaSteerDefense, AlphaSteerConfig,
    compute_null_space_projection, compute_delta_matrix,
    save_alphasteer_matrices,
)
from src.eval.safety_eval import run_safety_eval, save_eval_results
from src.eval.openended_eval import run_openended_eval, save_openended_results
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.models.hooks import ActivationHook
from src.utils.paths import probe_results_dir, vector_results_dir, RESULTS_DIR, DATA_DIR
from src.utils.seeding import seed_everything

seed_everything(42)
device = "cuda"

# Check for local probe
LOCAL_PROBE = Path("results/probes/mistral-7b/jailbreak/best_probe_local.pkl")
LOCAL_VECTOR = Path("results/vectors/mistral-7b/refusal/vector_layer14_local.pt")

if not LOCAL_PROBE.exists():
    print("ERROR: Local probe not found. Copy from local machine first.")
    sys.exit(1)


def build_attack_set_2():
    """2-attack set (prefilling + intent laundering + benign)"""
    prompts = []
    for attack in ["prefilling", "intent_laundering"]:
        prompts.extend(load_attack_set(attack, n_attacks=20, include_benign=False))
    for i in range(min(20, len(BENIGN_QUERIES))):
        prompts.append({"prompt": BENIGN_QUERIES[i], "attack_type": "benign"})
    return prompts


def build_attack_set_5():
    """5-attack set (all attacks + benign)"""
    prompts = []
    for attack_name in ["prefilling", "intent_laundering", "deep_inception", "gcg", "autodan"]:
        try:
            attacks = load_attack_set(attack_name, n_attacks=20, include_benign=False)
            prompts.extend(attacks)
        except Exception as e:
            print(f"  Skipping {attack_name}: {e}")
    for i in range(min(20, len(BENIGN_QUERIES))):
        prompts.append({"prompt": BENIGN_QUERIES[i], "attack_type": "benign"})
    return prompts


def setup_defense(model, model_name, defense_type, steering_layer,
                  probe_path=None, vector_path=None, alpha_dir=None):
    """Set up a defense and return (pgs, static, alpha_def)"""
    pgs, static, alpha_def = None, None, None

    if defense_type == "static_1.3":
        static = StaticSteering(model, str(vector_path), steering_layer, 1.3)
    elif defense_type == "gated_bb_1.3":
        probe_dir = probe_results_dir(model_name, "jailbreak")
        with open(probe_dir / "best_layer.json") as f:
            pl = json.load(f)["layer"]
        pgs = ProbeGatedSteering(model, GateConfig(
            probe_path=probe_path or (probe_dir / "best_probe.pkl"),
            steering_vector_path=vector_path,
            probe_layer=pl if probe_path is None else 6,  # layer 6 for local probe
            steering_layer=steering_layer,
            controller_type="bang_bang", threshold=0.3, max_strength=1.3,
        ))
    elif defense_type == "alphasteer":
        alpha_def = AlphaSteerDefense(model, AlphaSteerConfig(
            model_name=model_name,
            matrices_dir=alpha_dir,
            steering_layers=[steering_layer],
            strength=1.0,
        ))
    elif defense_type == "cast":
        # Load CAST from saved results
        from src.steering.extract_vectors import load_steering_vector
        sv = load_steering_vector(vector_path)
        benign_prompts = list(BENIGN_QUERIES[:20])
        hook = ActivationHook(model, steering_layer)
        acts = []
        for p in benign_prompts:
            msgs = [{"role": "user", "content": p}]
            fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
            with torch.no_grad():
                model(**inp)
            acts.append(hook.get_last().cpu().float().numpy())
            hook.clear()
        hook.remove()
        cond = np.mean(acts, axis=0).squeeze()
        cond = cond / np.linalg.norm(cond)
        cond_t = torch.from_numpy(cond).float()

        class CASTHook:
            def __init__(self, mdl, layer, sv, cond, threshold=0.3, strength=1.0):
                self.sv, self.cond, self.threshold, self.strength = sv, cond, threshold, strength
                layer_mod = ActivationHook._resolve_layer(mdl, layer)
                self._handle = layer_mod.register_forward_hook(self._hook)
            def _hook(self, mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                act = h[:, -1, :].detach().cpu().float()
                cos = torch.nn.functional.cosine_similarity(act, self.cond.unsqueeze(0))
                if cos.item() < self.threshold:
                    s = self.sv.to(h.device, dtype=h.dtype)
                    h[:, -1, :] += self.strength * s
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h
            def remove(self):
                self._handle.remove()

        static = CASTHook(model, steering_layer, sv, cond_t, threshold=0.3, strength=1.0)

    return pgs, static, alpha_def


def cleanup_defense(pgs, static, alpha_def):
    if pgs: pgs.remove()
    if static: static.remove()
    if alpha_def: alpha_def.remove()


# ========================================================================
# PHASE 1: Re-run ablation + expanded eval with local probe
# ========================================================================
print("=" * 70)
print(f"[{time.strftime('%H:%M:%S')}] PHASE 1: Fix ablation + expanded eval (Mistral, local probe)")
print("=" * 70)

model_name = "mistral-7b"
model, tokenizer, config = load_model(model_name, device=device)
steering_layer = config.optimal_steering_layer
vector_path = vector_results_dir(model_name, "refusal") / f"vector_layer{steering_layer}.pt"

# 1a: Ablation with local probe
print(f"\n[{time.strftime('%H:%M:%S')}] Phase 1a: Ablation (corrected)")
attack_prompts_2 = build_attack_set_2()
ablation_dir = RESULTS_DIR / "ablation_corrected" / model_name
ablation_dir.mkdir(parents=True, exist_ok=True)

ablation_configs = {
    "none": {"type": "none"},
    "probe_only": {"type": "gated_bb_1.3", "strength_override": 0.0},
    "steer_only": {"type": "static_1.3"},
    "probe_steer": {"type": "gated_bb_1.3"},
}

ablation_results = {}
for name, cfg in ablation_configs.items():
    print(f"  [{name}]")
    pgs, static, alpha_def = None, None, None

    if cfg["type"] == "none":
        pass
    elif cfg["type"] == "static_1.3":
        static = StaticSteering(model, str(LOCAL_VECTOR), steering_layer, 1.3)
    elif cfg["type"] == "gated_bb_1.3":
        strength = cfg.get("strength_override", 1.3)
        pgs = ProbeGatedSteering(model, GateConfig(
            probe_path=LOCAL_PROBE,
            steering_vector_path=LOCAL_VECTOR,
            probe_layer=6, steering_layer=steering_layer,
            controller_type="bang_bang", threshold=0.3, max_strength=strength,
        ))

    _, metrics = run_safety_eval(model, tokenizer, attack_prompts_2, name, model_name, device, pgs=pgs)
    oe = run_openended_eval(model, tokenizer, device, defense_name=name)
    save_openended_results(ablation_dir / "openended", oe)
    cleanup_defense(pgs, static, alpha_def)

    ablation_results[name] = {"dsr": metrics.dsr, "hedging_rate": oe.hedging_rate, "fpr": metrics.fpr}
    print(f"    DSR={metrics.dsr:.3f} Hedge={oe.hedging_rate:.0%}")

with open(ablation_dir / "ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)
print(f"\n  Ablation saved to {ablation_dir}")

# 1b: Expanded eval with local probe
print(f"\n[{time.strftime('%H:%M:%S')}] Phase 1b: Expanded eval (5 attacks, corrected)")
attack_prompts_5 = build_attack_set_5()
expanded_dir = RESULTS_DIR / "expanded_eval_corrected" / model_name
expanded_dir.mkdir(parents=True, exist_ok=True)

expanded_results = {}
for defense_name in ["none", "static_1.3", "gated_bb_1.3"]:
    print(f"  [{defense_name}]")
    pgs, static, alpha_def = None, None, None
    if defense_name == "static_1.3":
        static = StaticSteering(model, str(LOCAL_VECTOR), steering_layer, 1.3)
    elif defense_name == "gated_bb_1.3":
        pgs = ProbeGatedSteering(model, GateConfig(
            probe_path=LOCAL_PROBE,
            steering_vector_path=LOCAL_VECTOR,
            probe_layer=6, steering_layer=steering_layer,
            controller_type="bang_bang", threshold=0.3, max_strength=1.3,
        ))

    _, metrics = run_safety_eval(model, tokenizer, attack_prompts_5, defense_name, model_name, device, pgs=pgs)
    save_eval_results(expanded_dir, [], metrics, defense_name, model_name)
    cleanup_defense(pgs, static, alpha_def)

    expanded_results[defense_name] = {"dsr": metrics.dsr, "per_attack_asr": metrics.per_attack_asr}
    print(f"    DSR={metrics.dsr:.3f}")
    for at, asr in sorted(metrics.per_attack_asr.items()):
        print(f"      {at}: ASR={asr:.0%}")

with open(expanded_dir / "expanded_results.json", "w") as f:
    json.dump(expanded_results, f, indent=2)

del model
torch.cuda.empty_cache()

# ========================================================================
# PHASE 2: GCG optimization (proper)
# ========================================================================
print("\n" + "=" * 70)
print(f"[{time.strftime('%H:%M:%S')}] PHASE 2: Proper GCG optimization")
print("=" * 70)

try:
    import nanogcg
    HAS_NANOGCG = True
except ImportError:
    print("  nanogcg not available — installing...")
    os.system("pip install nanogcg --quiet")
    try:
        import nanogcg
        HAS_NANOGCG = True
    except ImportError:
        HAS_NANOGCG = False
        print("  WARNING: nanogcg install failed. Skipping GCG phase.")

if HAS_NANOGCG:
    for gcg_model in ["mistral-7b", "llama31-8b"]:
        print(f"\n  [{time.strftime('%H:%M:%S')}] GCG for {gcg_model}")
        gcg_config = get_config(gcg_model)
        model, tokenizer, _ = load_model(gcg_model, device=device)

        behaviors = ADVBENCH_BEHAVIORS[:100]
        out_dir = DATA_DIR / "attacks" / "gcg"
        out_dir.mkdir(parents=True, exist_ok=True)

        suffixes = []
        losses = []
        for i, behavior in enumerate(behaviors):
            print(f"    [{i+1}/{len(behaviors)}] {behavior[:50]}...")
            try:
                result = nanogcg.run(
                    model, tokenizer, behavior,
                    num_steps=512, batch_size=256,
                )
                suffixes.append(result.best_string)
                losses.append(float(result.best_loss))
                print(f"      Loss: {result.best_loss:.4f}")
            except Exception as e:
                print(f"      ERROR: {e}")
                suffixes.append("")
                losses.append(999.0)

            # Checkpoint every 10
            if (i + 1) % 10 == 0:
                with open(out_dir / f"{gcg_model}_suffixes.json", "w") as f:
                    json.dump(suffixes, f, indent=2)
                with open(out_dir / f"{gcg_model}_losses.json", "w") as f:
                    json.dump(losses, f, indent=2)
                avg_loss = np.mean([l for l in losses if l < 999])
                print(f"      Checkpoint: {len(suffixes)} suffixes, avg loss={avg_loss:.4f}")

        with open(out_dir / f"{gcg_model}_suffixes.json", "w") as f:
            json.dump(suffixes, f, indent=2)
        with open(out_dir / f"{gcg_model}_losses.json", "w") as f:
            json.dump(losses, f, indent=2)

        valid = sum(1 for l in losses if l < 5.0)
        print(f"    {gcg_model}: {valid}/{len(suffixes)} valid suffixes")

        del model
        torch.cuda.empty_cache()

# ========================================================================
# PHASE 3: Full per-attack matrix for ALL paradigms × ALL models
# ========================================================================
print("\n" + "=" * 70)
print(f"[{time.strftime('%H:%M:%S')}] PHASE 3: Full paradigm × attack × model matrix")
print("=" * 70)

matrix_results = {}

for model_name in ["mistral-7b", "gemma2-9b", "qwen25-7b", "llama31-8b"]:
    print(f"\n  [{time.strftime('%H:%M:%S')}] Model: {model_name}")
    model_config = get_config(model_name)
    model, tokenizer, model_config = load_model(model_name, device=device)
    sl = model_config.optimal_steering_layer

    vec_dir = vector_results_dir(model_name, "refusal")
    vec_path = vec_dir / f"vector_layer{sl}.pt"
    probe_dir = probe_results_dir(model_name, "jailbreak")
    alpha_dir = RESULTS_DIR / "alphasteer" / model_name

    # Use local probe for Mistral, A100 probe for others
    if model_name == "mistral-7b":
        use_probe = LOCAL_PROBE
        use_vector = LOCAL_VECTOR
        probe_layer = 6
    else:
        use_probe = probe_dir / "best_probe.pkl"
        use_vector = vec_path
        with open(probe_dir / "best_layer.json") as f:
            probe_layer = json.load(f)["layer"]

    attack_prompts = build_attack_set_5()
    n_atk = sum(1 for p in attack_prompts if p["attack_type"] != "benign")
    print(f"    {n_atk} attack + {len(attack_prompts)-n_atk} benign prompts")

    matrix_results[model_name] = {}

    paradigms = ["none", "static_1.3", "gated_bb_1.3", "alphasteer", "cast"]

    for paradigm in paradigms:
        print(f"    [{paradigm}]...", end=" ", flush=True)
        pgs, static, alpha_def = None, None, None

        try:
            if paradigm == "none":
                pass
            elif paradigm == "static_1.3":
                static = StaticSteering(model, str(use_vector), sl, 1.3)
            elif paradigm == "gated_bb_1.3":
                pgs = ProbeGatedSteering(model, GateConfig(
                    probe_path=use_probe, steering_vector_path=use_vector,
                    probe_layer=probe_layer, steering_layer=sl,
                    controller_type="bang_bang", threshold=0.3, max_strength=1.3,
                ))
            elif paradigm == "alphasteer":
                if not (alpha_dir / f"delta_layer{sl}.pt").exists():
                    # Need to train AlphaSteer for this model
                    if model_name == "mistral-7b":
                        # Use corrected Q1 matrices
                        alpha_dir = RESULTS_DIR / "alphasteer_corrected" / model_name / "matrices"
                        if not alpha_dir.exists():
                            alpha_dir = RESULTS_DIR / "alphasteer" / model_name
                    if not (alpha_dir / f"delta_layer{sl}.pt").exists():
                        print(f"SKIP (no matrices)")
                        matrix_results[model_name][paradigm] = {"dsr": None, "per_attack_asr": {}}
                        continue
                alpha_def = AlphaSteerDefense(model, AlphaSteerConfig(
                    model_name=model_name, matrices_dir=alpha_dir,
                    steering_layers=[sl], strength=1.0,
                ))
            elif paradigm == "cast":
                from src.steering.extract_vectors import load_steering_vector
                if not vec_path.exists():
                    print(f"SKIP (no vector)")
                    matrix_results[model_name][paradigm] = {"dsr": None, "per_attack_asr": {}}
                    continue
                sv = load_steering_vector(vec_path)
                benign = list(BENIGN_QUERIES[:20])
                hook = ActivationHook(model, sl)
                acts = []
                for p in benign:
                    msgs = [{"role": "user", "content": p}]
                    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    inp = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
                    with torch.no_grad():
                        model(**inp)
                    acts.append(hook.get_last().cpu().float().numpy())
                    hook.clear()
                hook.remove()
                cond = np.mean(acts, axis=0).squeeze()
                cond = cond / np.linalg.norm(cond)
                cond_t = torch.from_numpy(cond).float()

                class CASTHookInner:
                    def __init__(self, mdl, layer, sv, cond):
                        self.sv, self.cond = sv, cond
                        layer_mod = ActivationHook._resolve_layer(mdl, layer)
                        self._handle = layer_mod.register_forward_hook(self._hook)
                    def _hook(self, mod, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        act = h[:, -1, :].detach().cpu().float()
                        cos = torch.nn.functional.cosine_similarity(act, self.cond.unsqueeze(0))
                        if cos.item() < 0.3:
                            s = self.sv.to(h.device, dtype=h.dtype)
                            h[:, -1, :] += 1.0 * s
                        if isinstance(out, tuple):
                            return (h,) + out[1:]
                        return h
                    def remove(self):
                        self._handle.remove()

                static = CASTHookInner(model, sl, sv, cond_t)

            _, metrics = run_safety_eval(
                model, tokenizer, attack_prompts, paradigm, model_name, device, pgs=pgs,
            )
            cleanup_defense(pgs, static, alpha_def)

            matrix_results[model_name][paradigm] = {
                "dsr": metrics.dsr,
                "per_attack_asr": metrics.per_attack_asr,
                "fpr": metrics.fpr,
            }
            print(f"DSR={metrics.dsr:.3f}")

        except Exception as e:
            print(f"ERROR: {e}")
            matrix_results[model_name][paradigm] = {"dsr": None, "per_attack_asr": {}, "error": str(e)}
            cleanup_defense(pgs, static, alpha_def)

    # Hedging for each paradigm
    print(f"    Running hedging eval...")
    for paradigm in paradigms:
        if matrix_results[model_name].get(paradigm, {}).get("dsr") is None:
            matrix_results[model_name][paradigm]["hedging_rate"] = None
            continue

        pgs, static, alpha_def = None, None, None
        try:
            if paradigm == "static_1.3":
                static = StaticSteering(model, str(use_vector), sl, 1.3)
            elif paradigm == "gated_bb_1.3":
                pgs = ProbeGatedSteering(model, GateConfig(
                    probe_path=use_probe, steering_vector_path=use_vector,
                    probe_layer=probe_layer, steering_layer=sl,
                    controller_type="bang_bang", threshold=0.3, max_strength=1.3,
                ))

            oe = run_openended_eval(model, tokenizer, device, defense_name=paradigm)
            cleanup_defense(pgs, static, alpha_def)
            matrix_results[model_name][paradigm]["hedging_rate"] = oe.hedging_rate
        except Exception as e:
            cleanup_defense(pgs, static, alpha_def)
            matrix_results[model_name][paradigm]["hedging_rate"] = None

    del model
    torch.cuda.empty_cache()

# ========================================================================
# PHASE 4: Generate master table
# ========================================================================
print("\n" + "=" * 70)
print(f"[{time.strftime('%H:%M:%S')}] PHASE 4: Master table")
print("=" * 70)

final_dir = RESULTS_DIR / "final"
final_dir.mkdir(parents=True, exist_ok=True)

# Save corrected ablation
with open(final_dir / "ablation_corrected.json", "w") as f:
    json.dump(ablation_results, f, indent=2)

# Save full matrix
with open(final_dir / "master_table.json", "w") as f:
    json.dump(matrix_results, f, indent=2)

# Generate CSV
attacks = ["gcg", "autodan", "deep_inception", "prefilling", "intent_laundering"]
models = ["mistral-7b", "gemma2-9b", "qwen25-7b", "llama31-8b"]
paradigms_list = ["none", "static_1.3", "gated_bb_1.3", "alphasteer", "cast"]

csv_lines = ["model,paradigm,dsr,hedging_rate,fpr," + ",".join(f"asr_{a}" for a in attacks)]
for model_name in models:
    for paradigm in paradigms_list:
        r = matrix_results.get(model_name, {}).get(paradigm, {})
        dsr = r.get("dsr", "")
        hedge = r.get("hedging_rate", "")
        fpr = r.get("fpr", "")
        pa = r.get("per_attack_asr", {})
        attack_vals = [str(pa.get(a, "")) for a in attacks]
        csv_lines.append(f"{model_name},{paradigm},{dsr},{hedge},{fpr},{','.join(attack_vals)}")

with open(final_dir / "master_table.csv", "w") as f:
    f.write("\n".join(csv_lines))

# Print the table
print("\n" + "=" * 100)
print("MASTER TABLE: 5 Paradigms × 5 Attacks × 4 Models")
print("=" * 100)

print(f"\n{'Model':<12} {'Paradigm':<15} {'DSR':>5} {'Hedge':>6} "
      + " ".join(f"{a[:7]:>7}" for a in attacks))
print("-" * 90)

for model_name in models:
    for paradigm in paradigms_list:
        r = matrix_results.get(model_name, {}).get(paradigm, {})
        dsr = r.get("dsr")
        hedge = r.get("hedging_rate")
        pa = r.get("per_attack_asr", {})

        dsr_s = f"{dsr:.3f}" if dsr is not None else "  N/A"
        hedge_s = f"{hedge:.0%}" if hedge is not None else " N/A"
        attack_s = []
        for a in attacks:
            v = pa.get(a)
            attack_s.append(f"{v:.0%}" if v is not None else "  N/A")

        print(f"{model_name:<12} {paradigm:<15} {dsr_s:>5} {hedge_s:>6} " + " ".join(f"{s:>7}" for s in attack_s))
    print()

# Ablation
print("\nABLATION (Mistral-7B, corrected):")
print(f"{'Condition':<15} {'DSR':>6} {'Hedge':>7}")
for name, r in ablation_results.items():
    print(f"{name:<15} {r['dsr']:>6.3f} {r['hedging_rate']:>6.0%}")

print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE. Results in {final_dir}")
