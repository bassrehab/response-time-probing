#!/usr/bin/env python3
"""Eval pipeline scale-up: P0 items first, then full n=200 re-eval.

Phase A: Train response probes on Gemma-2-9B and Qwen-2.5-7B (P0 #2)
Phase B: Asymmetry analysis — probe weights vs refusal vector (P0 #3)
Phase C: Template robustness — retrain with 15 diverse templates (P2 #4)
Phase D: Full n=200 re-eval across all paradigms × models (P1 #1)
         (waits for GCG suffixes from Pod 1)

Usage: python scripts/gpu_eval_scaleup.py [--phase A|B|C|D|all]
"""

import json, os, pickle, sys, time, argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
# Pass HF token from default cache to workspace cache
_token_path = Path.home() / ".cache" / "huggingface" / "token"
if _token_path.exists():
    os.environ["HF_TOKEN"] = _token_path.read_text().strip()

sys.path.insert(0, ".")
from src.models.loader import load_model
from src.models.configs import get_config
from src.eval.judge import judge_response
from src.eval.safety_eval import generate_response
from src.gating.probe_gate import GateConfig, ProbeGatedSteering
from src.attacks.prefilling import generate_prefilling_attacks
from src.attacks.datasets.prompts import ADVBENCH_BEHAVIORS, BENIGN_QUERIES, BENIGN_OPENENDED
from src.utils.paths import RESULTS_DIR, probe_results_dir, vector_results_dir
from src.utils.seeding import seed_everything

seed_everything(42)
device = "cuda"
N_RESP_TOKENS = 5
RESP_THRESHOLD = 0.5
REFUSAL_TEXT = "I cannot help with that request. This appears to be a harmful query."

OUT = RESULTS_DIR / "scaleup"
OUT.mkdir(parents=True, exist_ok=True)


def check_done(name):
    return (OUT / f".{name}_done").exists()


def mark_done(name):
    (OUT / f".{name}_done").touch()


def generate_with_hidden(model, tokenizer, prompt, max_new_tokens=10, prefill=None):
    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
            return_dict_in_generate=True, output_hidden_states=True,
        )
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = out.sequences[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if prefill:
        text = prefill + text
    per_token = []
    for ls in out.hidden_states:
        d = {i: s[0, -1, :].detach().cpu().float().numpy() for i, s in enumerate(ls)}
        per_token.append(d)
    return text, per_token


def train_response_probe(all_acts, labels, num_layers):
    results = {}
    best_layer, best_auroc, best_probe, best_scaler = -1, -1, None, None
    for layer in range(num_layers):
        X = np.stack(all_acts[layer])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aurocs = []
        for tr, va in skf.split(X, labels):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xva = sc.transform(X[va])
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, C=1.0)
            clf.fit(Xtr, labels[tr])
            try:
                a = roc_auc_score(labels[va], clf.predict_proba(Xva)[:, 1])
            except ValueError:
                a = 0.5
            fold_aurocs.append(a)
        mean_a = float(np.mean(fold_aurocs))
        results[layer] = {"auroc": mean_a, "std": float(np.std(fold_aurocs))}
        if mean_a > best_auroc:
            best_auroc = mean_a
            best_layer = layer
    # Retrain best on all data
    X = np.stack(all_acts[best_layer])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, C=1.0)
    clf.fit(Xs, labels)
    return results, best_layer, best_auroc, clf, sc


def collect_activations(model, tokenizer, samples, label_type):
    num_layers = model.config.num_hidden_layers + 1
    all_acts = {l: [] for l in range(num_layers)}
    labels = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        prefill = s.get("prefill")
        try:
            resp, per_token = generate_with_hidden(model, tokenizer, prompt, N_RESP_TOKENS, prefill)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            continue
        is_attack = label_type == "attack"
        judge = judge_response(resp, is_attack=is_attack)
        label = 1 if judge.is_harmful else 0
        n_avail = min(N_RESP_TOKENS, len(per_token))
        if n_avail == 0:
            continue
        for li in range(num_layers):
            la = np.stack([per_token[t][li] for t in range(n_avail)])
            all_acts[li].append(la.mean(axis=0))
        labels.append(label)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(samples)}")
    return all_acts, np.array(labels)


# ================================================================
# PHASE A: Train response probes on Gemma and Qwen
# ================================================================
def phase_a():
    if check_done("phase_a"):
        print("SKIP Phase A — already done")
        return
    print(f"\n[{time.strftime('%H:%M:%S')}] PHASE A: Response probes for Gemma + Qwen")

    attacks = generate_prefilling_attacks(n_attacks=40)
    benign_all = list(BENIGN_QUERIES) + list(BENIGN_OPENENDED[:10])
    benign = [{"prompt": q} for q in benign_all[:50]]

    for model_name in ["gemma2-9b", "qwen25-7b"]:
        print(f"\n  [{time.strftime('%H:%M:%S')}] {model_name}")
        model, tokenizer, _ = load_model(model_name, device=device)

        print(f"    Collecting attack activations...")
        a_acts, a_labels = collect_activations(model, tokenizer, attacks, "attack")
        print(f"    Collecting benign activations...")
        b_acts, b_labels = collect_activations(model, tokenizer, benign, "benign")

        # Merge
        num_layers = model.config.num_hidden_layers + 1
        all_acts = {l: a_acts[l] + b_acts[l] for l in range(num_layers)}
        labels = np.concatenate([a_labels, b_labels])
        print(f"    Samples: {len(labels)} ({labels.sum()} positive, {(1-labels).sum()} negative)")

        if labels.sum() < 3:
            print(f"    SKIP: too few positives")
            del model; torch.cuda.empty_cache()
            continue

        sweep, best_layer, best_auroc, probe, scaler = train_response_probe(all_acts, labels, num_layers)

        # Save
        pkl_path = OUT / f"{model_name}_response_probe.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"probe": probe, "scaler": scaler}, f)
        json.dump({
            "best_layer": best_layer, "best_auroc": best_auroc,
            "n_samples": len(labels), "n_positive": int(labels.sum()),
            "layer_sweep": sweep,
        }, open(OUT / f"{model_name}_response_probe_meta.json", "w"), indent=2)
        print(f"    Best layer: {best_layer}, AUROC: {best_auroc:.4f}")

        # Run dual-probe intervention test (halt mode on prefilling)
        print(f"    Testing halt intervention on 40 prefilling attacks...")
        with open(pkl_path, "rb") as f:
            rp_data = pickle.load(f)
        r_probe = rp_data["probe"]
        r_scaler = rp_data["scaler"]

        halt_results = []
        for j, att in enumerate(attacks):
            prompt = att["prompt"]
            prefill = att.get("prefill")
            try:
                _, per_token = generate_with_hidden(model, tokenizer, prompt, N_RESP_TOKENS, prefill)
            except:
                halt_results.append({"halted": False, "is_harmful": True})
                continue
            n_av = min(N_RESP_TOKENS, len(per_token))
            la = np.stack([per_token[t][best_layer] for t in range(n_av)])
            ma = la.mean(axis=0).reshape(1, -1)
            score = float(r_probe.predict_proba(r_scaler.transform(ma))[0, 1])
            fired = score > RESP_THRESHOLD
            if fired:
                halt_results.append({"halted": True, "is_harmful": False, "score": score})
            else:
                resp, _ = generate_response(model, tokenizer, prompt, device, 200, prefill=prefill)
                harmful = judge_response(resp, is_attack=True).is_harmful
                halt_results.append({"halted": False, "is_harmful": bool(harmful), "score": score})

        n_halt = sum(r["halted"] for r in halt_results)
        n_harm = sum(r["is_harmful"] for r in halt_results)
        asr = n_harm / len(halt_results)
        print(f"    Halt rate: {n_halt}/{len(halt_results)}, ASR after halt: {asr:.2%}")

        json.dump({
            "halt_rate": n_halt / len(halt_results), "asr": asr,
            "n_halted": n_halt, "n_harmful": n_harm, "n_total": len(halt_results),
        }, open(OUT / f"{model_name}_halt_test.json", "w"), indent=2)

        del model; torch.cuda.empty_cache()

    mark_done("phase_a")
    print(f"[{time.strftime('%H:%M:%S')}] Phase A done")


# ================================================================
# PHASE B: Asymmetry analysis (probe weights vs refusal vector)
# ================================================================
def phase_b():
    if check_done("phase_b"):
        print("SKIP Phase B — already done")
        return
    print(f"\n[{time.strftime('%H:%M:%S')}] PHASE B: Asymmetry analysis")

    analysis = {}

    for model_name in ["mistral-7b", "llama31-8b", "gemma2-9b", "qwen25-7b"]:
        print(f"\n  [{model_name}]")

        # Load response probe (Mistral/Llama from previous run, Gemma/Qwen from Phase A)
        if model_name in ("mistral-7b", "llama31-8b"):
            rp_dir = RESULTS_DIR / "response_probe_full"
            prefix = "mistral" if "mistral" in model_name else "llama"
            pkl = rp_dir / f"{prefix}_response_probe.pkl"
            meta_file = "phase1_probe.json" if "mistral" in model_name else "phase2_probe.json"
            meta = json.load(open(rp_dir / meta_file))
        else:
            pkl = OUT / f"{model_name}_response_probe.pkl"
            meta = json.load(open(OUT / f"{model_name}_response_probe_meta.json"))

        if not pkl.exists():
            print(f"    SKIP: no probe found at {pkl}")
            continue

        with open(pkl, "rb") as f:
            rp_data = pickle.load(f)
        probe_weights = rp_data["probe"].coef_[0]  # shape (hidden_dim,)
        best_layer = meta["best_layer"]
        best_auroc = meta["best_auroc"]

        # Load refusal steering vector at steering layer
        cfg = get_config(model_name)
        sl = cfg.optimal_steering_layer
        vec_path = vector_results_dir(model_name, "refusal") / f"vector_layer{sl}.pt"
        if vec_path.exists():
            refusal_vec = torch.load(vec_path, weights_only=True).cpu().float().numpy()
        else:
            print(f"    SKIP: no refusal vector at {vec_path}")
            refusal_vec = None

        # Also load refusal vector at PROBE layer if different from steering layer
        vec_path_probe = vector_results_dir(model_name, "refusal") / f"vector_layer{best_layer}.pt"
        refusal_vec_probe_layer = None
        if vec_path_probe.exists():
            refusal_vec_probe_layer = torch.load(vec_path_probe, weights_only=True).cpu().float().numpy()

        # Analysis 1: Cosine similarity between probe weights and refusal vector
        cos_steer = None
        cos_probe_layer = None
        if refusal_vec is not None and len(probe_weights) == len(refusal_vec):
            cos_steer = float(np.dot(probe_weights, refusal_vec) /
                              (np.linalg.norm(probe_weights) * np.linalg.norm(refusal_vec) + 1e-10))
        if refusal_vec_probe_layer is not None and len(probe_weights) == len(refusal_vec_probe_layer):
            cos_probe_layer = float(np.dot(probe_weights, refusal_vec_probe_layer) /
                                    (np.linalg.norm(probe_weights) * np.linalg.norm(refusal_vec_probe_layer) + 1e-10))

        # Analysis 2: Top-k probe dimensions
        top_k = 20
        top_dims = np.argsort(np.abs(probe_weights))[-top_k:][::-1]
        top_weights = [(int(d), float(probe_weights[d])) for d in top_dims]

        # Analysis 3: Weight vector statistics
        w_norm = float(np.linalg.norm(probe_weights))
        w_sparsity = float((np.abs(probe_weights) < 0.01 * w_norm).mean())

        result = {
            "model": model_name,
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "probe_dim": len(probe_weights),
            "cos_sim_refusal_steer_layer": cos_steer,
            "cos_sim_refusal_probe_layer": cos_probe_layer,
            "steering_layer": sl,
            "weight_norm": w_norm,
            "weight_sparsity_01pct": w_sparsity,
            "top_20_dims": top_weights,
        }
        analysis[model_name] = result
        print(f"    Layer {best_layer}, AUROC {best_auroc:.4f}")
        print(f"    cos(probe, refusal@steer_layer): {cos_steer}")
        print(f"    cos(probe, refusal@probe_layer): {cos_probe_layer}")
        print(f"    Weight norm: {w_norm:.4f}, sparsity: {w_sparsity:.2%}")

    # Cross-layer probe swap: train Llama@layer11 and Mistral@layer1
    print(f"\n  Cross-layer probe swap...")
    for model_name, swap_layer in [("mistral-7b", 1), ("llama31-8b", 11)]:
        print(f"    {model_name} @ layer {swap_layer}")
        model, tokenizer, _ = load_model(model_name, device=device)
        attacks = generate_prefilling_attacks(n_attacks=40)
        benign = [{"prompt": q} for q in list(BENIGN_QUERIES)[:50]]

        a_acts, a_labels = collect_activations(model, tokenizer, attacks, "attack")
        b_acts, b_labels = collect_activations(model, tokenizer, benign, "benign")

        num_layers = model.config.num_hidden_layers + 1
        merged = {l: a_acts[l] + b_acts[l] for l in range(num_layers)}
        labels = np.concatenate([a_labels, b_labels])

        if swap_layer < num_layers and labels.sum() >= 3:
            X = np.stack(merged[swap_layer])
            sc = StandardScaler()
            Xs = sc.fit_transform(X)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aurocs = []
            for tr, va in skf.split(Xs, labels):
                clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
                clf.fit(Xs[tr], labels[tr])
                try:
                    aurocs.append(roc_auc_score(labels[va], clf.predict_proba(Xs[va])[:, 1]))
                except:
                    aurocs.append(0.5)
            swap_auroc = float(np.mean(aurocs))
            analysis[f"{model_name}_swap_layer{swap_layer}"] = {
                "model": model_name, "swap_layer": swap_layer,
                "auroc": swap_auroc, "std": float(np.std(aurocs)),
            }
            print(f"    AUROC at layer {swap_layer}: {swap_auroc:.4f}")

        del model; torch.cuda.empty_cache()

    # PCA visualization of activations
    print(f"\n  PCA visualization...")
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model_name in ["mistral-7b", "llama31-8b"]:
        if model_name in ("mistral-7b",):
            rp_dir = RESULTS_DIR / "response_probe_full"
            meta = json.load(open(rp_dir / "phase1_probe.json"))
        else:
            rp_dir = RESULTS_DIR / "response_probe_full"
            meta = json.load(open(rp_dir / "phase2_probe.json"))
        best_layer = meta["best_layer"]

        model, tokenizer, _ = load_model(model_name, device=device)
        from src.attacks.deep_inception import generate_deep_inception_attacks
        from src.attacks.registry import load_attack_set

        sample_groups = {
            "prefilling": generate_prefilling_attacks(n_attacks=30),
            "autodan": load_attack_set("autodan", n_attacks=20, include_benign=False),
            "benign": [{"prompt": q} for q in list(BENIGN_QUERIES)[:30]],
        }

        pca_data = {}
        for group_name, samples in sample_groups.items():
            acts_all, _ = collect_activations(model, tokenizer, samples,
                                              "attack" if group_name != "benign" else "benign")
            if acts_all[best_layer]:
                pca_data[group_name] = np.stack(acts_all[best_layer])

        del model; torch.cuda.empty_cache()

        if len(pca_data) >= 2:
            all_X = np.vstack(list(pca_data.values()))
            pca = PCA(n_components=2)
            all_2d = pca.fit_transform(all_X)
            colors = {"prefilling": "#d62728", "autodan": "#1f77b4", "benign": "#2ca02c"}
            fig, ax = plt.subplots(figsize=(4, 3.5))
            offset = 0
            for gn, X in pca_data.items():
                n = len(X)
                ax.scatter(all_2d[offset:offset+n, 0], all_2d[offset:offset+n, 1],
                           s=25, alpha=0.7, label=gn, c=colors.get(gn, "gray"))
                offset += n
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title(f"{model_name} response activations (layer {best_layer})")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(OUT / f"{model_name}_pca_layer{best_layer}.pdf")
            plt.savefig(OUT / f"{model_name}_pca_layer{best_layer}.png", dpi=150)
            plt.close()
            print(f"    Saved PCA for {model_name}")

    json.dump(analysis, open(OUT / "asymmetry_analysis.json", "w"), indent=2)
    mark_done("phase_b")
    print(f"[{time.strftime('%H:%M:%S')}] Phase B done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="AB", help="Which phases to run: A, B, C, D, AB, all")
    args = parser.parse_args()

    phases = args.phase.upper()
    if "A" in phases or phases == "ALL":
        phase_a()
    if "B" in phases or phases == "ALL":
        phase_b()
    # Phase C (template robustness) and D (n=200 re-eval) will be added after A+B complete


if __name__ == "__main__":
    main()
