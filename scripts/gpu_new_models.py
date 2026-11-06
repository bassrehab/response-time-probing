#!/usr/bin/env python3
"""Full pipeline for new models: Qwen3-8B and Gemma-3-12B-it.

For each model:
  1. Train prompt-time jailbreak probe (contrast pairs → layer sweep)
  2. Extract refusal steering vector (CAA)
  3. Train response-time probe (prefilling + benign activations)
  4. Test halt intervention on prefilling attacks
  5. Asymmetry analysis (probe weights vs refusal vector)

Usage: python scripts/gpu_new_models.py [--model qwen3-8b|gemma3-12b|both]
"""

import json, os, pickle, sys, time, argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
_token_path = Path.home() / ".cache" / "huggingface" / "token"
if _token_path.exists():
    os.environ["HF_TOKEN"] = _token_path.read_text().strip()

sys.path.insert(0, ".")
from src.attacks.prefilling import generate_prefilling_attacks
from src.attacks.datasets.prompts import BENIGN_QUERIES, BENIGN_OPENENDED
from src.eval.judge import judge_response
from src.utils.seeding import seed_everything

seed_everything(42)
device = "cuda"
N_RESP_TOKENS = 5
RESP_THRESHOLD = 0.5

OUT = Path("results/new_models")
OUT.mkdir(parents=True, exist_ok=True)

NEW_MODELS = {
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "num_layers": 36,
        "hidden_dim": 4096,
        "steering_layer": 18,
    },
    "gemma3-12b": {
        "hf_id": "google/gemma-3-12b-it",
        "num_layers": 48,
        "hidden_dim": 3840,
        "steering_layer": 24,
    },
    "gemma4-31b": {
        "hf_id": "google/gemma-4-31b-it",
        "num_layers": 62,  # will be auto-detected
        "hidden_dim": 4608,  # will be auto-detected
        "steering_layer": 31,
    },
}


def check_done(name):
    return (OUT / f".{name}_done").exists()


def mark_done(name):
    (OUT / f".{name}_done").touch()


def load_new_model(model_name):
    """Load model handling architecture differences."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = NEW_MODELS[model_name]
    print(f"  Loading {cfg['hf_id']}...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get actual layer count from model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        actual_layers = len(model.model.layers)
    elif hasattr(model.config, 'text_config'):
        actual_layers = model.config.text_config.num_hidden_layers
    else:
        actual_layers = model.config.num_hidden_layers
    cfg["num_layers"] = actual_layers
    print(f"  Loaded in {time.time()-t0:.0f}s, {actual_layers} layers")
    return model, tokenizer, cfg


def get_layer(model, idx):
    """Get transformer layer by index, handling different architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[idx]
    raise ValueError(f"Cannot find layers for {type(model)}")


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
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
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


def extract_activations(model, tokenizer, prompts, layer):
    """Extract last-token activations at a given layer for a list of prompts."""
    from src.models.hooks import ActivationHook
    hook = ActivationHook(model, layer)
    acts = []
    for i, p in enumerate(prompts):
        prompt = p if isinstance(p, str) else p.get("prompt", p)
        messages = [{"role": "user", "content": prompt}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            model(**inputs)
        act = hook.get_last().cpu().float().numpy()
        acts.append(act)
        hook.clear()
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(prompts)}")
    hook.remove()
    return np.stack(acts).squeeze()


def run_pipeline(model_name):
    """Full pipeline for a single new model."""
    model_out = OUT / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    if check_done(f"{model_name}_all"):
        print(f"SKIP {model_name} — already done")
        return

    print(f"\n{'='*60}")
    print(f"PIPELINE: {model_name}")
    print(f"{'='*60}")

    model, tokenizer, cfg = load_new_model(model_name)
    sl = cfg["steering_layer"]
    num_layers = cfg["num_layers"]

    # ---- Step 1: Train prompt-time jailbreak probe ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 1: Prompt-time jailbreak probe")

    # Load contrast pairs (format: {"prompt": str, "label": 0|1})
    contrast_file = Path("data/contrast_pairs/jailbreak_pairs.json")
    pairs = json.load(open(contrast_file))
    harmful_prompts = [p["prompt"] for p in pairs if p["label"] == 1]
    benign_prompts = [p["prompt"] for p in pairs if p["label"] == 0]

    # Sweep layers for best probe
    print(f"    Extracting activations for probe sweep...")
    best_auroc = -1
    best_layer = -1
    best_probe_data = None
    probe_sweep = {}

    # Sweep a subset of layers (every 2nd to save time, plus first and last few)
    sweep_layers = sorted(set(
        list(range(0, min(8, num_layers))) +
        list(range(0, num_layers, 2)) +
        list(range(max(0, num_layers - 8), num_layers))
    ))

    for layer in sweep_layers:
        h_acts = extract_activations(model, tokenizer, harmful_prompts, layer)
        b_acts = extract_activations(model, tokenizer, benign_prompts, layer)

        X = np.vstack([h_acts, b_acts])
        y = np.array([1] * len(h_acts) + [0] * len(b_acts))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aurocs = []
        for tr, va in skf.split(X, y):
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            try:
                aurocs.append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
            except:
                aurocs.append(0.5)
        mean_a = float(np.mean(aurocs))
        probe_sweep[layer] = {"auroc": mean_a, "std": float(np.std(aurocs))}
        print(f"    Layer {layer}: AUROC {mean_a:.4f}")

        if mean_a > best_auroc:
            best_auroc = mean_a
            best_layer = layer
            sc_final = StandardScaler()
            clf_final = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            clf_final.fit(sc_final.fit_transform(X), y)
            best_probe_data = {"probe": clf_final, "scaler": sc_final}

    # Save prompt probe
    probe_dir = model_out / "jailbreak_probe"
    probe_dir.mkdir(exist_ok=True)
    with open(probe_dir / "best_probe.pkl", "wb") as f:
        pickle.dump(best_probe_data, f)
    json.dump({"layer": best_layer, "auroc": best_auroc}, open(probe_dir / "best_layer.json", "w"), indent=2)
    json.dump(probe_sweep, open(probe_dir / "layer_sweep.json", "w"), indent=2)
    print(f"    Best prompt probe: layer {best_layer}, AUROC {best_auroc:.4f}")

    # ---- Step 2: Extract refusal steering vector ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 2: Refusal steering vector at layer {sl}")
    h_acts = extract_activations(model, tokenizer, harmful_prompts, sl)
    b_acts = extract_activations(model, tokenizer, benign_prompts, sl)
    vec = (h_acts.mean(axis=0) - b_acts.mean(axis=0))
    vec = vec / np.linalg.norm(vec)
    vec_t = torch.from_numpy(vec).float()
    vec_dir = model_out / "refusal_vector"
    vec_dir.mkdir(exist_ok=True)
    torch.save(vec_t, vec_dir / f"vector_layer{sl}.pt")
    print(f"    Saved refusal vector (norm={np.linalg.norm(vec):.4f})")

    # ---- Step 3: Train response-time probe ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 3: Response-time probe")
    attacks = generate_prefilling_attacks(n_attacks=40)
    benign_all = list(BENIGN_QUERIES) + list(BENIGN_OPENENDED[:10])
    benign = [{"prompt": q} for q in benign_all[:50]]

    # Collect response activations
    all_acts = {l: [] for l in range(num_layers + 1)}  # +1 for embedding
    labels = []

    for group_name, samples, label_type in [("attacks", attacks, "attack"), ("benign", benign, "benign")]:
        print(f"    Collecting {group_name} response activations...")
        for i, s in enumerate(samples):
            prompt = s["prompt"] if isinstance(s, dict) else s
            prefill = s.get("prefill") if isinstance(s, dict) else None
            try:
                resp, per_token = generate_with_hidden(model, tokenizer, prompt, N_RESP_TOKENS, prefill)
            except Exception as e:
                print(f"      [{i}] ERROR: {e}")
                continue
            is_attack = label_type == "attack"
            judge = judge_response(resp, is_attack=is_attack)
            label = 1 if judge.is_harmful else 0
            n_avail = min(N_RESP_TOKENS, len(per_token))
            if n_avail == 0:
                continue
            for li in range(min(num_layers + 1, len(per_token[0]))):
                la = np.stack([per_token[t][li] for t in range(n_avail)])
                all_acts[li].append(la.mean(axis=0))
            labels.append(label)
            if (i + 1) % 10 == 0:
                print(f"      {i+1}/{len(samples)}")

    labels = np.array(labels)
    print(f"    Samples: {len(labels)} ({labels.sum()} positive, {(1-labels).sum()} negative)")

    if labels.sum() < 3:
        print(f"    WARNING: too few positives for response probe")
        del model; torch.cuda.empty_cache()
        return

    # Train response probe with layer sweep
    resp_sweep = {}
    resp_best_layer, resp_best_auroc, resp_best_probe, resp_best_scaler = -1, -1, None, None
    available_layers = [l for l in all_acts if len(all_acts[l]) == len(labels)]

    for layer in available_layers:
        X = np.stack(all_acts[layer])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aurocs = []
        for tr, va in skf.split(X, labels):
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            clf.fit(sc.fit_transform(X[tr]), labels[tr])
            try:
                aurocs.append(roc_auc_score(labels[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
            except:
                aurocs.append(0.5)
        mean_a = float(np.mean(aurocs))
        resp_sweep[layer] = {"auroc": mean_a, "std": float(np.std(aurocs))}
        if mean_a > resp_best_auroc:
            resp_best_auroc = mean_a
            resp_best_layer = layer

    # Retrain best on all data
    X = np.stack(all_acts[resp_best_layer])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(Xs, labels)

    with open(model_out / "response_probe.pkl", "wb") as f:
        pickle.dump({"probe": clf, "scaler": sc}, f)
    json.dump({
        "best_layer": resp_best_layer, "best_auroc": resp_best_auroc,
        "n_samples": len(labels), "n_positive": int(labels.sum()),
        "layer_sweep": {str(k): v for k, v in resp_sweep.items()},
    }, open(model_out / "response_probe_meta.json", "w"), indent=2)
    print(f"    Response probe: layer {resp_best_layer}, AUROC {resp_best_auroc:.4f}")

    # ---- Step 4: Halt intervention test ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 4: Halt intervention on 40 prefilling attacks")
    r_probe = clf
    r_scaler = sc

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
        la = np.stack([per_token[t][resp_best_layer] for t in range(n_av)])
        ma = la.mean(axis=0).reshape(1, -1)
        score = float(r_probe.predict_proba(r_scaler.transform(ma))[0, 1])
        fired = score > RESP_THRESHOLD
        if fired:
            halt_results.append({"halted": True, "is_harmful": False, "score": score})
        else:
            from src.eval.safety_eval import generate_response
            resp, _ = generate_response(model, tokenizer, prompt, device, 200, prefill=prefill)
            harmful = judge_response(resp, is_attack=True).is_harmful
            halt_results.append({"halted": False, "is_harmful": bool(harmful), "score": score})
        if (j + 1) % 10 == 0:
            print(f"    {j+1}/{len(attacks)}")

    n_halt = sum(r["halted"] for r in halt_results)
    n_harm = sum(r["is_harmful"] for r in halt_results)
    asr = n_harm / len(halt_results) if halt_results else 1.0
    print(f"    Halt rate: {n_halt}/{len(halt_results)}, ASR: {asr:.2%}")

    json.dump({
        "halt_rate": n_halt / len(halt_results), "asr": asr,
        "n_halted": n_halt, "n_harmful": n_harm, "n_total": len(halt_results),
    }, open(model_out / "halt_test.json", "w"), indent=2)

    # ---- Step 5: Asymmetry analysis ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 5: Asymmetry analysis")
    probe_weights = clf.coef_[0]
    refusal_vec_np = vec_t.numpy()

    # Cosine similarity at steering layer
    if len(probe_weights) == len(refusal_vec_np):
        cos_sim = float(np.dot(probe_weights, refusal_vec_np) /
                        (np.linalg.norm(probe_weights) * np.linalg.norm(refusal_vec_np) + 1e-10))
    else:
        cos_sim = None
        print(f"    Dim mismatch: probe {len(probe_weights)} vs vec {len(refusal_vec_np)}")

    analysis = {
        "model": model_name,
        "prompt_probe_layer": best_layer, "prompt_probe_auroc": best_auroc,
        "response_probe_layer": resp_best_layer, "response_probe_auroc": resp_best_auroc,
        "steering_layer": sl,
        "cos_sim_response_probe_vs_refusal": cos_sim,
        "halt_asr": asr, "halt_rate": n_halt / len(halt_results),
        "weight_norm": float(np.linalg.norm(probe_weights)),
        "weight_sparsity": float((np.abs(probe_weights) < 0.01 * np.linalg.norm(probe_weights)).mean()),
    }
    json.dump(analysis, open(model_out / "analysis.json", "w"), indent=2)
    print(f"    cos(response_probe, refusal): {cos_sim}")

    del model; torch.cuda.empty_cache()
    mark_done(f"{model_name}_all")
    print(f"\n[{time.strftime('%H:%M:%S')}] {model_name} pipeline DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["qwen3-8b", "gemma3-12b", "gemma4-31b", "both"])
    args = parser.parse_args()

    if args.model == "both":
        run_pipeline("qwen3-8b")
        run_pipeline("gemma3-12b")
    else:
        run_pipeline(args.model)


if __name__ == "__main__":
    main()
