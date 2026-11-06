#!/usr/bin/env python3
"""Response-time probe for prefilling defense.

Key idea: current probes detect harmful INPUT. Prefilling bypasses this by
making the input look benign. But the model KNOWS it's generating harmful
content — that knowledge should be detectable in response-time activations.

Experiment:
1. Run 50 prefilling attacks on Mistral, capture activations during first
   5 generated response tokens
2. Run 50 benign queries, capture activations during first 5 generated tokens
3. Label: positive = judge says response is harmful, negative = benign
4. Train linear probe on response-token activations, layer sweep
5. Report AUROC

If AUROC > 0.85, this is a novel defense mechanism for prefilling attacks.

Usage:
    python scripts/run_response_probe.py
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.attacks.prefilling import generate_prefilling_attacks
from src.attacks.datasets.prompts import BENIGN_QUERIES, BENIGN_OPENENDED
from src.eval.judge import judge_response
from src.models.loader import get_device, load_model
from src.utils.paths import RESULTS_DIR
from src.utils.seeding import seed_everything

seed_everything(42)
device = get_device()
model_name = "mistral-7b"
N_ATTACK = 40  # 40 prefilling attacks available
N_BENIGN = 40  # Match
N_RESPONSE_TOKENS = 5  # First N generated tokens to extract activations from

out_dir = RESULTS_DIR / "response_probe" / model_name
out_dir.mkdir(parents=True, exist_ok=True)


def generate_with_hidden_states(model, tokenizer, prompt, device, max_new_tokens=10, prefill=None):
    """Generate a response and return hidden states for each generated token.

    Returns:
        response_text: generated text
        hidden_states_per_token: list of dicts[layer_idx -> tensor(hidden_dim,)]
        generated_ids: generated token IDs
    """
    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    generated_ids = output.sequences[0][prompt_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if prefill:
        response_text = prefill + response_text

    # output.hidden_states is a tuple of length num_generated_tokens
    # Each element is a tuple of (num_layers+1) tensors of shape (batch, seq_len, hidden_dim)
    # For token t > 0, seq_len = 1 (just the new token)
    # For token 0, seq_len = prompt_len

    # Extract last-token activation at each layer for each generated token
    per_token_activations = []
    for token_idx, layer_states in enumerate(output.hidden_states):
        # layer_states is a tuple of (num_layers+1) tensors
        # tensor shape: (1, seq_len_at_this_step, hidden_dim)
        # For token 0: seq_len = prompt_len, take last position
        # For token > 0: seq_len = 1, take position 0
        token_acts = {}
        for layer_idx, state in enumerate(layer_states):
            # state shape: (1, seq_len, hidden_dim)
            last_pos_act = state[0, -1, :].detach().cpu().float().numpy()
            token_acts[layer_idx] = last_pos_act
        per_token_activations.append(token_acts)

    return response_text, per_token_activations, generated_ids


def main():
    print(f"=" * 60)
    print(f"RESPONSE-TIME PROBE EXPERIMENT: {model_name}")
    print(f"=" * 60)

    print(f"\n[{time.strftime('%H:%M:%S')}] Loading model...")
    t0 = time.time()
    model, tokenizer, config = load_model(model_name, device=device)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Hidden dim: {config.hidden_dim}, Layers: {config.num_layers}")

    # Build attack + benign sets
    prefill_attacks = generate_prefilling_attacks(n_attacks=N_ATTACK)
    benign_all = list(BENIGN_QUERIES) + list(BENIGN_OPENENDED)
    benign_queries = [{"prompt": q, "attack_type": "benign"} for q in benign_all[:N_BENIGN]]

    print(f"\n[{time.strftime('%H:%M:%S')}] Generating + extracting activations")
    print(f"  {len(prefill_attacks)} prefilling attacks + {len(benign_queries)} benign")
    print(f"  Extracting first {N_RESPONSE_TOKENS} response tokens at all layers")

    # Storage: layer_idx -> list of activations (one per sample), labels
    # Use mean-pooled activation over first N response tokens at each layer
    num_layers = config.num_layers + 1  # +1 for embedding layer
    all_activations = {l: [] for l in range(num_layers)}
    labels = []
    sample_records = []  # For per-sample inspection

    def process_sample(prompt_info, is_attack):
        prompt = prompt_info["prompt"]
        prefill = prompt_info.get("prefill")
        response, per_token_acts, _ = generate_with_hidden_states(
            model, tokenizer, prompt, device,
            max_new_tokens=N_RESPONSE_TOKENS, prefill=prefill,
        )

        # Judge the response
        judge_result = judge_response(response, is_attack=is_attack)

        # For each layer, average activations across first N response tokens
        # per_token_acts[t] is dict[layer -> array(hidden_dim,)]
        # For prefilled attacks, token 0 is AFTER the prefill so it's the response continuation
        # Take tokens 1 onwards to avoid contamination from prompt processing
        response_token_indices = list(range(min(N_RESPONSE_TOKENS, len(per_token_acts))))
        if len(response_token_indices) == 0:
            return None

        for layer_idx in range(num_layers):
            layer_acts = np.stack([per_token_acts[t][layer_idx] for t in response_token_indices])
            # Mean-pool across tokens
            mean_act = layer_acts.mean(axis=0)
            all_activations[layer_idx].append(mean_act)

        return response, judge_result

    pos_count = 0
    neg_count = 0

    for i, attack in enumerate(prefill_attacks):
        result = process_sample(attack, is_attack=True)
        if result is None:
            continue
        response, judge_result = result
        is_harmful = judge_result.is_harmful
        labels.append(1 if is_harmful else 0)
        pos_count += int(is_harmful)
        sample_records.append({
            "type": "prefilling",
            "prompt": attack["prompt"][:80],
            "is_harmful": is_harmful,
            "response_preview": response[:100],
        })
        if (i + 1) % 10 == 0:
            print(f"    Attacks {i+1}/{len(prefill_attacks)}")

    for i, benign in enumerate(benign_queries):
        result = process_sample(benign, is_attack=False)
        if result is None:
            continue
        response, judge_result = result
        labels.append(0)  # Always label benign as 0
        neg_count += 1
        sample_records.append({
            "type": "benign",
            "prompt": benign["prompt"][:80],
            "is_harmful": False,
            "response_preview": response[:100],
        })
        if (i + 1) % 10 == 0:
            print(f"    Benign {i+1}/{len(benign_queries)}")

    print(f"\n  Collected: {len(labels)} samples ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
    print(f"  Prefilling attacks: {pos_count}/{len(prefill_attacks)} judged harmful")

    # Save sample records for inspection
    with open(out_dir / "samples.json", "w") as f:
        json.dump(sample_records, f, indent=2)

    labels = np.array(labels)

    # Check class balance
    if labels.sum() < 5 or (1 - labels).sum() < 5:
        print(f"\n  WARNING: Severe class imbalance. Cannot train reliable probe.")
        print(f"  Positive: {labels.sum()}, Negative: {(1-labels).sum()}")
        # Save what we have and exit
        return

    # Free model
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # Train probe at each layer with 5-fold CV
    print(f"\n[{time.strftime('%H:%M:%S')}] Training probes (5-fold CV)")
    print(f"  Layer sweep across {num_layers} layers")

    layer_results = {}
    best_auroc = -1
    best_layer = -1

    for layer_idx in range(num_layers):
        acts = np.stack(all_activations[layer_idx])  # (n_samples, hidden_dim)

        # Cross-validated AUROC
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aurocs = []
        for train_idx, val_idx in skf.split(acts, labels):
            X_train, X_val = acts[train_idx], acts[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            probe = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, C=1.0)
            probe.fit(X_train_s, y_train)

            probs = probe.predict_proba(X_val_s)[:, 1]
            try:
                auroc = roc_auc_score(y_val, probs)
            except ValueError:
                auroc = 0.5
            fold_aurocs.append(auroc)

        mean_auroc = float(np.mean(fold_aurocs))
        std_auroc = float(np.std(fold_aurocs))
        layer_results[layer_idx] = {
            "auroc": mean_auroc,
            "std": std_auroc,
            "fold_aurocs": [float(a) for a in fold_aurocs],
        }
        print(f"  Layer {layer_idx:3d}: AUROC={mean_auroc:.4f} (±{std_auroc:.4f})")

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_layer = layer_idx

    # Train final probe at best layer on all data
    best_acts = np.stack(all_activations[best_layer])
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(best_acts)
    probe_final = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, C=1.0)
    probe_final.fit(X_scaled, labels)

    # Save
    with open(out_dir / "best_probe.pkl", "wb") as f:
        pickle.dump({"probe": probe_final, "scaler": scaler_final}, f)

    with open(out_dir / "layer_sweep.json", "w") as f:
        json.dump(layer_results, f, indent=2)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "model": model_name,
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "n_samples": len(labels),
            "n_positive": int(labels.sum()),
            "n_negative": int((1 - labels).sum()),
            "n_response_tokens": N_RESPONSE_TOKENS,
            "pooling": "mean",
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Best layer: {best_layer}")
    print(f"  Best AUROC: {best_auroc:.4f}")
    print(f"  Samples: {len(labels)} ({labels.sum()} harmful, {(1-labels).sum()} benign)")

    if best_auroc > 0.85:
        print(f"\n  POSITIVE RESULT: Response probe achieves AUROC > 0.85")
        print(f"  → Wire into gated system, run Phase 2 (full eval on prefilling)")
    elif best_auroc > 0.70:
        print(f"\n  MARGINAL: AUROC {best_auroc:.3f} is promising but not strong")
        print(f"  → Try different pooling, more response tokens, or tempted sampling")
    else:
        print(f"\n  NEGATIVE RESULT: Response probe cannot distinguish harmful generation")
        print(f"  → Report as negative result in paper")

    print(f"\n  Saved to {out_dir}")


if __name__ == "__main__":
    main()
