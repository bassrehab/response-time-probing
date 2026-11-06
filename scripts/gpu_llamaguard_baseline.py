#!/usr/bin/env python3
"""Llama Guard 3 baseline: compare output-level safety classifier to response probe.

Approach:
1. Generate responses from target model (Mistral/Llama) under prefilling attacks
2. Load Llama-Guard-3-8B as a separate judge
3. Classify each response as safe/unsafe
4. Compare: Llama Guard detection rate vs response probe halt rate

Key comparison point: Llama Guard requires full response generation + second
inference pass. Response probe detects after 5 tokens and halts before harmful
content is generated.

Usage: python scripts/gpu_llamaguard_baseline.py
"""

import json, os, sys, time
from pathlib import Path

import torch

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
_token_path = Path.home() / ".cache" / "huggingface" / "token"
if _token_path.exists():
    os.environ["HF_TOKEN"] = _token_path.read_text().strip()

sys.path.insert(0, ".")
from src.models.loader import load_model
from src.eval.judge import judge_response
from src.attacks.prefilling import generate_prefilling_attacks
from src.attacks.datasets.prompts import ADVBENCH_BEHAVIORS, BENIGN_QUERIES, BENIGN_OPENENDED
from src.attacks.registry import load_attack_set
from src.utils.seeding import seed_everything

seed_everything(42)
device = "cuda"

OUT = Path("results/llamaguard_baseline")
OUT.mkdir(parents=True, exist_ok=True)


def generate_response_text(model, tokenizer, prompt, prefill=None, max_new=200):
    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(fmt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    if prefill:
        text = prefill + text
    return text


def classify_with_llamaguard(guard_model, guard_tokenizer, prompt, response):
    """Use Llama Guard to classify a prompt+response pair."""
    # Llama Guard 3 uses a specific chat format
    chat = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        formatted = guard_tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            output = guard_model.generate(formatted, max_new_tokens=100, do_sample=False)
        result = guard_tokenizer.decode(output[0][formatted.shape[1]:], skip_special_tokens=True).strip()
        # Llama Guard outputs "safe" or "unsafe\nS1" etc.
        is_unsafe = "unsafe" in result.lower()
        return is_unsafe, result
    except Exception as e:
        return False, f"error: {e}"


def main():
    print(f"{'='*60}")
    print(f"LLAMA GUARD BASELINE COMPARISON")
    print(f"{'='*60}")

    # Build attack sets
    prefilling = generate_prefilling_attacks(n_attacks=40)
    autodan = load_attack_set("autodan", n_attacks=40, include_benign=False)
    benign = [{"prompt": q, "attack_type": "benign"} for q in list(BENIGN_QUERIES)[:40]]
    all_attacks = prefilling + autodan + benign

    results = {}

    for target_name in ["mistral-7b", "llama31-8b"]:
        print(f"\n[{time.strftime('%H:%M:%S')}] Target model: {target_name}")

        # Step 1: Generate responses from target model
        print(f"  Generating responses...")
        target_model, target_tok, _ = load_model(target_name, device=device)

        responses = []
        for i, attack in enumerate(all_attacks):
            prompt = attack["prompt"]
            prefill = attack.get("prefill")
            at = attack["attack_type"]
            try:
                resp = generate_response_text(target_model, target_tok, prompt, prefill)
            except:
                resp = ""
            heuristic = judge_response(resp, is_attack=(at != "benign"))
            responses.append({
                "prompt": prompt, "response": resp, "attack_type": at,
                "prefill": prefill, "heuristic_harmful": bool(heuristic.is_harmful),
            })
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(all_attacks)}")

        del target_model; torch.cuda.empty_cache()

        # Step 2: Load Llama Guard and classify all responses
        print(f"\n  Loading Llama Guard 3...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        guard_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-Guard-3-8B",
            torch_dtype=torch.bfloat16, device_map="auto",
        )
        guard_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")
        print(f"  Llama Guard loaded")

        for i, r in enumerate(responses):
            is_unsafe, guard_output = classify_with_llamaguard(
                guard_model, guard_tok, r["prompt"], r["response"]
            )
            r["llamaguard_unsafe"] = is_unsafe
            r["llamaguard_output"] = guard_output[:80]
            if (i + 1) % 20 == 0:
                print(f"    Judged {i+1}/{len(responses)}")

        del guard_model; torch.cuda.empty_cache()

        # Step 3: Compute comparison metrics
        for at in ["prefilling", "autodan", "benign"]:
            subset = [r for r in responses if r["attack_type"] == at]
            if not subset:
                continue
            n = len(subset)
            heuristic_harmful = sum(r["heuristic_harmful"] for r in subset)
            guard_caught = sum(r["llamaguard_unsafe"] for r in subset)

            is_attack = at != "benign"
            if is_attack:
                # ASR = fraction harmful (heuristic)
                heur_asr = heuristic_harmful / n
                # Llama Guard detection = fraction it flags as unsafe
                guard_detect = guard_caught / n
                print(f"\n  {target_name} / {at} (n={n}):")
                print(f"    Heuristic harmful: {heuristic_harmful}/{n} ({heur_asr:.1%})")
                print(f"    Llama Guard unsafe: {guard_caught}/{n} ({guard_detect:.1%})")
            else:
                # Benign FPR
                guard_fpr = guard_caught / n
                print(f"\n  {target_name} / benign (n={n}):")
                print(f"    Llama Guard FPR: {guard_caught}/{n} ({guard_fpr:.1%})")

        # Save
        model_results = {
            "target_model": target_name,
            "n_prefilling": len([r for r in responses if r["attack_type"] == "prefilling"]),
            "n_autodan": len([r for r in responses if r["attack_type"] == "autodan"]),
            "n_benign": len([r for r in responses if r["attack_type"] == "benign"]),
        }
        for at in ["prefilling", "autodan", "benign"]:
            subset = [r for r in responses if r["attack_type"] == at]
            if not subset:
                continue
            model_results[f"{at}_heuristic_harmful"] = sum(r["heuristic_harmful"] for r in subset)
            model_results[f"{at}_guard_caught"] = sum(r["llamaguard_unsafe"] for r in subset)
            model_results[f"{at}_total"] = len(subset)

        results[target_name] = model_results
        json.dump(responses, open(OUT / f"{target_name}_samples.json", "w"), indent=2)

    json.dump(results, open(OUT / "comparison.json", "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Attack':<12} {'Heuristic':>10} {'LlamaGuard':>12} {'n':>5}")
    for mn, mr in results.items():
        for at in ["prefilling", "autodan", "benign"]:
            h = mr.get(f"{at}_heuristic_harmful", 0)
            g = mr.get(f"{at}_guard_caught", 0)
            n = mr.get(f"{at}_total", 0)
            if n > 0:
                print(f"{mn:<15} {at:<12} {h/n:>9.1%} {g/n:>11.1%} {n:>5}")

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE")


if __name__ == "__main__":
    main()
