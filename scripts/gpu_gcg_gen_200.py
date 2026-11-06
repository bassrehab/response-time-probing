#!/usr/bin/env python3
"""Generate 200 GCG adversarial suffixes for Mistral-7B and Llama-3.1-8B.

Uses nanogcg with 512 optimization steps per suffix.
Saves suffixes + losses to data/attacks/gcg/.

Usage: python scripts/gpu_gcg_gen_200.py [--model mistral-7b|llama31-8b|both]
"""

import json, os, sys, time, argparse
from pathlib import Path

import torch

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

import nanogcg
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path("data/attacks/gcg")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load full AdvBench (520 behaviors) from GitHub
import csv
import urllib.request

sys.path.insert(0, ".")

ADVBENCH_CSV_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
ADVBENCH_CACHE = Path("data/advbench_full.csv")


def load_advbench(n=200):
    """Load n behaviors from the full 520-item AdvBench set."""
    if not ADVBENCH_CACHE.exists():
        ADVBENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ADVBENCH_CSV_URL, ADVBENCH_CACHE)
        print(f"  Downloaded AdvBench to {ADVBENCH_CACHE}")
    with open(ADVBENCH_CACHE) as f:
        reader = csv.DictReader(f)
        behaviors = [row["goal"] for row in reader]
    print(f"  AdvBench: {len(behaviors)} total, using first {min(n, len(behaviors))}")
    return behaviors[:n]

MODEL_CONFIGS = {
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "dtype": torch.bfloat16,
    },
    "llama31-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "dtype": torch.bfloat16,
    },
}

N_SUFFIXES = 200
GCG_STEPS = 512
TARGET = "Sure, here is"


def generate_suffixes(model_name):
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"GCG GENERATION: {model_name} ({N_SUFFIXES} suffixes, {GCG_STEPS} steps)")
    print(f"{'='*60}")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading {cfg['hf_id']}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=cfg["dtype"], device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Use first N_SUFFIXES behaviors from full AdvBench (520 available)
    behaviors = load_advbench(N_SUFFIXES)

    suffixes = []
    losses = []
    suffix_file = DATA_DIR / f"{model_name}_suffixes_n200.json"
    loss_file = DATA_DIR / f"{model_name}_losses_n200.json"
    checkpoint_file = DATA_DIR / f"{model_name}_checkpoint_n200.json"

    # Resume from checkpoint if exists
    start_idx = 0
    if checkpoint_file.exists():
        cp = json.load(open(checkpoint_file))
        suffixes = cp["suffixes"]
        losses = cp["losses"]
        start_idx = len(suffixes)
        print(f"  Resuming from checkpoint at index {start_idx}")

    for i in range(start_idx, len(behaviors)):
        behavior = behaviors[i]
        print(f"\n[{time.strftime('%H:%M:%S')}] Suffix {i+1}/{len(behaviors)}: {behavior[:60]}...")

        try:
            config = nanogcg.GCGConfig(
                num_steps=GCG_STEPS,
                search_width=512,
                topk=256,
                seed=42 + i,
            )
            result = nanogcg.run(model, tokenizer, behavior, TARGET, config=config)
            suffix = result.best_string
            loss = float(result.best_loss)
            print(f"  Loss: {loss:.4f} | Suffix: {suffix[:50]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            suffix = ""
            loss = 999.0

        suffixes.append(suffix)
        losses.append(loss)

        # Checkpoint every 10
        if (i + 1) % 10 == 0:
            json.dump({"suffixes": suffixes, "losses": losses}, open(checkpoint_file, "w"), indent=2)
            print(f"  Checkpoint saved ({i+1}/{len(behaviors)})")

    # Save final
    json.dump(suffixes, open(suffix_file, "w"), indent=2)
    json.dump(losses, open(loss_file, "w"), indent=2)
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Stats
    valid = sum(1 for s in suffixes if s)
    low_loss = sum(1 for l in losses if l < 0.5)
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE: {model_name}")
    print(f"  Valid suffixes: {valid}/{len(suffixes)}")
    print(f"  Low-loss (<0.5): {low_loss}/{len(suffixes)}")
    print(f"  Mean loss: {sum(losses)/len(losses):.4f}")
    print(f"  Elapsed: {elapsed/3600:.1f} hours")
    print(f"  Saved: {suffix_file}")
    print(f"{'='*60}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["mistral-7b", "llama31-8b", "both"])
    args = parser.parse_args()

    if args.model == "both":
        generate_suffixes("mistral-7b")
        generate_suffixes("llama31-8b")
    else:
        generate_suffixes(args.model)


if __name__ == "__main__":
    main()
