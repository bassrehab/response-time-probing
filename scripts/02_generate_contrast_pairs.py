#!/usr/bin/env python3
"""Generate contrast pair datasets for probe training.

Uses the representation engineering approach: contrast harmful prompts (positive)
against benign prompts (negative). Activations are extracted at the last prompt
token (before generation), capturing the model's internal intent representation.

Currently implements: jailbreak probe type only.
Source: AdvBench harmful behaviors + matched benign queries.

Usage:
    python scripts/02_generate_contrast_pairs.py --probe_type jailbreak --n_pairs 20
    python scripts/02_generate_contrast_pairs.py --probe_type jailbreak --n_pairs 40
"""

import argparse
import json
import random

from src.attacks.datasets.prompts import (
    ADVBENCH_BEHAVIORS, BENIGN_QUERIES, BENIGN_OPENENDED, SYCOPHANCY_PAIRS,
)
from src.utils.paths import contrast_pairs_dir
from src.utils.seeding import seed_everything


def generate_jailbreak_pairs(n_pairs: int, seed: int = 42, diverse: bool = True) -> list[dict]:
    """Generate contrast pairs for jailbreak probe training.

    Each sample is a single prompt with a label:
      - positive (label=1): harmful prompt from AdvBench
      - negative (label=0): diverse benign prompts

    When diverse=True (default), negatives are drawn from 4 categories:
      - Topic-matched benign (safety-adjacent)
      - Open-ended writing/advice
      - Coding/technical
      - Casual/conversational

    This ensures the probe learns "actually harmful" vs "everything else"
    instead of "safety-adjacent topic" vs "safe topic."
    """
    random.seed(seed)
    n_pairs = min(n_pairs, len(ADVBENCH_BEHAVIORS))

    # Build positive set
    positives = ADVBENCH_BEHAVIORS[:n_pairs]

    # Build negative set
    if diverse and len(BENIGN_OPENENDED) > 0:
        # Mix of topic-matched + diverse benign
        n_topic = min(n_pairs // 4, len(BENIGN_QUERIES))
        n_diverse = n_pairs - n_topic
        n_diverse = min(n_diverse, len(BENIGN_OPENENDED))

        negatives = (
            BENIGN_QUERIES[:n_topic]
            + random.sample(BENIGN_OPENENDED, n_diverse)
        )
        random.shuffle(negatives)
        negatives = negatives[:n_pairs]
    else:
        negatives = BENIGN_QUERIES[:n_pairs]

    pairs = []
    for prompt in positives:
        pairs.append({"prompt": prompt, "label": 1, "probe_type": "jailbreak"})
    for prompt in negatives:
        pairs.append({"prompt": prompt, "label": 0, "probe_type": "jailbreak"})

    random.shuffle(pairs)
    return pairs


def generate_sycophancy_pairs(n_pairs: int, seed: int = 42) -> list[dict]:
    """Generate contrast pairs for sycophancy probe training.

    Uses user assertions containing common misconceptions. The probe should
    detect the model's internal state when it's about to agree with a wrong
    claim (sycophantic, label=1) vs correct the user (truthful, label=0).

    For each misconception:
      - positive (label=1): user asserts wrong claim → sycophancy-inducing prompt
      - negative (label=0): neutral question on same topic → no sycophancy pressure

    The key insight: the user's confident assertion of a wrong claim puts
    social pressure on the model to agree. The probe learns to detect this
    "about to be sycophantic" internal state.
    """
    random.seed(seed)
    n_pairs = min(n_pairs, len(SYCOPHANCY_PAIRS))

    pairs = []
    for i in range(n_pairs):
        sp = SYCOPHANCY_PAIRS[i]

        # Positive: user confidently asserts wrong claim (sycophancy pressure)
        pairs.append({
            "prompt": sp["user"],
            "label": 1,  # Sycophancy-inducing
            "probe_type": "sycophancy",
        })

        # Negative: neutral question about the same topic (no sycophancy pressure)
        # Extract the topic and create a neutral factual question
        neutral = _make_neutral_question(sp)
        pairs.append({
            "prompt": neutral,
            "label": 0,  # No sycophancy pressure
            "probe_type": "sycophancy",
        })

    random.shuffle(pairs)
    return pairs


def _make_neutral_question(sp: dict) -> str:
    """Convert a sycophancy pair into a neutral factual question."""
    # Map from misconception topics to neutral questions
    neutral_map = {
        "geography": "What are some commonly misunderstood facts about world geography?",
        "history": "What are some popular historical myths that have been debunked?",
        "neuroscience": "How does the human brain allocate its processing resources?",
        "biology": "What are some interesting facts about animal cognition and behavior?",
        "physics": "What are some counterintuitive facts about everyday physics?",
        "astronomy": "What causes the seasons to change throughout the year?",
        "medicine": "What does scientific research say about common cold remedies?",
        "language": "What are some common misconceptions about food terminology?",
        "chemistry": "What is the molecular structure of common materials like glass?",
        "technology": "What hardware components matter most for different computing tasks?",
    }
    return neutral_map.get(sp["topic"], "What are some commonly held beliefs that turn out to be incorrect?")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contrast pairs for probe training")
    parser.add_argument("--probe_type", type=str, default="jailbreak",
                        choices=["jailbreak", "sycophancy"],
                        help="Probe type to generate pairs for")
    parser.add_argument("--n_pairs", type=int, default=20,
                        help="Number of contrast PAIRS (each = 1 positive + 1 negative)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    print(f"Generating {args.n_pairs} contrast pairs for probe type: {args.probe_type}")

    if args.probe_type == "jailbreak":
        samples = generate_jailbreak_pairs(args.n_pairs, seed=args.seed)
    elif args.probe_type == "sycophancy":
        samples = generate_sycophancy_pairs(args.n_pairs, seed=args.seed)
    else:
        raise ValueError(f"Unsupported probe type: {args.probe_type}")

    output_dir = contrast_pairs_dir()
    output_path = output_dir / f"{args.probe_type}_pairs.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = sum(1 for s in samples if s["label"] == 0)
    print(f"Saved {len(samples)} samples ({n_pos} harmful, {n_neg} benign) to {output_path}")


if __name__ == "__main__":
    main()
