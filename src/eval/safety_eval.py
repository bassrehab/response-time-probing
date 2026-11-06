"""Safety evaluation: ASR/DSR computation across attack-defense combinations.

Orchestrates: attack prompts → defense → generation → judge → metrics.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from src.eval.judge import judge_response, JudgeResult
from src.eval.metrics import compute_safety_metrics, SafetyMetrics


@dataclass
class EvalSample:
    """Single evaluation sample with full provenance."""

    prompt: str
    response: str
    attack_type: str
    defense_type: str
    model: str
    judge_result: dict          # Serialized JudgeResult
    probe_scores: list[float]   # Probe score history during generation
    latency_ms: float


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 200,
    prefill: str | None = None,
) -> tuple[str, float]:
    """Generate a response, optionally with a prefilled assistant start.

    Returns:
        (response_text, latency_ms)
    """
    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=4096
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if prefill:
        response = prefill + response

    return response, latency_ms


def run_safety_eval(
    model,
    tokenizer,
    attack_prompts: list[dict],
    defense_type: str,
    model_name: str,
    device: str,
    pgs=None,
    max_new_tokens: int = 200,
    judge_method: str = "heuristic",
) -> tuple[list[EvalSample], SafetyMetrics]:
    """Run safety evaluation across a set of attack prompts.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        attack_prompts: List of dicts with keys: prompt, attack_type, and optionally prefill.
        defense_type: Name of the defense being evaluated.
        model_name: Name of the model.
        device: Device string.
        pgs: ProbeGatedSteering instance (None for ungated).
        max_new_tokens: Max tokens to generate.
        judge_method: Judge method ("heuristic" or "llm").

    Returns:
        (samples, metrics) tuple.
    """
    samples = []

    for i, attack in enumerate(attack_prompts):
        prompt = attack["prompt"]
        attack_type = attack["attack_type"]
        prefill = attack.get("prefill")

        # Reset probe-gated steering state if active
        if pgs is not None:
            pgs.reset()

        response, latency_ms = generate_response(
            model, tokenizer, prompt, device,
            max_new_tokens=max_new_tokens,
            prefill=prefill,
        )

        # Judge the response — benign prompts always return safe
        is_attack = attack_type != "benign"
        judge_result = judge_response(response, is_attack=is_attack, method=judge_method)

        # Collect probe scores if available
        probe_scores = list(pgs.score_history) if pgs is not None else []

        sample = EvalSample(
            prompt=prompt,
            response=response,
            attack_type=attack_type,
            defense_type=defense_type,
            model=model_name,
            judge_result=asdict(judge_result),
            probe_scores=probe_scores,
            latency_ms=latency_ms,
        )
        samples.append(sample)

        if (i + 1) % 10 == 0:
            print(f"    Evaluated {i + 1}/{len(attack_prompts)} prompts")

    # Compute metrics
    is_harmful_list = [s.judge_result["is_harmful"] for s in samples]
    attack_types = [s.attack_type for s in samples]
    latencies = [s.latency_ms for s in samples]

    # Collect max probe scores on benign samples for hard FPR computation
    # Hard FPR = fraction of benign samples where probe fires above threshold
    probe_scores_benign = []
    for s in samples:
        if s.attack_type == "benign" and s.probe_scores:
            # Use max probe score during generation as the detection signal
            probe_scores_benign.append(max(s.probe_scores))

    metrics = compute_safety_metrics(
        is_harmful_list, attack_types, latencies,
        probe_scores_benign=probe_scores_benign if probe_scores_benign else None,
    )

    return samples, metrics


def save_eval_results(
    output_dir: Path,
    samples: list[EvalSample],
    metrics: SafetyMetrics,
    defense_type: str,
    model_name: str,
) -> None:
    """Save evaluation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw samples as JSONL
    with open(output_dir / f"{defense_type}_samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")

    # Save aggregate metrics
    with open(output_dir / f"{defense_type}_metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)
