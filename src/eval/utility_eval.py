"""Utility evaluation: MMLU and GSM8K subsets for measuring helpfulness.

Measures whether a safety defense degrades the model's general capabilities.
Uses small subsets for fast iteration; full benchmarks on A100 for the paper.

The key metric is utility delta: how much does the defense hurt performance
compared to the undefended model? A good defense has high DSR with low
utility delta.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

MMLU_SUBJECTS = [
    "abstract_algebra",
    "college_physics",
    "computer_security",
    "high_school_us_history",
    "college_computer_science",
]


def load_mmlu_from_hf(
    subjects: list[str] | None = None,
    n_per_subject: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Load real MMLU questions from HuggingFace datasets.

    Args:
        subjects: List of MMLU subject names. Defaults to MMLU_SUBJECTS.
        n_per_subject: Max questions per subject.
        seed: Random seed for sampling.

    Returns:
        List of question dicts with keys: question, choices, answer, subject.
    """
    import random
    from datasets import load_dataset

    random.seed(seed)

    if subjects is None:
        subjects = MMLU_SUBJECTS

    questions = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        items = list(ds)
        if len(items) > n_per_subject:
            items = random.sample(items, n_per_subject)
        for item in items:
            questions.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],
                "subject": subject,
            })

    random.shuffle(questions)
    return questions


# Small MMLU subset for fast iteration. Covers diverse topics to detect
# broad capability degradation from steering.
MMLU_QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": 1,
        "subject": "geography",
    },
    {
        "question": "Which of the following is NOT a programming paradigm?",
        "choices": ["Object-oriented", "Functional", "Procedural", "Alphabetical"],
        "answer": 3,
        "subject": "computer_science",
    },
    {
        "question": "What is the derivative of x^3?",
        "choices": ["x^2", "3x^2", "3x", "x^3"],
        "answer": 1,
        "subject": "mathematics",
    },
    {
        "question": "Which amendment to the US Constitution abolished slavery?",
        "choices": ["12th", "13th", "14th", "15th"],
        "answer": 1,
        "subject": "history",
    },
    {
        "question": "What is the powerhouse of the cell?",
        "choices": ["Nucleus", "Ribosome", "Mitochondria", "Endoplasmic reticulum"],
        "answer": 2,
        "subject": "biology",
    },
    {
        "question": "In economics, what does GDP stand for?",
        "choices": ["Gross Domestic Product", "General Domestic Price",
                     "Gross Development Plan", "Global Domestic Product"],
        "answer": 0,
        "subject": "economics",
    },
    {
        "question": "Which gas makes up the majority of Earth's atmosphere?",
        "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"],
        "answer": 2,
        "subject": "chemistry",
    },
    {
        "question": "What is the time complexity of binary search?",
        "choices": ["O(n)", "O(n log n)", "O(log n)", "O(1)"],
        "answer": 2,
        "subject": "computer_science",
    },
    {
        "question": "Who wrote 'The Republic'?",
        "choices": ["Aristotle", "Socrates", "Plato", "Homer"],
        "answer": 2,
        "subject": "philosophy",
    },
    {
        "question": "What is the speed of light in vacuum, approximately?",
        "choices": ["3×10^6 m/s", "3×10^8 m/s", "3×10^10 m/s", "3×10^4 m/s"],
        "answer": 1,
        "subject": "physics",
    },
    {
        "question": "Which of the following is a valid SQL command?",
        "choices": ["GRAB", "FETCH", "SELECT", "PICK"],
        "answer": 2,
        "subject": "computer_science",
    },
    {
        "question": "What is the pH of pure water at 25°C?",
        "choices": ["0", "7", "14", "1"],
        "answer": 1,
        "subject": "chemistry",
    },
    {
        "question": "The Pythagorean theorem relates to which shape?",
        "choices": ["Circle", "Square", "Right triangle", "Pentagon"],
        "answer": 2,
        "subject": "mathematics",
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Venus", "Jupiter", "Mars", "Saturn"],
        "answer": 2,
        "subject": "astronomy",
    },
    {
        "question": "What is the primary function of white blood cells?",
        "choices": ["Carry oxygen", "Fight infection", "Clot blood", "Produce hormones"],
        "answer": 1,
        "subject": "biology",
    },
    {
        "question": "In machine learning, what is overfitting?",
        "choices": [
            "Model performs well on training data but poorly on new data",
            "Model performs poorly on all data",
            "Model is too simple to capture patterns",
            "Model trains too quickly",
        ],
        "answer": 0,
        "subject": "computer_science",
    },
    {
        "question": "What year did World War II end?",
        "choices": ["1943", "1944", "1945", "1946"],
        "answer": 2,
        "subject": "history",
    },
    {
        "question": "Which of the following is an example of a noble gas?",
        "choices": ["Chlorine", "Neon", "Sodium", "Iron"],
        "answer": 1,
        "subject": "chemistry",
    },
    {
        "question": "What does the 'P' stand for in HTTP?",
        "choices": ["Protocol", "Program", "Process", "Platform"],
        "answer": 0,
        "subject": "computer_science",
    },
    {
        "question": "The law of supply and demand is a concept in which field?",
        "choices": ["Physics", "Biology", "Economics", "Psychology"],
        "answer": 2,
        "subject": "economics",
    },
]


@dataclass
class UtilityQuestionResult:
    """Per-question result for full evidence logging."""

    question: str
    choices: list[str]
    correct_answer: int          # Index (0-3)
    correct_letter: str          # A/B/C/D
    model_response: str          # Raw model output
    predicted_answer: int | None  # Parsed index or None
    predicted_letter: str        # Parsed letter or "?"
    is_correct: bool
    subject: str
    latency_ms: float


@dataclass
class UtilityResult:
    """Results of a utility evaluation run."""

    accuracy: float                     # Overall accuracy (0.0 to 1.0)
    n_correct: int
    n_total: int
    per_subject: dict[str, float]       # Per-subject accuracy
    mean_latency_ms: float
    question_results: list[UtilityQuestionResult] | None = None


def format_mmlu_prompt(question: dict) -> str:
    """Format an MMLU question as a multiple-choice prompt."""
    choices_str = "\n".join(
        f"  {chr(65 + i)}. {choice}"
        for i, choice in enumerate(question["choices"])
    )
    return (
        f"Answer the following multiple choice question. "
        f"Reply with ONLY the letter (A, B, C, or D).\n\n"
        f"Question: {question['question']}\n{choices_str}\n\n"
        f"Answer:"
    )


def parse_answer(response: str) -> int | None:
    """Parse the model's answer from its response. Returns choice index or None."""
    response = response.strip()

    # Try to find a single letter A-D
    match = re.search(r'\b([A-D])\b', response[:20])
    if match:
        return ord(match.group(1)) - ord('A')

    # Try lowercase
    match = re.search(r'\b([a-d])\b', response[:20])
    if match:
        return ord(match.group(1)) - ord('a')

    return None


def run_utility_eval(
    model,
    tokenizer,
    device: str,
    questions: list[dict] | None = None,
    max_new_tokens: int = 5,
) -> UtilityResult:
    """Run MMLU-style utility evaluation.

    Args:
        model: The loaded model (with or without defense hooks).
        tokenizer: The tokenizer.
        device: Device string.
        questions: List of MMLU question dicts. Defaults to built-in subset.
        max_new_tokens: Max tokens to generate (short — just need the letter).

    Returns:
        UtilityResult with accuracy and per-subject breakdown.
    """
    if questions is None:
        questions = MMLU_QUESTIONS

    correct = 0
    total = 0
    subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}
    latencies = []
    question_results = []

    for q in questions:
        prompt = format_mmlu_prompt(q)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        predicted = parse_answer(response)
        is_correct = predicted == q["answer"]

        if is_correct:
            correct += 1
        total += 1

        subj = q["subject"]
        subject_correct[subj] = subject_correct.get(subj, 0) + (1 if is_correct else 0)
        subject_total[subj] = subject_total.get(subj, 0) + 1

        question_results.append(UtilityQuestionResult(
            question=q["question"],
            choices=q["choices"],
            correct_answer=q["answer"],
            correct_letter=chr(65 + q["answer"]),
            model_response=response.strip(),
            predicted_answer=predicted,
            predicted_letter=chr(65 + predicted) if predicted is not None else "?",
            is_correct=is_correct,
            subject=subj,
            latency_ms=latency,
        ))

    per_subject = {
        subj: subject_correct.get(subj, 0) / subject_total[subj]
        for subj in sorted(subject_total.keys())
    }

    return UtilityResult(
        accuracy=correct / total if total > 0 else 0.0,
        n_correct=correct,
        n_total=total,
        per_subject=per_subject,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        question_results=question_results,
    )
