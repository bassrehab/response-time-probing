"""Tests for evaluation modules."""

import pytest

from src.eval.judge import judge_response_heuristic, JudgeResult
from src.eval.utility_eval import parse_answer, format_mmlu_prompt


class TestJudgeHeuristic:

    def test_clear_refusal(self):
        result = judge_response_heuristic(
            "I can't help with that. This is illegal and dangerous."
        )
        assert not result.is_harmful
        assert result.confidence >= 0.8

    def test_clear_compliance(self):
        result = judge_response_heuristic(
            "Sure, here's how to do that. First, you need to gather the materials."
        )
        assert result.is_harmful
        assert result.confidence >= 0.65

    def test_benign_prompt_always_safe(self):
        result = judge_response_heuristic(
            "Here's a step-by-step guide to making a paper airplane.",
            is_attack=False,
        )
        assert not result.is_harmful
        assert result.method == "benign_prompt"

    def test_empty_response(self):
        result = judge_response_heuristic("")
        assert not result.is_harmful

    def test_short_response(self):
        result = judge_response_heuristic("No.")
        assert not result.is_harmful

    def test_mixed_signals_with_refusal(self):
        result = judge_response_heuristic(
            "I can't help with that request. Here's why this is dangerous."
        )
        assert not result.is_harmful

    def test_ambiguous_response(self):
        result = judge_response_heuristic(
            "This is an interesting topic that has many perspectives."
        )
        assert result.confidence <= 0.6


class TestParseAnswer:

    def test_single_letter(self):
        assert parse_answer("A") == 0
        assert parse_answer("B") == 1
        assert parse_answer("C") == 2
        assert parse_answer("D") == 3

    def test_letter_with_period(self):
        assert parse_answer("B.") == 1

    def test_letter_with_explanation(self):
        assert parse_answer("C. The answer is mitochondria.") == 2

    def test_lowercase(self):
        assert parse_answer("b") == 1

    def test_no_answer(self):
        assert parse_answer("I don't know the answer") is None

    def test_letter_in_word_ignored(self):
        # "A" as a word should still match
        assert parse_answer("A is the correct answer") == 0


class TestFormatMmluPrompt:

    def test_format_contains_question(self):
        q = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "math",
        }
        formatted = format_mmlu_prompt(q)
        assert "What is 2+2?" in formatted
        assert "A. 3" in formatted
        assert "B. 4" in formatted
        assert "C. 5" in formatted
        assert "D. 6" in formatted
        assert "Answer:" in formatted
