"""AIME-specific evaluation: integer extraction + exact match scoring."""
from __future__ import annotations

import re
from typing import Callable


def extract_answer_integer(text: str) -> str | None:
    r"""Extract the final integer answer from LLM output.

    AIME answers are integers 000–999.  Patterns tried (in priority order):
      1. \boxed{123}           — LaTeX convention, most reliable
      2. "the answer is 123"   — natural language
      3. last standalone int   — on its own line
      4. last integer anywhere — fallback
    """
    # 1. \boxed{...}
    boxed = re.findall(r"\\boxed\{(\d+)\}", text)
    if boxed:
        return boxed[-1]

    # 2. "answer is <int>" / "answer: <int>"
    answer_pattern = re.findall(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*\**(\d+)\**",
        text,
        re.IGNORECASE,
    )
    if answer_pattern:
        return answer_pattern[-1]

    # 3. Last standalone integer on its own line
    last_int = re.findall(r"(?:^|\n)\s*\**(\d{1,3})\**\s*$", text, re.MULTILINE)
    if last_int:
        return last_int[-1]

    # 4. Fallback: last integer anywhere
    all_ints = re.findall(r"\b(\d{1,3})\b", text)
    if all_ints:
        return all_ints[-1]

    return None


def make_aime_evaluate_fn(llm_fn: Callable[[str], str]):
    """Create an AIME-specific evaluate_fn for PRISM.

    Returns:
        evaluate_fn(prompt, task) -> (score, trace, output)
    """

    def evaluate_fn(prompt: str, task: dict) -> tuple[float, str, str]:
        question = task["question"]
        expected = task["answer"]

        full_prompt = (
            f"{prompt}\n\n"
            f"**Problem (AIME 2025 #{task.get('problem_idx', '?')}):**\n\n"
            f"{question}\n\n"
            "Solve this step by step. The answer is an integer between 000 and 999.\n"
            "Put your final answer in \\boxed{{}}."
        )

        output = llm_fn(full_prompt)

        extracted = extract_answer_integer(output)
        score = 1.0 if extracted is not None and int(extracted) == int(expected) else 0.0

        trace = (
            f"AIME #{task.get('problem_idx', '?')} "
            f"[{', '.join(task.get('problem_type', []))}]\n"
            f"Expected: {expected}  Extracted: {extracted}  "
            f"Score: {score}\n"
            f"Output (first 500 chars): {output[:500]}"
        )
        return (score, trace, output)

    return evaluate_fn
