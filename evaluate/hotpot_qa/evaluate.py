"""HotpotQA evaluation: F1 score between predicted and gold answer."""
from __future__ import annotations

import re
import string
from typing import Callable


def _normalize(text: str) -> str:
    """Lowercase, strip articles/punctuation/whitespace."""
    text = text.lower()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = " ".join(text.split())
    return text


def _f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def extract_answer(text: str) -> str:
    """Extract the final answer from LLM output.

    Looks for patterns like:
      - "the answer is ..."
      - "**Answer:** ..."
      - "Answer: ..."
    Falls back to the last non-empty line.
    """
    # Pattern 1: "the answer is X" / "**Answer:** X"
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?\*{0,2}answer\*{0,2}\s*(?:is|:)\s*(.+?)(?:\.|$)",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().strip("*").strip().strip('"').strip("'")

    # Pattern 2: "**Answer:** X" or "Answer: X"
    match = re.search(r"\*{0,2}Answer\*{0,2}\s*:?\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        ans = match.group(1).strip().strip("*").strip('"').strip("'")
        if ans:
            return ans

    # Fallback: last non-empty line
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def make_hotpotqa_evaluate_fn(llm_fn: Callable[[str], str]):
    """Create a HotpotQA evaluate_fn for PRISM.

    Returns:
        evaluate_fn(prompt, task) -> (score, trace, output)
    """

    def evaluate_fn(prompt: str, task: dict) -> tuple[float, str, str]:
        question = task["question"]
        context = task.get("context", "")
        expected = task["answer"]

        full_prompt = (
            f"{prompt}\n\n"
            f"## Context\n\n{context}\n\n"
            f"## Question\n\n{question}\n\n"
            "Answer the question based on the context above. "
            "Think step by step, then state your final answer clearly as: "
            '"The answer is <your answer>."'
        )

        output = llm_fn(full_prompt)
        extracted = extract_answer(output)

        f1 = _f1_score(extracted, expected)
        em = _exact_match(extracted, expected)
        # Use F1 as the primary score (more granular than EM)
        score = f1

        trace = (
            f"HotpotQA [{task.get('question_type', '?')}, {task.get('level', '?')}]\n"
            f"Expected: {expected}\n"
            f"Extracted: {extracted}\n"
            f"F1: {f1:.3f}  EM: {em:.0f}\n"
            f"Output (first 500 chars): {output[:500]}"
        )
        return (score, trace, output)

    return evaluate_fn
