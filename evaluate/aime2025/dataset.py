"""AIME 2025 dataset loader from HuggingFace."""
from __future__ import annotations

import pandas as pd

HF_PARQUET_URL = (
    "https://huggingface.co/api/datasets/MathArena/aime_2025"
    "/parquet/default/train/0.parquet"
)


def load_aime2025(problem_range: tuple[int, int] | None = None) -> list[dict]:
    """Load AIME 2025 from HuggingFace as a list of PRISM task dicts.

    Args:
        problem_range: Optional (lo, hi) inclusive range to filter by problem_idx.
                       e.g. (1, 15) for AIME I, (16, 30) for AIME II.

    Returns:
        List of task dicts with keys: question, answer, problem_idx, problem_type, type.
    """
    df = pd.read_parquet(HF_PARQUET_URL)

    if problem_range:
        lo, hi = problem_range
        df = df[(df["problem_idx"] >= lo) & (df["problem_idx"] <= hi)]

    tasks = []
    for _, row in df.iterrows():
        problem_types = row["problem_type"]
        if hasattr(problem_types, "tolist"):
            problem_types = problem_types.tolist()
        tasks.append({
            "question": row["problem"],
            "answer": str(int(row["answer"])),
            "problem_idx": int(row["problem_idx"]),
            "problem_type": problem_types,
            "type": "math",
        })
    return tasks
