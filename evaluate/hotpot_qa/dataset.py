"""HotpotQA dataset loader from HuggingFace parquet."""
from __future__ import annotations

import pandas as pd

# HotpotQA distractor setting — validation split (7,405 examples)
HF_PARQUET_URL = (
    "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa"
    "/parquet/distractor/validation/0.parquet"
)


def _build_context(row) -> str:
    """Flatten context paragraphs into a single string."""
    titles = row.get("context", {}).get("title", [])
    sentences_list = row.get("context", {}).get("sentences", [])
    parts = []
    for title, sentences in zip(titles, sentences_list):
        parts.append(f"### {title}")
        parts.append("".join(sentences))
    return "\n\n".join(parts)


def load_hotpotqa(
    n: int | None = None,
    difficulty: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load HotpotQA distractor-setting validation split.

    Args:
        n: Number of examples to sample. None = all.
        difficulty: Filter by level: "easy", "medium", "hard". None = all.
        seed: Random seed for sampling.

    Returns:
        List of task dicts with keys:
            question, answer, context, question_type, level, type
    """
    df = pd.read_parquet(HF_PARQUET_URL)

    if difficulty:
        df = df[df["level"] == difficulty]

    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)

    tasks = []
    for idx, row in df.iterrows():
        context = _build_context(row)
        tasks.append({
            "question": row["question"],
            "answer": row["answer"],
            "context": context,
            "question_type": row.get("type", "unknown"),
            "level": row.get("level", "unknown"),
            "type": "qa",
        })
    return tasks
