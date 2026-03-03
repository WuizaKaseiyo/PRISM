"""AIME 2025 benchmark for PRISM.

Dataset: https://huggingface.co/datasets/MathArena/aime_2025
30 competition-level problems (AIME I + II), integer answers 000–999.
"""

from evaluate.aime2025.dataset import load_aime2025
from evaluate.aime2025.evaluate import extract_answer_integer, make_aime_evaluate_fn
from evaluate.aime2025.seeds import SEED_SKILLS

__all__ = [
    "load_aime2025",
    "extract_answer_integer",
    "make_aime_evaluate_fn",
    "SEED_SKILLS",
]
