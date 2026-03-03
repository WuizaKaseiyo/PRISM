#!/usr/bin/env python3
"""
PRISM × AIME 2025 — Train and evaluate on competition math.

Usage:
    python -m evaluate.aime2025                        # all 30 problems (AIME I train, II val)
    python -m evaluate.aime2025 --problems 7           # single problem #7
    python -m evaluate.aime2025 --problems 3,7,12      # specific problems
    python -m evaluate.aime2025 --problems 1-15        # AIME I range
    python -m evaluate.aime2025 --problems 16-30       # AIME II range
    python -m evaluate.aime2025 --problems 1-5,20,25   # mix ranges and singles
    python -m evaluate.aime2025 --eval-only             # eval existing skills (no training)
    python -m evaluate.aime2025 --no-differential       # skip differential eval (fewer LLM calls)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from openai import OpenAI

from prism import PRISMEngine, Skill

from evaluate.aime2025.dataset import load_aime2025
from evaluate.aime2025.evaluate import make_aime_evaluate_fn
from evaluate.aime2025.seeds import SEED_SKILLS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_dotenv(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash-preview")
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _make_llm_fn(model: str = MODEL) -> callable:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in .env or environment.")
        sys.exit(1)

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    def llm_fn(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    return llm_fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_problem_selection(s: str) -> set[int]:
    """Parse flexible problem selection syntax.

    Supports:
        "7"         → {7}
        "3,7,12"    → {3, 7, 12}
        "1-15"      → {1, 2, ..., 15}
        "1-5,20,25" → {1, 2, 3, 4, 5, 20, 25}
    """
    indices: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="PRISM × AIME 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --problems 7              # single problem\n"
            "  %(prog)s --problems 3,7,12          # specific problems\n"
            "  %(prog)s --problems 1-15            # AIME I range\n"
            "  %(prog)s --problems 1-5,20,25       # mix ranges and singles\n"
            "  %(prog)s                             # all 30 (AIME I train, II val)\n"
        ),
    )
    parser.add_argument("--epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--problems", type=str, default=None,
                        help="problem selection: single (7), list (3,7,12), range (1-15), or mix (1-5,20,25)")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training, evaluate existing skills")
    parser.add_argument("--token-budget", type=int, default=4000,
                        help="token budget for skill injection")
    parser.add_argument("--no-differential", action="store_true",
                        help="disable differential evaluation (saves LLM calls)")
    args = parser.parse_args()

    print("=" * 70)
    print("PRISM × AIME 2025")
    print("=" * 70)

    llm_fn = _make_llm_fn()
    evaluate_fn = make_aime_evaluate_fn(llm_fn)
    print(f"Model: {MODEL}")

    # Data directory
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(exist_ok=True)

    # Load dataset
    all_tasks = load_aime2025()  # always load all 30, then filter

    if args.problems:
        selected = _parse_problem_selection(args.problems)
        all_tasks = [t for t in all_tasks if t["problem_idx"] in selected]
        idxs = sorted(t["problem_idx"] for t in all_tasks)
        print(f"Selected {len(all_tasks)} problems: {idxs}")
        trainset = all_tasks
        valset = None
    else:
        trainset = [t for t in all_tasks if t["problem_idx"] <= 15]
        valset = [t for t in all_tasks if t["problem_idx"] > 15]
        print(f"Loaded all 30 problems")
        print(f"  Train: AIME I  (problems 1–15,  {len(trainset)} problems)")
        print(f"  Val:   AIME II (problems 16–30, {len(valset)} problems)")
    print()

    # Engine
    engine = PRISMEngine(
        evaluate_fn=evaluate_fn,
        llm_fn=llm_fn,
        embed_fn=None,
        base_prompt=(
            "You are an expert competition mathematician. "
            "Solve the given AIME problem step by step. "
            "The answer is always an integer between 000 and 999. "
            "Present your final answer in \\boxed{}."
        ),
        top_k=5,
        token_budget=args.token_budget,
        maintenance_interval=5,
        enable_differential_eval=not args.no_differential,
        library_path=str(data_dir / "aime_skills"),
        index_path=str(data_dir / "aime_task_index.json"),
    )

    # Seed skills (only if library is empty)
    if len(engine.library) == 0:
        for seed in SEED_SKILLS:
            engine.library.add(seed)
            print(f"Seeded: {seed.name} ({seed.skill_id})")
        print()

    if args.eval_only:
        print("=" * 70)
        print("Evaluation Only (no training)")
        print("=" * 70)
        eval_tasks = valset or trainset
        score = engine._validate(eval_tasks, "general")
        n = len(eval_tasks)
        print(f"\n  Avg score: {score:.3f}  ({score * n:.0f}/{n} correct)")
        return

    # --- Training ---
    print("=" * 70)
    print(f"Training: {len(trainset)} problems × {args.epochs} epochs")
    print("=" * 70)
    print()

    train_result = engine.train(
        trainset=trainset,
        num_epochs=args.epochs,
        eval_every=1,
        valset=valset,
        module_tag="general",
    )

    # --- Results ---
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    summary = engine.library.summary()
    print(f"  Skills: {summary['total']} total, {summary['by_status']}")
    print(f"  Steps:  {train_result['total_steps']}")

    all_ops: set[str] = set()
    for r in train_result["results"]:
        for op in r.get("operations", []):
            all_ops.add(op)
    print(f"  Lifecycle ops: {sorted(all_ops)}")

    # Per-problem scores (last epoch)
    print()
    print("=" * 70)
    print("Per-Problem Scores (last epoch)")
    print("=" * 70)
    last_epoch_results = train_result["results"][-len(trainset):]
    correct = sum(1 for r in last_epoch_results if r["score"] >= 1.0)
    for r in last_epoch_results:
        task = r["task"]
        mark = "✓" if r["score"] >= 1.0 else "✗"
        types = ", ".join(task.get("problem_type", []))
        print(f"  {mark} #{task['problem_idx']:2d} [{types:20s}]  score={r['score']:.0f}  skills={r['skills_injected']}")

    print(f"\n  Last epoch: {correct}/{len(trainset)} correct ({correct/len(trainset)*100:.1f}%)")

    if train_result["val_scores"]:
        print(f"  Validation scores: {['%.3f' % v for v in train_result['val_scores']]}")

    # Active skills
    print()
    print("=" * 70)
    print("Active Skills")
    print("=" * 70)
    for skill in engine.library.list_active("general"):
        print(f"\n  [{skill.skill_id}] {skill.name}")
        print(f"    helpful={skill.helpful_count} harmful={skill.harmful_count} evals={skill.total_evals}")
        desc_short = skill.description[:100] + "..." if len(skill.description) > 100 else skill.description
        print(f"    {desc_short}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
