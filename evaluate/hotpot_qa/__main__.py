#!/usr/bin/env python3
"""
PRISM × HotpotQA — Train and evaluate on multi-hop reasoning.

Usage:
    python -m evaluate.hotpot_qa                          # default: 150 train, 50 val
    python -m evaluate.hotpot_qa --n-train 50 --n-val 20  # smaller run
    python -m evaluate.hotpot_qa --difficulty hard         # only hard questions
    python -m evaluate.hotpot_qa --eval-only               # eval existing skills
    python -m evaluate.hotpot_qa --no-differential         # skip differential eval
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from openai import OpenAI

from prism import PRISMEngine, Skill

from evaluate.hotpot_qa.dataset import load_hotpotqa
from evaluate.hotpot_qa.evaluate import make_hotpotqa_evaluate_fn
from evaluate.hotpot_qa.seeds import SEED_SKILLS

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

def main():
    parser = argparse.ArgumentParser(
        description="PRISM × HotpotQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--n-train", type=int, default=150, help="number of training examples")
    parser.add_argument("--n-val", type=int, default=50, help="number of validation examples")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard"],
                        help="filter by difficulty level")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training, evaluate existing skills")
    parser.add_argument("--token-budget", type=int, default=4000,
                        help="token budget for skill injection")
    parser.add_argument("--no-differential", action="store_true",
                        help="disable differential evaluation")
    args = parser.parse_args()

    print("=" * 70)
    print("PRISM × HotpotQA")
    print("=" * 70)

    llm_fn = _make_llm_fn()
    evaluate_fn = make_hotpotqa_evaluate_fn(llm_fn)
    print(f"Model: {MODEL}")

    # Data directory
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(exist_ok=True)

    # Load dataset — use different seeds for train/val to avoid overlap
    total_needed = args.n_train + args.n_val
    all_tasks = load_hotpotqa(n=total_needed, difficulty=args.difficulty, seed=42)

    trainset = all_tasks[:args.n_train]
    valset = all_tasks[args.n_train:args.n_train + args.n_val] if args.n_val > 0 else None

    print(f"Loaded {len(all_tasks)} examples" +
          (f" (difficulty={args.difficulty})" if args.difficulty else ""))
    print(f"  Train: {len(trainset)} examples")
    if valset:
        print(f"  Val:   {len(valset)} examples")
    print()

    # Engine
    engine = PRISMEngine(
        evaluate_fn=evaluate_fn,
        llm_fn=llm_fn,
        embed_fn=None,
        base_prompt=(
            "You are an expert at multi-hop reasoning. "
            "Given several passages of context, answer the question by "
            "identifying and chaining relevant facts across passages. "
            "State your final answer clearly as: The answer is <your answer>."
        ),
        top_k=5,
        token_budget=args.token_budget,
        maintenance_interval=10,
        enable_differential_eval=not args.no_differential,
        library_path=str(data_dir / "hotpotqa_skills"),
        index_path=str(data_dir / "hotpotqa_task_index.json"),
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
        print(f"\n  Avg F1: {score:.3f}")
        return

    # --- Training ---
    print("=" * 70)
    print(f"Training: {len(trainset)} examples × {args.epochs} epochs")
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

    # Per-example scores (last epoch)
    print()
    print("=" * 70)
    print("Per-Example Scores (last epoch, first 30)")
    print("=" * 70)
    last_epoch_results = train_result["results"][-len(trainset):]
    scores = [r["score"] for r in last_epoch_results]
    for i, r in enumerate(last_epoch_results[:30]):
        task = r["task"]
        q_short = task["question"][:60] + ("..." if len(task["question"]) > 60 else "")
        mark = "✓" if r["score"] >= 0.5 else "✗"
        print(f"  {mark} [{task.get('question_type', '?'):10s}] F1={r['score']:.2f}  {q_short}")

    avg = sum(scores) / len(scores) if scores else 0.0
    above_half = sum(1 for s in scores if s >= 0.5)
    print(f"\n  Last epoch avg F1: {avg:.3f}  ({above_half}/{len(scores)} with F1≥0.5)")

    if train_result["val_scores"]:
        print(f"  Validation F1: {['%.3f' % v for v in train_result['val_scores']]}")

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
