#!/usr/bin/env python3
"""
PRISM Demo — Full 6-step loop with OpenRouter LLM
All components (evaluate, reflect, curate, assemble) use real LLM calls.
"""
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from prism import PRISMEngine, Skill, SkillLibrary

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load .env from prism project root
# ---------------------------------------------------------------------------

def load_dotenv(path: str | Path) -> None:
    """Minimal .env loader — no external dependency."""
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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# OpenRouter LLM
# ---------------------------------------------------------------------------

MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash-preview")
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def make_llm_fn(model: str = MODEL) -> callable:
    """Create an llm_fn(prompt) -> str backed by OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in .env or environment.")
        print("  echo 'OPENROUTER_API_KEY=sk-or-...' > prism/.env")
        sys.exit(1)

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )

    def llm_fn(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    return llm_fn


# ---------------------------------------------------------------------------
# evaluate_fn: LLM solves the problem, then score against ground truth
# ---------------------------------------------------------------------------

def make_evaluate_fn(llm_fn):
    """Create an evaluate_fn that uses the LLM to solve tasks and scores the output."""

    def evaluate_fn(prompt: str, task: dict) -> tuple[float, str, str]:
        question = task.get("question", "")
        answer = task.get("answer", "")

        # Ask LLM to solve the problem
        full_prompt = f"{prompt}\n\nQuestion: {question}\n\nShow your reasoning step by step, then give your final answer."
        output = llm_fn(full_prompt)

        # Score: check if the ground truth answer appears in the LLM output
        score = 1.0 if answer and answer.lower() in output.lower() else 0.0

        trace = f"Question: {question}\nExpected: {answer}\nGot: {output[:300]}"
        return (score, trace, output)

    return evaluate_fn


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PRISM Demo — OpenRouter LLM (fully real)")
    print("=" * 60)

    llm_fn = make_llm_fn()
    evaluate_fn = make_evaluate_fn(llm_fn)
    print(f"Model: {MODEL}")
    print()

    # Persistent storage
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Create engine
    engine = PRISMEngine(
        evaluate_fn=evaluate_fn,
        llm_fn=llm_fn,
        embed_fn=None,
        base_prompt="You are a math tutor. Solve the given problem step by step.",
        top_k=5,
        token_budget=2000,
        maintenance_interval=5,
        enable_differential_eval=True,
        library_path=str(data_dir / "skills"),
        index_path=str(data_dir / "task_index.json"),
    )

    # Clean up old flat-file data if present
    old_skills_dir = data_dir / "skills"
    if old_skills_dir.exists():
        # Remove legacy flat .md files (not subdirectories)
        for f in old_skills_dir.glob("*.md"):
            f.unlink()

    # Seed one skill (Claude Code format: rich markdown body with sections)
    seed_skill = Skill(
        name="Basic Arithmetic",
        description="Use when the task involves basic arithmetic operations such as addition, subtraction, multiplication, or division of numbers.",
        content=(
            "# Basic Arithmetic\n"
            "\n"
            "## Overview\n"
            "Strategy for solving arithmetic problems by identifying the operation,\n"
            "applying it step by step, and verifying the result.\n"
            "\n"
            "## Workflow\n"
            "1. **Read the problem** — identify the numbers and the operation(s) required\n"
            "2. **Plan the steps** — break compound expressions into single operations\n"
            "3. **Execute each step** — perform the operation, showing intermediate results\n"
            "4. **Verify** — use the inverse operation or estimation to confirm the answer\n"
            "\n"
            "## Key Principles\n"
            "- Follow order of operations (PEMDAS/BODMAS) for compound expressions\n"
            "- Show work for each intermediate step so errors are easy to spot\n"
            "- For division, check by multiplying the quotient by the divisor\n"
            "- For subtraction, check by adding the difference to the subtrahend\n"
            "\n"
            "## Examples\n"
            "\n"
            "### Addition\n"
            "```\n"
            "15 + 27 = 42\n"
            "Check: 42 - 27 = 15 ✓\n"
            "```\n"
            "\n"
            "### Multi-step\n"
            "```\n"
            "3 × (4 + 5) − 2\n"
            "  = 3 × 9 − 2\n"
            "  = 27 − 2\n"
            "  = 25\n"
            "```"
        ),
        module_tag="general",
        keywords=["arithmetic", "add", "subtract", "multiply", "divide", "calculate"],
        task_types=["math"],
    )
    engine.library.add(seed_skill)
    print(f"Seeded skill: {seed_skill.name} ({seed_skill.skill_id})")
    print()

    # Training tasks
    trainset = [
        {"question": "What is 15 + 27?", "answer": "42", "type": "math"},
        {"question": "Solve: 3x + 7 = 22", "answer": "x=5", "type": "math"},
        {"question": "Calculate the area of a triangle with base 10 and height 6", "answer": "30", "type": "math"},
        {"question": "Hard: Find the derivative of x^3 + 2x^2 - 5x + 3", "answer": "3x^2 + 4x - 5", "type": "math"},
        {"question": "What is 144 / 12?", "answer": "12", "type": "math"},
    ]

    print("=" * 60)
    print("Training: 5 tasks x 2 epochs")
    print("=" * 60)
    print()

    train_result = engine.train(
        trainset=trainset,
        num_epochs=2,
        eval_every=1,
        module_tag="general",
    )

    # --- Summary ---
    print()
    print("=" * 60)
    print("Final Library Summary")
    print("=" * 60)

    summary = engine.library.summary()
    print(f"  Total skills: {summary['total']}")
    print(f"  By status: {summary['by_status']}")
    print(f"  By module: {summary['by_module']}")
    print(f"  Total training steps: {train_result['total_steps']}")
    print()

    all_ops: set[str] = set()
    for r in train_result["results"]:
        for op in r.get("operations", []):
            all_ops.add(op)
    print(f"  Lifecycle operations triggered: {sorted(all_ops)}")

    # Show learned skills
    print()
    print("=" * 60)
    print("Active Skills")
    print("=" * 60)
    for skill in engine.library.list_active("general"):
        print(f"\n  [{skill.skill_id}] {skill.name}")
        print(f"    helpful={skill.helpful_count} harmful={skill.harmful_count} evals={skill.total_evals}")
        print(f"    {skill.content[:120]}...")

    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()
