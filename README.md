# SCULPT — Skill Curation Using Lifecycle Pareto Tracking

A persistent, self-curating skill library for LLM systems with principled evaluation and lifecycle management.

**Core idea**: Every time an LLM solves a task, SCULPT reflects on what worked, attributes credit to individual skills (even when multiple are co-selected), and automatically creates, refines, splits, merges, or retires reusable "skills" — governed by Pareto-based lifecycle decisions and content quality gates.

## What Makes SCULPT Different

- **Credit attribution**: Solves the set-vs-skill problem — when multiple skills are selected together, attribution-refined scoring (Tier 1) and targeted leave-one-out verification (Tier 2) isolate each skill's causal contribution
- **Pareto-based lifecycle**: Skills are retired via soft ε-domination over per-instance score vectors, not heuristic thresholds — the library converges to a non-redundant Pareto front
- **Dual feedback loop**: Outcome signals (score matrix) drive composition-changing decisions; content quality signals (SkillValidator) drive content improvement — neither alone suffices
- **Frozen weights**: No training required — works with any black-box LLM API

## How It Works

Each task goes through a 5-step loop:

```
task ──→ ① RETRIEVE    5-step pipeline: task-type filter → semantic similarity
                        → Pareto boost → exploration bonus → pool-based selection
         ② EXECUTE     Run augmented prompt π₀ ⊕ Sᵢ on task instance
         ③ REFLECT     Per-skill attributions (helpful/neutral/harmful) + gap diagnoses
         ④ CURATE      6 lifecycle operations (see below)
         ⑤ EVALUATE    Differential eval: δ = μ_new − μ_ref → update score matrix
```

### 6 Skill Lifecycle Operations

| Operation | Trigger | What Happens | Verified? |
|-----------|---------|-------------|-----------|
| **NO-OP** | All skills helpful/neutral, no gaps | Skip curation | — |
| **BIRTH** | Coverage gap: best score < 0.3 on ≥2 instances | LLM creates a new skill | LOO ✓ |
| **ENRICH** | Gap matches existing skill, or quality audit | Append (α > 0) or full rewrite (α ≈ 0) | — |
| **SPECIALIZE** | Pareto freq ≥ 0.4 AND harmful ratio ≥ 0.25 (5+ evals) | LLM splits into 2 focused children | LOO ✓ |
| **GENERALIZE** | Two skills share ≥60% keyword overlap (Jaccard) | Merge into one combined skill | LOO ✓ |
| **RETIRE** | ε-dominated by another skill (ε=0.05, ≥3 shared instances) | Deactivate the skill | LOO ✓ |

Composition-changing operations (BIRTH, RETIRE, SPECIALIZE, GENERALIZE) require targeted leave-one-out verification before committing.

## Quick Start

### 1. Install

```bash
conda create -n sculpt python=3.10 -y
conda activate sculpt
pip install -e .
pip install openai   # for OpenRouter
```

### 2. Configure

Create a `.env` file in the `prism/` directory:

```
OPENROUTER_API_KEY="sk-or-..."
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
OPENROUTER_MODEL="google/gemini-2.5-flash-preview"
```

### 3. Run

```bash
python examples/run_prism.py
```

### 4. Benchmarks

```bash
pip install pandas pyarrow

# AIME 2025 (30 competition math problems)
python -m evaluate.aime2025                        # all 30 problems
python -m evaluate.aime2025 --problems 7           # single problem
python -m evaluate.aime2025 --problems 1-5,20,25   # mix ranges and singles
python -m evaluate.aime2025 --eval-only            # eval existing skills (no training)

# HotpotQA (multi-hop QA, F1 scoring)
python -m evaluate.hotpot_qa --n-train 150 --n-val 50 --epochs 2
python -m evaluate.hotpot_qa --no-differential     # skip differential eval (faster)
```

## Usage

```python
from prism import PRISMEngine, Skill

def my_evaluate(prompt: str, task: dict) -> tuple[float, str, str]:
    """Your evaluation function. Returns (score, trace, output)."""
    output = call_your_llm(prompt + task["question"])
    score = 1.0 if task["answer"] in output else 0.0
    return score, f"expected={task['answer']}", output

def my_llm(prompt: str) -> str:
    """Any LLM callable: str → str."""
    return call_your_llm(prompt)

engine = PRISMEngine(
    evaluate_fn=my_evaluate,
    llm_fn=my_llm,
    base_prompt="You are a helpful assistant.",
    top_k=5,
    token_budget=2000,
    library_path="data/skills",
    index_path="data/task_index.json",
)

# Optional: seed a skill
engine.library.add(Skill(
    name="My First Skill",
    description="Use when the task involves...",
    content="# My First Skill\n\n## Overview\n...\n\n## Workflow\n1. ...",
    module_tag="general",
    keywords=["..."],
    task_types=["qa"],
))

# Train
result = engine.train(
    trainset=[{"question": "...", "answer": "...", "type": "math"}],
    num_epochs=3,
    module_tag="general",
)

# Single step
result = engine.step({"question": "..."}, module_tag="general")
```

## Architecture

```
prism/
├── prism/
│   ├── __init__.py              # Exports: PRISMEngine, SkillLibrary, Skill
│   ├── engine.py                # 5-step loop + training orchestrator
│   ├── utils.py                 # Shared JSON extraction (3-strategy fallback)
│   ├── skill_library/
│   │   ├── skill.py             # Skill dataclass + SKILL.md serialization
│   │   └── library.py           # Directory-per-skill storage + _meta.json persistence
│   ├── task_index/
│   │   └── index.py             # task_type → skill_id → score matrix
│   ├── assembler/
│   │   └── assembler.py         # 5-step retrieval (filter→cosine→Pareto boost→explore→select)
│   └── lifecycle/
│       ├── reflector.py         # LLM-based trace analysis → attributions + gaps
│       └── curator.py           # 6 operations: NO-OP/BIRTH/ENRICH/SPECIALIZE/GENERALIZE/RETIRE
├── evaluate/
│   ├── aime2025/                # AIME 2025 benchmark (30 competition math problems)
│   └── hotpot_qa/               # HotpotQA benchmark (multi-hop QA, F1 scoring)
├── examples/
│   └── run_prism.py             # Full demo with OpenRouter
├── pyproject.toml
└── .env                         # OPENROUTER_API_KEY, MODEL, BASE_URL
```

### Skill Storage

```
data/skills/
├── basic-arithmetic/
│   └── SKILL.md                 # name + description frontmatter, rich markdown body
├── exact-format-matching/
│   └── SKILL.md
└── _meta.json                   # All SCULPT tracking data keyed by skill_id
```

### Dependency Layers

```
L0  Skill, utils                  ← zero dependencies
L1  SkillLibrary, TaskTypeIndex   ← uses L0
L2  SkillAssembler, Reflector     ← uses L0-L1
L3  SkillCurator                  ← uses L0-L2
L4  SCULPTEngine                  ← uses L0-L3
```

## Key Interfaces

### evaluate_fn

```python
def evaluate_fn(prompt: str, task: dict) -> tuple[float, str, str]:
    """
    Args:
        prompt: Base prompt augmented with skill context
        task: Task dict (your format, e.g. {"question": ..., "answer": ...})
    Returns:
        score: Float, higher is better (0.0-1.0 recommended)
        trace: String describing execution for reflection
        output: Raw LLM output
    """
```

### llm_fn

```python
def llm_fn(prompt: str) -> str:
    """Any callable that takes a prompt string and returns a response string."""
```

### embed_fn (optional)

```python
def embed_fn(text: str) -> list[float]:
    """Returns embedding vector. Enables semantic retrieval in Step 2."""
```

## Tuning

Edit constants in `prism/lifecycle/curator.py` and `prism/assembler/assembler.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `EPSILON` | 0.05 | Soft ε-domination threshold for RETIRE |
| `MIN_SHARED_INSTANCES` | 3 | Minimum shared tasks to compare two skills |
| `SPECIALIZE_PARETO_FREQ` | 0.40 | Pareto frequency threshold for SPECIALIZE |
| `SPECIALIZE_HARMFUL_RATIO` | 0.25 | Harmful ratio threshold for SPECIALIZE |
| `MIN_EVALS_FOR_LIFECYCLE` | 5 | Minimum evaluations before SPECIALIZE triggers |
| `COVERAGE_SCORE_THRESHOLD` | 0.3 | Instances scoring below this are "uncovered" |
| `MIN_GAP_CLUSTER_SIZE` | 2 | Need ≥2 uncovered instances to trigger BIRTH |
| `MERGE_KEYWORD_OVERLAP` | 0.60 | Lower = merge skills more aggressively |
| `MAX_SKILL_CONTENT_CHARS` | 8000 | Cap on skill content length (~2000 tokens) |
| `EXPLORE_SLOTS` | 1 | Explore slots for untested skills in retrieval |
| `EXPLORE_BONUS_NEW` | 0.6 | Score bonus for unevaluated skills |
| `LIBRARY_MATURITY_THRESHOLD` | 50 | Total evals before library is "mature" |

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Zero required dependencies | numpy/sentence-transformers optional; pure Python cosine fallback |
| Directory-per-skill storage | SKILL.md + centralized _meta.json for tracking |
| 3-strategy JSON extraction | LLM output unreliable: direct parse → code block → brace counting |
| Per-instance score matrix | Enables Pareto-based lifecycle (ε-domination, coverage gaps) |
| Attribution-refined deltas | Solves set-vs-skill credit problem without full LOO cost |
| Two-tier evaluation | Routine scoring is free (Tier 1); LOO only for composition-changing ops (Tier 2) |
| Differential eval is optional | Avoids doubling API cost when not needed |
| Skills have parent/children | Tracks lineage through SPECIALIZE/GENERALIZE operations |
| Exploration bonus for new skills | Prevents cold-start dead loop for newly created skills |
| Graceful degradation everywhere | No embed_fn → skip semantic retrieval. LLM parse failure → log warning, continue |

## Dependencies

**Required**: Python >=3.10 (no packages required)

**Optional**:

| Package | Purpose |
|---------|---------|
| `openai` | OpenRouter / OpenAI API calls |
| `numpy` | Faster cosine similarity |
| `sentence-transformers` | Embedding-based semantic retrieval |
| `pandas`, `pyarrow` | AIME 2025 benchmark dataset |
| `pytest`, `ruff`, `pyright` | Development |

Install optional groups:

```bash
pip install -e ".[embedding]"   # numpy + sentence-transformers
pip install -e ".[dev]"         # pytest + ruff + pyright
```
