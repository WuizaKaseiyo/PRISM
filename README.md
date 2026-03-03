# PRISM — Self-Evolving Skill Library

PRISM combines [GEPA](https://github.com/gepa-ai/gepa)'s Pareto evolutionary search with [ACE](https://github.com/ace-project/ace)'s incremental knowledge curation into a persistent, self-evolving skill library for LLM systems.

**Core idea**: Every time an LLM solves a task, PRISM reflects on what worked, what didn't, and what knowledge was missing — then automatically creates, refines, splits, merges, or retires reusable "skills" that improve future performance.

## How It Works

Each task goes through a 6-step loop:

```
task ──→ ① RETRIEVE   4-layer skill retrieval (filter → embedding → EMA → LLM)
         ② EXECUTE    Run augmented prompt through evaluate_fn
         ③ REFLECT    LLM analyzes: which skills helped? what's missing?
         ④ CURATE     5 lifecycle operations (see below)
         ⑤ DIFFERENTIAL  Compare score with vs without skills
         ⑥ INDEX      Update EMA scores for future retrieval
```

### 5 Skill Lifecycle Operations

| Operation | Trigger | What Happens |
|-----------|---------|-------------|
| **BIRTH** | Knowledge gap found, no matching skill | LLM creates a new skill |
| **ENRICH** | Gap matches existing skill | Append strategy note (no rewrite) |
| **SPECIALIZE** | Skill has high score variance (>0.20) after 5+ evals | LLM splits into 2 focused children |
| **GENERALIZE** | Two skills share ≥60% keyword overlap (Jaccard) | Merge into one combined skill |
| **RETIRE** | harmful - helpful > 3 | Deactivate the skill |

## Quick Start

### 1. Install

```bash
conda create -n prism python=3.10 -y
conda activate prism
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

Expected output:

```
PRISM Demo — OpenRouter LLM (fully real)
Model: google/gemini-2.5-flash-preview

Seeded skill: Basic Arithmetic (e84095c3fc17)

=== Epoch 1/2 ===
[Step 1] RETRIEVE: 1 skills assembled
[Step 1] EXECUTE: score=1.000
[Step 1] REFLECT: 1 attributions, 4 gaps
[Curator] ENRICH: e84095c3fc17 with note: When solving arithmetic problems...
[Step 1] CURATE: ['ENRICH']
...
[Curator] BIRTH: Response Formatting and Precision (b60e12c421aa)
...
[Curator] SPECIALIZE: e84095c3fc17 → ['cb830e375cf5', 'ec3169c06c4b']
...

Final Library Summary
  Total skills: 10
  Lifecycle operations triggered: ['BIRTH', 'ENRICH', 'SPECIALIZE']
```

## Usage

### Standalone

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
    library_path="skills.json",    # persistence
    index_path="task_index.json",  # persistence
)

# Optional: seed a skill
engine.library.add(Skill(
    name="My First Skill",
    description="...",
    content="...",
    module_tag="general",
    keywords=["..."],
    task_types=["qa"],
))

# Train
result = engine.train(
    trainset=[{"question": "...", "answer": "..."}],
    num_epochs=3,
    module_tag="general",
)

# Single step
result = engine.step({"question": "..."}, module_tag="general")
```

### With GEPA

PRISM wraps any `GEPAAdapter` transparently — GEPA doesn't know PRISM exists:

```python
from prism import SkillLibrary
from prism.task_index import TaskTypeIndex
from prism.gepa_integration import PRISMWrappedAdapter
import gepa

inner_adapter = YourGEPAAdapter(...)
library = SkillLibrary(path="skills.json")
task_index = TaskTypeIndex(path="index.json")

wrapped = PRISMWrappedAdapter(
    inner_adapter=inner_adapter,
    library=library,
    task_index=task_index,
    llm_fn=my_llm,
    module_tag="Generator",
)

result = gepa.optimize(
    seed_candidate={"instructions": "..."},
    trainset=data,
    adapter=wrapped,
    reflection_lm="openai/gpt-4o",
)
```

### With ACE

PRISM replaces ACE's 3-agent system (Generator + Reflector + Curator):

```python
from prism import SkillLibrary
from prism.task_index import TaskTypeIndex
from prism.ace_integration import PRISMACEBridge

library = SkillLibrary(path="skills.json")
task_index = TaskTypeIndex(path="index.json")
bridge = PRISMACEBridge(library, task_index, llm_fn=my_llm)

# Import existing ACE playbook
bridge.import_from_ace_playbook(open("playbook.txt").read())

# Use in place of ACE agents
result = bridge.generate(base_prompt, task)
bridge.reflect_and_curate(task, result, trace, ground_truth)

# Export back to ACE format
bridge.export_to_ace_playbook("updated_playbook.txt")
```

## Architecture

```
prism/
├── prism/
│   ├── __init__.py              # Exports: PRISMEngine, SkillLibrary, Skill
│   ├── engine.py                # 6-step loop + training orchestrator
│   ├── utils.py                 # Shared JSON extraction (3-strategy fallback)
│   ├── skill_library/
│   │   ├── skill.py             # Skill dataclass (20 fields)
│   │   └── library.py           # In-memory store + JSON persistence
│   ├── task_index/
│   │   └── index.py             # task_type → skill_id → EMA score
│   ├── assembler/
│   │   └── assembler.py         # 4-layer retrieval (filter→cosine→EMA→LLM)
│   ├── lifecycle/
│   │   ├── reflector.py         # LLM-based trace analysis → attributions + gaps
│   │   └── curator.py           # 5 operations: BIRTH/ENRICH/SPECIALIZE/GENERALIZE/RETIRE
│   ├── gepa_integration/
│   │   └── adapter.py           # Wraps GEPAAdapter, injects skills into evaluate()
│   └── ace_integration/
│       └── bridge.py            # Replaces ACE Generator+Reflector+Curator
├── examples/
│   └── run_prism.py             # Full demo with OpenRouter
├── pyproject.toml
├── requirements.txt
└── .env                         # OPENROUTER_API_KEY, MODEL, BASE_URL
```

### Dependency Layers

```
L0  Skill, utils                  ← zero dependencies
L1  SkillLibrary, TaskTypeIndex   ← uses L0
L2  SkillAssembler, PRISMReflector ← uses L0-L1
L3  SkillCurator                  ← uses L0-L2
L4  PRISMEngine                   ← uses L0-L3
L5  PRISMWrappedAdapter, PRISMACEBridge ← uses L0-L3
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
    """Returns embedding vector. Enables Layer 2 semantic retrieval."""
```

## Tuning

Edit constants in `prism/lifecycle/curator.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `RETIRE_GAP` | 3 | Higher = more tolerant of harmful skills |
| `SPECIALIZE_VARIANCE` | 0.20 | Lower = split skills more aggressively |
| `MIN_EVALS_FOR_LIFECYCLE` | 5 | Minimum evaluations before SPECIALIZE triggers |
| `MERGE_KEYWORD_OVERLAP` | 0.60 | Lower = merge skills more aggressively |

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Zero required dependencies | numpy/sentence-transformers optional; pure Python cosine fallback |
| JSON persistence | Simple, debuggable, git-friendly. Adequate for <1000 skills |
| 3-strategy JSON extraction | LLM output unreliable: direct parse → code block → brace counting |
| EMA scoring (alpha=0.3) | Remembers history while responding quickly to changes |
| Differential eval is optional | Avoids doubling API cost when not needed |
| Skills have parent/children | Tracks lineage through SPECIALIZE operations |
| Graceful degradation everywhere | No embed_fn → skip Layer 2. LLM parse failure → log warning, continue |

## Dependencies

**Required**: Python >=3.10 (no packages required)

**Optional**:

| Package | Purpose |
|---------|---------|
| `openai` | OpenRouter / OpenAI API calls |
| `numpy` | Faster cosine similarity |
| `sentence-transformers` | Layer 2 embedding retrieval |
| `gepa` | GEPA integration |
| `pytest`, `ruff`, `pyright` | Development |

Install optional groups:

```bash
pip install -e ".[embedding]"   # numpy + sentence-transformers
pip install -e ".[dev]"         # pytest + ruff + pyright
```
