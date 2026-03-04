"""Seed skills for HotpotQA multi-hop reasoning."""
from __future__ import annotations

from prism import Skill

SEED_SKILLS = [
    Skill(
        name="Multi-Hop Reasoning Strategy",
        description=(
            "Use when answering questions that require combining information "
            "from multiple paragraphs or documents. The question cannot be "
            "answered from a single passage alone — you must identify, connect, "
            "and synthesize evidence across sources."
        ),
        content="""\
# Multi-Hop Reasoning Strategy

## Overview
Multi-hop questions require chaining facts from 2+ sources to reach the answer.
Unlike single-hop lookup, you must identify which pieces of evidence are relevant,
determine how they connect, and synthesize a final answer.

## Workflow
1. **Parse the question** — identify what is being asked and what entities are involved
2. **Scan all paragraphs** — read every provided passage, noting key facts per entity
3. **Identify the reasoning chain** — determine which facts connect to answer the question
   - Bridge questions: entity in passage A leads to passage B (e.g., "What team does the director of X play for?")
   - Comparison questions: compare attributes across two entities (e.g., "Which film was released first, X or Y?")
4. **Extract supporting facts** — pinpoint the exact sentences that form the evidence chain
5. **Synthesize the answer** — combine the facts to produce a concise, direct answer
6. **Verify** — re-read the evidence chain to confirm logical consistency

## Key Principles
- Always read ALL provided passages before answering — the answer often requires the least obvious paragraph
- For bridge questions: the answer to the first hop is usually an entity name that appears as a title in another paragraph
- For comparison questions: extract the comparable attribute from both entities, then compare directly
- Keep answers concise — typically a name, date, number, or short phrase
- If two passages seem to contradict, check for disambiguation (same name, different entities)

## Common Patterns

### Bridge Questions
"Who is the director of the film that starred [Actor X]?"
→ Hop 1: Find which film starred Actor X → Film Y
→ Hop 2: Find the director of Film Y → Director Z
→ Answer: Director Z

### Comparison Questions
"Are both [X] and [Y] from the same country?"
→ Find country of X → Country A
→ Find country of Y → Country B
→ Compare A and B → Answer yes/no

### "Yes/No" Questions
- Look for comparison or verification patterns
- Extract the specific facts needed, compare, answer yes or no
""",
        module_tag="general",
        keywords=[
            "multi-hop", "reasoning", "qa", "bridge", "comparison",
            "evidence", "synthesis", "reading-comprehension",
        ],
        task_types=["qa"],
    ),
]
