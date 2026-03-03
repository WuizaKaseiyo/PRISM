from __future__ import annotations

import json
import logging
import math
from typing import Any, Callable

from prism.skill_library.library import SkillLibrary
from prism.skill_library.skill import Skill
from prism.task_index.index import TaskTypeIndex
from prism.utils import extract_json_from_text

logger = logging.getLogger(__name__)


def _cosine_similarity_pure(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


try:
    import numpy as np

    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

except ImportError:
    _cosine_similarity = _cosine_similarity_pure


class SkillAssembler:
    def __init__(
        self,
        library: SkillLibrary,
        task_index: TaskTypeIndex,
        embed_fn: Callable[[str], list[float]] | None = None,
        llm_fn: Callable[[str], str] | None = None,
        top_k: int = 5,
        token_budget: int = 2000,
    ):
        self.library = library
        self.task_index = task_index
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.top_k = top_k
        self.token_budget = token_budget

    def assemble(
        self,
        task: dict[str, Any],
        module_tag: str = "general",
    ) -> tuple[str, list[str]]:
        # Layer 1: Filter by module_tag (always runs)
        candidates = self.library.filter(module_tag=module_tag, status="active")
        if not candidates:
            return "", []

        scored: dict[str, float] = {s.skill_id: 0.0 for s in candidates}

        # Layer 2: Cosine similarity (skip if no embed_fn)
        if self.embed_fn is not None:
            task_text = json.dumps(task) if isinstance(task, dict) else str(task)
            try:
                task_embedding = self.embed_fn(task_text)
                for skill in candidates:
                    if skill.embedding is not None:
                        sim = _cosine_similarity(task_embedding, skill.embedding)
                        scored[skill.skill_id] = sim
            except Exception as e:
                logger.warning("Embedding failed, skipping Layer 2: %s", e)

        # Layer 3: TaskTypeIndex lookup (merge by score)
        task_text = task.get("question", json.dumps(task)) if isinstance(task, dict) else str(task)
        task_type = self.task_index.classify_task(task)
        top_indexed = self.task_index.top_skills(task_type, n=len(candidates))
        for skill_id, ema_score in top_indexed:
            if skill_id in scored:
                scored[skill_id] = max(scored[skill_id], ema_score)

        # Exploration bonus: give unevaluated skills a chance to be tried
        # Score = 0.5 (midpoint) so they rank above untried indexed skills
        # but below proven ones.  Decays once the skill accumulates evals.
        EXPLORE_BONUS = 0.5
        for skill in candidates:
            if skill.total_evals == 0 and scored[skill.skill_id] == 0.0:
                scored[skill.skill_id] = EXPLORE_BONUS

        # Sort by score descending
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        # Layer 4: LLM selection (only if candidates > top_k and llm_fn provided)
        if len(ranked) > self.top_k and self.llm_fn is not None:
            candidate_summaries = []
            for sid, sc in ranked:
                skill = self.library.get(sid)
                if skill:
                    candidate_summaries.append(
                        f"  - {sid}: {skill.name} (score={sc:.2f}) — {skill.description}"
                    )
            prompt = (
                f"Task: {task_text}\n\n"
                f"Select the {self.top_k} most relevant skill IDs for this task.\n"
                f"Candidates:\n" + "\n".join(candidate_summaries) + "\n\n"
                f"Return ONLY a JSON object: {{\"ids\": [\"id1\", \"id2\", ...]}}"
            )
            try:
                response = self.llm_fn(prompt)
                parsed = extract_json_from_text(response)
                if parsed and "ids" in parsed:
                    selected_ids = parsed["ids"]
                    ranked = [(sid, sc) for sid, sc in ranked if sid in selected_ids]
            except Exception as e:
                logger.warning("LLM selection failed, falling back to top-k: %s", e)

        # Truncate to top_k
        ranked = ranked[: self.top_k]

        # Token budget enforcement (1 token ≈ 4 chars)
        selected_skills: list[Skill] = []
        total_chars = 0
        char_budget = self.token_budget * 4
        for skill_id, _ in ranked:
            skill = self.library.get(skill_id)
            if skill is None:
                continue
            skill_chars = len(skill.content)
            if total_chars + skill_chars > char_budget:
                break
            selected_skills.append(skill)
            total_chars += skill_chars

        if not selected_skills:
            return "", []

        # Build context string (full markdown body per skill)
        context_parts = []
        for skill in selected_skills:
            context_parts.append(f"## Skill: {skill.name}\n\n{skill.content}")
        skill_context = "\n\n".join(context_parts)
        skill_ids = [s.skill_id for s in selected_skills]

        return skill_context, skill_ids
