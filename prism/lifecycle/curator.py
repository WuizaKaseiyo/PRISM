from __future__ import annotations

import json
import logging
from typing import Any, Callable

from prism.lifecycle.reflector import ReflectionResult
from prism.skill_library.library import SkillLibrary
from prism.skill_library.skill import Skill
from prism.utils import extract_json_from_text

logger = logging.getLogger(__name__)

# Lifecycle thresholds (legacy kept for backward compat)
RETIRE_GAP = 3
SPECIALIZE_VARIANCE = 0.20
MIN_EVALS_FOR_LIFECYCLE = 5
MERGE_KEYWORD_OVERLAP = 0.60
MAX_SKILL_CONTENT_CHARS = 8000  # ~2000 tokens; cap ENRICH growth

# Pareto-based thresholds
EPSILON = 0.05                    # soft domination threshold
MIN_SHARED_INSTANCES = 3          # minimum shared tasks to compare two skills
SPECIALIZE_PARETO_FREQ = 0.4     # must be Pareto-optimal >=40% of the time
SPECIALIZE_HARMFUL_RATIO = 0.25  # AND harmful ratio >=25%
COVERAGE_SCORE_THRESHOLD = 0.3   # instances below this are "uncovered"
MIN_GAP_CLUSTER_SIZE = 2         # need >=2 uncovered instances for BIRTH


BIRTH_OR_ENRICH_PROMPT = """Given these knowledge gaps and existing active skills, decide whether to CREATE a new skill or ENRICH an existing one.

Gaps identified:
{gaps}

Active skills:
{active_skills}

If a gap closely matches an existing skill, return ENRICH with additional content to append.
If the gap represents genuinely new knowledge, return CREATE with full skill details.

For CREATE, generate a skill in Claude Code format:
- "name": human-readable skill name (e.g. "Exact Format Matching")
- "description": trigger condition starting with "Use when..." — be specific and pushy
- "content": a FULL markdown document (50-200 lines) with:
  - `# Title` heading
  - `## Overview` section explaining what this skill does
  - `## Workflow` section with numbered steps
  - `## Examples` or `## Key Principles` sections as appropriate
  - Code examples in fenced blocks where applicable
  Do NOT write a one-liner. Write substantial operational instructions.

For ENRICH, provide a markdown section to append (with ## heading and content).

Return ONLY a JSON object:
{{
  "action": "CREATE" or "ENRICH",
  "skill_id": "existing skill ID (only for ENRICH)",
  "name": "skill name (only for CREATE)",
  "description": "Use when... trigger condition (only for CREATE)",
  "content": "full markdown body with # headings (only for CREATE)",
  "enrich_content": "## Section Heading\\nmarkdown content to append (only for ENRICH)",
  "keywords": ["keyword1", "keyword2"],
  "task_types": ["type1"]
}}"""

SPECIALIZE_PROMPT = """This skill is Pareto-optimal on some tasks (useful) but harmful on others, suggesting it should be split into specialized children.

Skill: {skill_name}
Content: {skill_content}
Description: {skill_description}
Pareto frequency: {pareto_frequency:.2f} (fraction of instances where skill is on the Pareto front)
Harmful ratio: {harmful_ratio:.2f}
Eval scores: {scores}

Split this into 2 more specialized skills. Each child should handle a specific subset of use cases.

Each child must be a FULL Claude Code format skill:
- "name": human-readable skill name (e.g. "Linear Equation Solving")
- "description": trigger condition starting with "Use when..." — be specific about when this child applies
- "content": a FULL markdown document (50-200 lines) with:
  - `# Title` heading
  - `## Overview` section
  - `## Workflow` section with numbered steps
  - `## Examples` or `## Key Principles` sections as appropriate
  Do NOT write a one-liner. Write substantial operational instructions.

Return ONLY a JSON object:
{{
  "children": [
    {{
      "name": "Specialized Name 1",
      "description": "Use when the task involves...",
      "content": "# Specialized Name 1\\n\\n## Overview\\n...\\n\\n## Workflow\\n1. ...\\n2. ...",
      "keywords": ["kw1"],
      "task_types": ["type1"]
    }},
    {{
      "name": "Specialized Name 2",
      "description": "Use when the task involves...",
      "content": "# Specialized Name 2\\n\\n## Overview\\n...\\n\\n## Workflow\\n1. ...\\n2. ...",
      "keywords": ["kw1"],
      "task_types": ["type1"]
    }}
  ]
}}"""


def _epsilon_dominates(a: Skill, b: Skill, eps: float = EPSILON) -> bool:
    """Return True if skill A epsilon-dominates skill B.

    A eps-dominates B iff:
    - They share >= MIN_SHARED_INSTANCES task instances
    - A >= B - eps on ALL shared instances
    - A > B on at least one shared instance
    """
    shared_keys = set(a.score_matrix) & set(b.score_matrix)
    if len(shared_keys) < MIN_SHARED_INSTANCES:
        return False

    strictly_better_on_one = False
    for key in shared_keys:
        if a.score_matrix[key] < b.score_matrix[key] - eps:
            return False
        if a.score_matrix[key] > b.score_matrix[key]:
            strictly_better_on_one = True

    return strictly_better_on_one


def compute_pareto_front(skills: list[Skill]) -> set[str]:
    """Return set of skill_ids that are Pareto-optimal (not eps-dominated by any other)."""
    pareto: set[str] = set()
    for skill in skills:
        if not skill.score_matrix:
            # No data — can't be dominated, include in front
            pareto.add(skill.skill_id)
            continue
        dominated = False
        for other in skills:
            if other.skill_id == skill.skill_id:
                continue
            if _epsilon_dominates(other, skill):
                dominated = True
                break
        if not dominated:
            pareto.add(skill.skill_id)
    return pareto


def update_pareto_frequencies(skills: list[Skill]) -> None:
    """Update pareto_frequency for each skill based on per-instance non-domination.

    For each task key, skills within EPSILON of the best score are "non-dominated"
    on that instance.  pareto_frequency = count_non_dominated / count_present.
    """
    # Collect all task keys across all skills
    all_keys: set[str] = set()
    for skill in skills:
        all_keys.update(skill.score_matrix.keys())

    if not all_keys:
        return

    # Per-skill tracking
    present_count: dict[str, int] = {s.skill_id: 0 for s in skills}
    nondom_count: dict[str, int] = {s.skill_id: 0 for s in skills}

    for key in all_keys:
        # Find skills present on this instance and the best score
        present_skills: list[Skill] = []
        best_score = float("-inf")
        for skill in skills:
            if key in skill.score_matrix:
                present_skills.append(skill)
                if skill.score_matrix[key] > best_score:
                    best_score = skill.score_matrix[key]

        for skill in present_skills:
            present_count[skill.skill_id] += 1
            if skill.score_matrix[key] >= best_score - EPSILON:
                nondom_count[skill.skill_id] += 1

    for skill in skills:
        if present_count[skill.skill_id] > 0:
            skill.pareto_frequency = nondom_count[skill.skill_id] / present_count[skill.skill_id]
        else:
            skill.pareto_frequency = 0.0


def _find_coverage_gaps(skills: list[Skill]) -> list[str]:
    """Find task instances where all skills perform poorly.

    Returns gap descriptions for BIRTH prompt if >= MIN_GAP_CLUSTER_SIZE uncovered instances.
    """
    # Collect all task keys and best score per key
    best_scores: dict[str, float] = {}
    for skill in skills:
        for key, score in skill.score_matrix.items():
            if key not in best_scores or score > best_scores[key]:
                best_scores[key] = score

    uncovered = [key for key, score in best_scores.items() if score < COVERAGE_SCORE_THRESHOLD]

    if len(uncovered) >= MIN_GAP_CLUSTER_SIZE:
        return [
            f"Coverage gap: {len(uncovered)} task instances where best skill score < {COVERAGE_SCORE_THRESHOLD} "
            f"(task keys: {', '.join(uncovered[:5])}{'...' if len(uncovered) > 5 else ''})"
        ]
    return []


class SkillCurator:
    def __init__(
        self,
        library: SkillLibrary,
        llm_fn: Callable[[str], str],
    ):
        self.library = library
        self.llm_fn = llm_fn

    def curate(
        self,
        reflection: ReflectionResult,
        module_tag: str = "general",
        task_type: str = "general",
    ) -> list[str]:
        """Run all applicable lifecycle operations. Returns list of operation names performed."""
        ops_performed: list[str] = []

        # Update attribution counts
        for attr in reflection.attributions:
            skill = self.library.get(attr.get("skill_id", ""))
            if skill is None:
                continue
            tag = attr.get("tag", "neutral")
            if tag == "helpful":
                skill.helpful_count += 1
            elif tag == "harmful":
                skill.harmful_count += 1
            else:
                skill.neutral_count += 1

        # Update Pareto frequencies for all active skills
        active = self.library.list_active(module_tag)
        update_pareto_frequencies(active)

        # RETIRE: soft domination (with fallback to harmful-ratio heuristic)
        retired = self._retire(module_tag)
        if retired:
            ops_performed.append("RETIRE")

        # BIRTH or ENRICH from gaps (including coverage gap detection)
        all_gaps = list(reflection.gaps)
        coverage_gaps = _find_coverage_gaps(active)
        if coverage_gaps:
            all_gaps.extend(coverage_gaps)
        if all_gaps:
            action = self._birth_or_enrich(all_gaps, module_tag, task_type)
            if action:
                ops_performed.append(action)

        # SPECIALIZE: Pareto-optimal + harmful → split into specialized children
        specialized = self._specialize(module_tag)
        if specialized:
            ops_performed.append("SPECIALIZE")

        # GENERALIZE: merge overlapping skills (unchanged)
        generalized = self._generalize(module_tag)
        if generalized:
            ops_performed.append("GENERALIZE")

        return ops_performed

    def _birth(
        self,
        gap: str,
        module_tag: str,
        task_type: str,
        name: str,
        description: str,
        content: str,
        keywords: list[str] | None = None,
        task_types: list[str] | None = None,
    ) -> Skill:
        skill = Skill(
            name=name,
            description=description,
            content=content,
            module_tag=module_tag,
            keywords=keywords or [],
            task_types=task_types or [task_type],
        )
        self.library.add(skill)
        logger.info("[Curator] BIRTH: %s (%s)", skill.name, skill.skill_id)
        return skill

    def _enrich(self, skill_id: str, enrich_content: str) -> bool:
        skill = self.library.get(skill_id)
        if skill is None:
            return False
        if not enrich_content.strip():
            return False

        new_section = enrich_content.strip()
        proposed = skill.content.rstrip() + "\n\n" + new_section + "\n"

        if len(proposed) <= MAX_SKILL_CONTENT_CHARS:
            # Fits within budget — append normally
            skill.content = proposed
        else:
            # Over budget — drop the oldest enrichment section to make room.
            # Enrichment sections are ## headings added after the original body.
            # Find all ## sections, keep the core (everything up to the first
            # enrichment), drop the oldest enrichment, then append the new one.
            import re
            # Split on ## headings (keep delimiters)
            parts = re.split(r"(?=\n## )", skill.content)
            if len(parts) > 2:
                # Drop the second part (first enrichment) to make room
                parts.pop(1)
            trimmed = "".join(parts).rstrip()
            skill.content = trimmed + "\n\n" + new_section + "\n"

            # If still over budget (new section itself is huge), hard-truncate
            if len(skill.content) > MAX_SKILL_CONTENT_CHARS:
                skill.content = skill.content[:MAX_SKILL_CONTENT_CHARS].rstrip() + "\n"

        logger.info("[Curator] ENRICH: %s with content: %s", skill.skill_id, enrich_content[:50])
        return True

    def _birth_or_enrich(
        self,
        gaps: list[str],
        module_tag: str,
        task_type: str,
    ) -> str | None:
        active_skills = self.library.list_active(module_tag)
        active_summaries = "\n".join(
            f"  - {s.skill_id}: {s.name} — {s.description}" for s in active_skills
        ) or "  (none)"

        prompt = BIRTH_OR_ENRICH_PROMPT.format(
            gaps=json.dumps(gaps),
            active_skills=active_summaries,
        )

        try:
            response = self.llm_fn(prompt)
            parsed = extract_json_from_text(response)
            if parsed is None:
                logger.warning("[Curator] Failed to parse birth/enrich response")
                return None

            action = parsed.get("action", "CREATE")
            if action == "ENRICH":
                skill_id = parsed.get("skill_id", "")
                enrich_content = parsed.get("enrich_content", parsed.get("strategy_note", ""))
                if skill_id and enrich_content:
                    self._enrich(skill_id, enrich_content)
                    return "ENRICH"
            else:
                name = parsed.get("name", "Unnamed Skill")
                description = parsed.get("description", "")
                content = parsed.get("content", "")
                keywords = parsed.get("keywords", [])
                task_types_list = parsed.get("task_types", [task_type])
                if content:
                    self._birth(
                        gap=gaps[0] if gaps else "",
                        module_tag=module_tag,
                        task_type=task_type,
                        name=name,
                        description=description,
                        content=content,
                        keywords=keywords,
                        task_types=task_types_list,
                    )
                    return "BIRTH"
        except Exception as e:
            logger.warning("[Curator] Error in birth/enrich: %s", e)

        return None

    def _specialize(self, module_tag: str) -> bool:
        active = self.library.list_active(module_tag)
        did_specialize = False

        for skill in active:
            if skill.total_evals < MIN_EVALS_FOR_LIFECYCLE:
                continue

            harmful_ratio = skill.harmful_count / skill.total_evals if skill.total_evals > 0 else 0.0

            # New trigger: Pareto-optimal often (useful) AND harmful sometimes
            if skill.pareto_frequency < SPECIALIZE_PARETO_FREQ or harmful_ratio < SPECIALIZE_HARMFUL_RATIO:
                continue

            logger.info(
                "[Curator] SPECIALIZE candidate: %s (pareto_freq=%.2f, harmful_ratio=%.2f)",
                skill.skill_id, skill.pareto_frequency, harmful_ratio,
            )

            prompt = SPECIALIZE_PROMPT.format(
                skill_name=skill.name,
                skill_content=skill.content,
                skill_description=skill.description,
                pareto_frequency=skill.pareto_frequency,
                harmful_ratio=harmful_ratio,
                scores=skill.eval_scores[-10:],
            )

            try:
                response = self.llm_fn(prompt)
                parsed = extract_json_from_text(response)
                if parsed is None or "children" not in parsed:
                    continue

                children = parsed["children"]
                if len(children) < 2:
                    continue

                child_ids = []
                for child_data in children[:2]:
                    child = Skill(
                        name=child_data.get("name", f"{skill.name} (child)"),
                        description=child_data.get("description", ""),
                        content=child_data.get("content", ""),
                        module_tag=module_tag,
                        parent_id=skill.skill_id,
                        keywords=child_data.get("keywords", skill.keywords.copy()),
                        task_types=child_data.get("task_types", skill.task_types.copy()),
                    )
                    self.library.add(child)
                    child_ids.append(child.skill_id)

                skill.children_ids.extend(child_ids)
                skill.status = "retired"
                logger.info(
                    "[Curator] SPECIALIZE: %s → %s", skill.skill_id, child_ids
                )
                did_specialize = True
            except Exception as e:
                logger.warning("[Curator] Error in specialize: %s", e)

        return did_specialize

    def _generalize(self, module_tag: str) -> bool:
        active = self.library.list_active(module_tag)
        if len(active) < 2:
            return False

        merged_ids: set[str] = set()
        did_merge = False

        for i, skill_a in enumerate(active):
            if skill_a.skill_id in merged_ids:
                continue
            for skill_b in active[i + 1 :]:
                if skill_b.skill_id in merged_ids:
                    continue
                overlap = _jaccard_overlap(skill_a.keywords, skill_b.keywords)
                if overlap >= MERGE_KEYWORD_OVERLAP:
                    merged = Skill(
                        name=f"{skill_a.name} + {skill_b.name}",
                        description=f"Merged: {skill_a.description} | {skill_b.description}",
                        content=f"{skill_a.content}\n\n{skill_b.content}",
                        module_tag=module_tag,
                        keywords=list(set(skill_a.keywords + skill_b.keywords)),
                        task_types=list(set(skill_a.task_types + skill_b.task_types)),
                        helpful_count=skill_a.helpful_count + skill_b.helpful_count,
                        harmful_count=skill_a.harmful_count + skill_b.harmful_count,
                        neutral_count=skill_a.neutral_count + skill_b.neutral_count,
                    )
                    self.library.add(merged)
                    skill_a.status = "retired"
                    skill_b.status = "retired"
                    merged_ids.add(skill_a.skill_id)
                    merged_ids.add(skill_b.skill_id)
                    logger.info(
                        "[Curator] GENERALIZE: %s + %s → %s",
                        skill_a.skill_id,
                        skill_b.skill_id,
                        merged.skill_id,
                    )
                    did_merge = True
                    break

        return did_merge

    def _retire(self, module_tag: str) -> bool:
        active = self.library.list_active(module_tag)
        did_retire = False
        for skill in active:
            # Primary: retire if epsilon-dominated by another active skill
            if len(skill.score_matrix) >= MIN_EVALS_FOR_LIFECYCLE:
                dominated_by = None
                for other in active:
                    if other.skill_id == skill.skill_id:
                        continue
                    if _epsilon_dominates(other, skill):
                        dominated_by = other.skill_id
                        break
                if dominated_by:
                    skill.status = "retired"
                    logger.info(
                        "[Curator] RETIRE: %s dominated by %s", skill.skill_id, dominated_by
                    )
                    did_retire = True
                    continue

            # Fallback: retire if harmful_ratio > 0.7 with sufficient evals
            if skill.total_evals >= MIN_EVALS_FOR_LIFECYCLE:
                harmful_ratio = skill.harmful_count / skill.total_evals
                if harmful_ratio > 0.7:
                    skill.status = "retired"
                    logger.info(
                        "[Curator] RETIRE: %s (harmful_ratio=%.2f)", skill.skill_id, harmful_ratio
                    )
                    did_retire = True
        return did_retire


def _jaccard_overlap(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


