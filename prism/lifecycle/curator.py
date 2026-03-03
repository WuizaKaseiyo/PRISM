from __future__ import annotations

import json
import logging
from typing import Any, Callable

from prism.lifecycle.reflector import ReflectionResult
from prism.skill_library.library import SkillLibrary
from prism.skill_library.skill import Skill
from prism.utils import extract_json_from_text

logger = logging.getLogger(__name__)

# Lifecycle thresholds
RETIRE_GAP = 3
SPECIALIZE_VARIANCE = 0.20
MIN_EVALS_FOR_LIFECYCLE = 5
MERGE_KEYWORD_OVERLAP = 0.60


BIRTH_OR_ENRICH_PROMPT = """Given these knowledge gaps and existing active skills, decide whether to CREATE a new skill or ENRICH an existing one.

Gaps identified:
{gaps}

Active skills:
{active_skills}

If a gap closely matches an existing skill, return ENRICH with a strategy note to add.
If the gap represents genuinely new knowledge, return CREATE with full skill details.

Return ONLY a JSON object:
{{
  "action": "CREATE" or "ENRICH",
  "skill_id": "existing skill ID (only for ENRICH)",
  "name": "skill name (only for CREATE)",
  "description": "skill description (only for CREATE)",
  "content": "skill content (only for CREATE)",
  "strategy_note": "note to add (only for ENRICH)",
  "keywords": ["keyword1", "keyword2"],
  "task_types": ["type1"]
}}"""

SPECIALIZE_PROMPT = """This skill has high variance in its evaluation scores, suggesting it works well for some tasks but poorly for others.

Skill: {skill_name}
Content: {skill_content}
Description: {skill_description}
Score variance: {variance:.3f}
Eval scores: {scores}

Split this into 2 more specialized skills. Each child should handle a specific subset of use cases.

Return ONLY a JSON object:
{{
  "children": [
    {{
      "name": "specialized name 1",
      "description": "when to use child 1",
      "content": "specialized content 1",
      "keywords": ["kw1"],
      "task_types": ["type1"]
    }},
    {{
      "name": "specialized name 2",
      "description": "when to use child 2",
      "content": "specialized content 2",
      "keywords": ["kw1"],
      "task_types": ["type1"]
    }}
  ]
}}"""


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

        # RETIRE: deactivate skills where harmful - helpful > RETIRE_GAP
        retired = self._retire(module_tag)
        if retired:
            ops_performed.append("RETIRE")

        # BIRTH or ENRICH from gaps
        if reflection.gaps:
            action = self._birth_or_enrich(reflection.gaps, module_tag, task_type)
            if action:
                ops_performed.append(action)

        # SPECIALIZE: split high-variance skills
        specialized = self._specialize(module_tag)
        if specialized:
            ops_performed.append("SPECIALIZE")

        # GENERALIZE: merge overlapping skills
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

    def _enrich(self, skill_id: str, strategy_note: str) -> bool:
        skill = self.library.get(skill_id)
        if skill is None:
            return False
        skill.strategy_notes.append(strategy_note)
        logger.info("[Curator] ENRICH: %s with note: %s", skill.skill_id, strategy_note[:50])
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
                note = parsed.get("strategy_note", "")
                if skill_id and note:
                    self._enrich(skill_id, note)
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
            if skill.score_variance < SPECIALIZE_VARIANCE:
                continue

            prompt = SPECIALIZE_PROMPT.format(
                skill_name=skill.name,
                skill_content=skill.content,
                skill_description=skill.description,
                variance=skill.score_variance,
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
            if skill.harmful_count - skill.helpful_count > RETIRE_GAP:
                skill.status = "retired"
                logger.info("[Curator] RETIRE: %s (net_value=%d)", skill.skill_id, skill.net_value)
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


