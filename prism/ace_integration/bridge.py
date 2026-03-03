from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from prism.assembler.assembler import SkillAssembler
from prism.lifecycle.curator import SkillCurator
from prism.lifecycle.reflector import PRISMReflector
from prism.skill_library.library import SkillLibrary
from prism.skill_library.skill import Skill
from prism.task_index.index import TaskTypeIndex

logger = logging.getLogger(__name__)

# ACE playbook line pattern: [id] helpful=X harmful=Y :: content
# Matches format used by ACE's parse_playbook_line in playbook_utils.py
ACE_LINE_PATTERN = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")

# ACE section slug mapping (mirrors ace/utils.py get_section_slug)
ACE_SLUG_MAP: dict[str, str] = {
    "financial_strategies_and_insights": "fin",
    "formulas_and_calculations": "calc",
    "code_snippets_and_templates": "code",
    "common_mistakes_to_avoid": "err",
    "problem_solving_heuristics": "prob",
    "context_clues_and_indicators": "ctx",
    "others": "misc",
    "meta_strategies": "meta",
    "general": "gen",
}


def _get_section_slug(section: str) -> str:
    """Map section name to 3-5 char slug, matching ACE's convention."""
    normalized = section.lower().replace(" ", "_").replace("&", "and")
    return ACE_SLUG_MAP.get(normalized, normalized[:4])


class PRISMACEBridge:
    """Bridge between PRISM and ACE, replacing ACE's Generator + Reflector + Curator.

    ACE coordinates three agents in a loop:
      Generator (produces answers using playbook)
      → Reflector (tags bullets as helpful/harmful/neutral)
      → update_counts (increments counters on playbook bullets)
      → Curator (generates ADD operations for new bullets)

    PRISMACEBridge replaces this with PRISM's structured skill lifecycle:
      generate() replaces Generator
      reflect_and_curate() replaces Reflector + Counter + Curator
      import/export methods handle playbook ↔ Skill conversion
    """

    def __init__(
        self,
        library: SkillLibrary,
        task_index: TaskTypeIndex,
        llm_fn: Callable[[str], str],
        embed_fn: Callable[[str], list[float]] | None = None,
        top_k: int = 5,
        token_budget: int = 2000,
    ):
        self.library = library
        self.task_index = task_index
        self.llm_fn = llm_fn

        self.assembler = SkillAssembler(
            library=library,
            task_index=task_index,
            embed_fn=embed_fn,
            llm_fn=llm_fn,
            top_k=top_k,
            token_budget=token_budget,
        )
        self.reflector = PRISMReflector(llm_fn=llm_fn)
        self.curator = SkillCurator(library=library, llm_fn=llm_fn)

    def generate(
        self,
        base_prompt: str,
        task: dict[str, Any],
        module_tag: str = "general",
    ) -> dict[str, Any]:
        """Replace ACE Generator: assemble skills and produce augmented prompt.

        ACE's Generator takes (question, playbook, context, reflection) and returns
        (full_response, bullet_ids_used, call_info). This method returns a dict with
        equivalent information.
        """
        skill_context, skill_ids = self.assembler.assemble(task, module_tag=module_tag)

        if skill_context:
            augmented_prompt = f"{base_prompt}\n\n## Relevant Skills\n{skill_context}"
        else:
            augmented_prompt = base_prompt

        response = self.llm_fn(augmented_prompt)

        return {
            "prompt": augmented_prompt,
            "response": response,
            "skill_ids": skill_ids,
            "skill_context": skill_context,
        }

    def reflect_and_curate(
        self,
        task: dict[str, Any],
        gen_result: dict[str, Any],
        trace: str,
        ground_truth: str = "",
        module_tag: str = "general",
        score: float | None = None,
    ) -> dict[str, Any]:
        """Replace ACE Reflector + Counter + Curator.

        ACE's Reflector analyzes outputs and returns bullet_tags (helpful/harmful/neutral).
        ACE's Counter layer increments counts via update_bullet_counts().
        ACE's Curator generates ADD operations for new bullets.

        This method runs all three operations through PRISM's lifecycle:
        reflection → attribution counting → curation (BIRTH/ENRICH/SPECIALIZE/GENERALIZE/RETIRE).

        Args:
            task: The task dict
            gen_result: Output from generate()
            trace: Reasoning trace from generator
            ground_truth: Expected answer (empty if unavailable, like ACE's no_ground_truth mode)
            module_tag: Module tag for skill filtering
            score: Explicit score. If None, uses heuristic (1.0 if ground_truth in output, else 0.5)
        """
        skill_ids = gen_result.get("skill_ids", [])
        output = gen_result.get("response", "")

        # Score: use explicit or heuristic fallback
        if score is not None:
            eval_score = score
        else:
            eval_score = 1.0 if ground_truth and ground_truth in output else 0.5

        reflection = self.reflector.reflect(
            task=task,
            score=eval_score,
            trace=trace,
            output=output,
            skill_ids=skill_ids,
        )

        task_type = self.task_index.classify(str(task))
        ops = self.curator.curate(
            reflection=reflection,
            module_tag=module_tag,
            task_type=task_type,
        )

        # Update index (EMA)
        for sid in skill_ids:
            self.task_index.update(task_type, sid, eval_score)
            skill = self.library.get(sid)
            if skill:
                skill.eval_scores.append(eval_score)

        return {
            "reflection": {
                "attributions": reflection.attributions,
                "gaps": reflection.gaps,
                "diagnosis": reflection.diagnosis,
            },
            "operations": ops,
            "score": eval_score,
        }

    def import_from_ace_playbook(
        self,
        playbook_text: str,
        module_tag: str = "general",
    ) -> list[str]:
        """Parse ACE bullet format → Skill objects. Returns list of new skill_ids.

        ACE playbook format:
          ## SECTION HEADER
          [slug-00001] helpful=X harmful=Y :: content text

        Each bullet becomes a Skill with:
          - name derived from ACE ID
          - helpful/harmful counts preserved
          - keywords extracted from content
          - section info stored in task_types
        """
        imported_ids: list[str] = []
        current_section = "general"

        for line in playbook_text.strip().split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Track section headers
            if stripped.startswith("##"):
                current_section = stripped[2:].strip().lower().replace(" ", "_").replace("&", "and")
                continue

            if stripped.startswith("#"):
                continue

            match = ACE_LINE_PATTERN.match(stripped)
            if match:
                ace_id = match.group(1)
                helpful = int(match.group(2))
                harmful = int(match.group(3))
                content = match.group(4)

                skill = Skill(
                    name=f"ACE-{ace_id}",
                    description=f"Imported from ACE playbook ({current_section}): {ace_id}",
                    content=content,
                    module_tag=module_tag,
                    helpful_count=helpful,
                    harmful_count=harmful,
                    keywords=_extract_keywords(content),
                    task_types=[current_section],
                )
                sid = self.library.add(skill)
                imported_ids.append(sid)

        logger.info("[ACEBridge] Imported %d skills from ACE playbook", len(imported_ids))
        return imported_ids

    def export_to_ace_playbook(self, path: str | Path | None = None) -> str:
        """Export Skill objects → ACE bullet format.

        Produces output compatible with ACE's parse_playbook_line():
          [slug-00001] helpful=X harmful=Y :: content

        Skills are grouped by task_type into ACE-style sections.
        """
        # Group active skills by their first task_type
        sections: dict[str, list[Skill]] = {}
        for skill in self.library.filter(status="active"):
            section = skill.task_types[0] if skill.task_types else "others"
            sections.setdefault(section, []).append(skill)

        lines: list[str] = []
        bullet_id = 1
        for section_name in sorted(sections.keys()):
            # Format section header like ACE
            header = section_name.upper().replace("_", " ")
            lines.append(f"## {header}")
            slug = _get_section_slug(section_name)
            for skill in sections[section_name]:
                ace_id = f"{slug}-{bullet_id:05d}"
                lines.append(
                    f"[{ace_id}] helpful={skill.helpful_count} "
                    f"harmful={skill.harmful_count} :: {skill.content}"
                )
                bullet_id += 1
            lines.append("")  # blank line between sections

        text = "\n".join(lines).rstrip()
        if path:
            Path(path).write_text(text)
        return text


def _extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """Simple keyword extraction from text."""
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    freq: dict[str, int] = {}
    stop_words = {
        "this", "that", "with", "from", "have", "been", "will", "would",
        "could", "should", "their", "there", "then", "than", "when", "what",
        "which", "where", "about", "into", "each", "make", "like", "just",
        "over", "such", "also", "more", "some", "very", "after", "before",
    }
    for word in words:
        if word not in stop_words:
            freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:max_keywords]]
