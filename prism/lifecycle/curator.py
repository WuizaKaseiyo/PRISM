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
SEMANTIC_SIM_THRESHOLD = 0.5     # condition 4: cross-niche domination gate


def _get_attributed(entry) -> float:
    """Extract attributed delta from a score matrix entry.

    Handles both old format (plain float) and new format (dict with 'attributed' key).
    """
    if isinstance(entry, dict):
        return entry.get("attributed", entry.get("delta", 0.0))
    return float(entry)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Pure-python cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


BIRTH_OR_ENRICH_PROMPT = """Given these knowledge gaps and existing active skills, decide whether to CREATE a new skill or ENRICH an existing one.

Gaps identified:
{gaps}

Active skills:
{active_skills}

If a gap closely matches an existing skill, return ENRICH with additional content to append.
If the gap represents genuinely new knowledge, return CREATE with full skill details.

## Quality Requirements for Skill Content

A good skill is a reusable decision-making guide that changes how a model reasons — NOT a textbook summary or generic advice. Apply these principles:

**1. Concrete over abstract.** Every instruction must be actionable. Bad: "Consider edge cases." Good: "Check n=0, n=1, and n=negative separately before applying the general formula."

**2. Decision procedures, not descriptions.** Write IF-THEN rules the model can follow. Bad: "Modular arithmetic is useful for remainder problems." Good: "When the problem asks for 'the last k digits' or 'remainder when divided by m': (1) Identify the modulus m, (2) Reduce the base mod m, (3) Apply Euler's theorem if gcd(base,m)=1, (4) Otherwise factor m and use CRT."

**3. Worked examples with reasoning.** Include at least one Input → Reasoning → Output example showing exactly how to apply the workflow. Show the decision points, not just the answer.

**4. Explain WHY, not just WHAT.** Instead of "Always verify your answer", write "Verify by substituting back into the original equation — competition problems often have extraneous solutions introduced by squaring or cross-multiplying."

**5. Generalize across the problem class.** The skill must help with ALL problems of this type, not just the one that triggered creation. Identify the common structure: what makes these problems similar? What varies?

**6. No filler.** Remove anything the model already knows or that doesn't change behavior. "Read the problem carefully" and "Show your work" are worthless. Every sentence must provide information the model wouldn't have without this skill.

**7. Pitfalls are high-value.** Specific common mistakes for this problem type are among the most useful content. Format as: "PITFALL: [what goes wrong] → FIX: [what to do instead]."

## Format

For CREATE:
- "name": concise skill name describing the method (e.g. "CRT-Based Modular Reduction")
- "description": trigger condition starting with "Use when..." — specific enough to match the right problems, broad enough to cover the class. Be pushy: include concrete signal words/patterns.
- "content": a FULL markdown document (50-200 lines) with:
  - `# Title`
  - `## Overview` — what class of problems this solves and why a specialized approach is needed
  - `## Workflow` — numbered decision procedure with IF-THEN branches
  - `## Worked Example` — at least one full Input → Reasoning → Output
  - `## Pitfalls` — common mistakes and their fixes
  Do NOT write generic advice. Every line must earn its place.

For ENRICH, provide a markdown section (with ## heading) containing NEW decision procedures, examples, or pitfalls — not restatements of what the skill already contains.

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

Analyze the parent skill's content to identify which parts help vs hurt. Split into 2 children where each child:
- Handles a DISTINCT subset of problems (no overlap in trigger conditions)
- Contains ONLY the strategies relevant to its subset
- Has a narrower, more specific trigger description

Each child must follow these quality requirements:
- Concrete decision procedures (IF-THEN rules), not abstract advice
- At least one worked example showing Input → Reasoning → Output
- Specific pitfalls for this subtype with fixes
- No filler — every sentence must change model behavior
- Explain WHY each step matters, not just WHAT to do

Return ONLY a JSON object:
{{
  "children": [
    {{
      "name": "Specialized Name 1",
      "description": "Use when the task involves... (specific trigger)",
      "content": "# Title\\n\\n## Overview\\n...\\n\\n## Workflow\\n1. ...\\n\\n## Worked Example\\n...\\n\\n## Pitfalls\\n...",
      "keywords": ["kw1"],
      "task_types": ["type1"]
    }},
    {{
      "name": "Specialized Name 2",
      "description": "Use when the task involves... (specific trigger)",
      "content": "# Title\\n\\n## Overview\\n...\\n\\n## Workflow\\n1. ...\\n\\n## Worked Example\\n...\\n\\n## Pitfalls\\n...",
      "keywords": ["kw1"],
      "task_types": ["type1"]
    }}
  ]
}}"""


def _epsilon_dominates(a: Skill, b: Skill, eps: float = EPSILON) -> bool:
    """Return True if skill A epsilon-dominates skill B.

    A eps-dominates B iff:
    1. They share >= MIN_SHARED_INSTANCES task instances
    2. A >= B - eps on ALL shared instances
    3. A > B on at least one shared instance
    4. Semantic similarity >= SEMANTIC_SIM_THRESHOLD (if embeddings available)
    """
    # Condition 4: semantic similarity gate — prevent cross-niche domination
    if a.embedding is not None and b.embedding is not None:
        sim = _cosine_sim(a.embedding, b.embedding)
        if sim < SEMANTIC_SIM_THRESHOLD:
            return False

    shared_keys = set(a.score_matrix) & set(b.score_matrix)
    if len(shared_keys) < MIN_SHARED_INSTANCES:
        return False

    strictly_better_on_one = False
    for key in shared_keys:
        a_score = _get_attributed(a.score_matrix[key])
        b_score = _get_attributed(b.score_matrix[key])
        if a_score < b_score - eps:
            return False
        if a_score > b_score:
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
                score = _get_attributed(skill.score_matrix[key])
                if score > best_score:
                    best_score = score

        for skill in present_skills:
            present_count[skill.skill_id] += 1
            if _get_attributed(skill.score_matrix[key]) >= best_score - EPSILON:
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
        for key, entry in skill.score_matrix.items():
            score = _get_attributed(entry)
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
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        self.library = library
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn

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

        # NO-OP: skip curation when skills performed well and no gaps identified
        has_harmful = any(
            attr.get("tag") == "harmful" for attr in reflection.attributions
        )
        if reflection.attributions and not has_harmful and not reflection.gaps:
            logger.info("[Curator] NO-OP: all skills helpful/neutral, no gaps")
            return []

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
        if self.embed_fn:
            try:
                skill.embedding = self.embed_fn(skill.description)
            except Exception as e:
                logger.warning("[Curator] Failed to embed skill %s: %s", skill.name, e)
        self.library.add(skill)
        logger.info("[Curator] BIRTH: %s (%s)", skill.name, skill.skill_id)
        return skill

    def _blind_compare(self, old_content: str, new_content: str, description: str) -> str:
        """Blind comparison of two skill versions. Returns 'A' or 'B' (winner)."""
        import random
        # Randomize presentation order to eliminate position bias
        if random.random() < 0.5:
            version_a, version_b = old_content, new_content
            mapping = {"A": "old", "B": "new"}
        else:
            version_a, version_b = new_content, old_content
            mapping = {"A": "new", "B": "old"}

        prompt = f"""Compare two versions of a skill document for the following purpose:
"{description}"

## Version A
{version_a[:3000]}

## Version B
{version_b[:3000]}

Evaluate on these criteria:
1. **Actionability**: Does it provide concrete IF-THEN decision procedures?
2. **Specificity**: Does it contain worked examples and specific pitfalls?
3. **Organization**: Is it well-structured with clear sections?
4. **Non-redundancy**: Does it avoid repeating itself?

Which version is better overall? Return ONLY a JSON object:
{{"winner": "A" or "B", "reason": "one sentence"}}"""

        try:
            response = self.llm_fn(prompt)
            parsed = extract_json_from_text(response)
            if parsed and "winner" in parsed:
                winner_label = parsed["winner"].strip().upper()
                if winner_label in ("A", "B"):
                    winner = mapping.get(winner_label, "new")
                    logger.info(
                        "[Curator] Blind compare: %s wins (%s)",
                        winner, parsed.get("reason", ""),
                    )
                    return winner
        except Exception as e:
            logger.warning("[Curator] Blind compare failed: %s", e)
        return "new"  # default: accept new version

    def _enrich(self, skill_id: str, enrich_content: str) -> bool:
        skill = self.library.get(skill_id)
        if skill is None:
            return False
        if not enrich_content.strip():
            return False

        old_content = skill.content
        new_section = enrich_content.strip()
        proposed = skill.content.rstrip() + "\n\n" + new_section + "\n"

        if len(proposed) <= MAX_SKILL_CONTENT_CHARS:
            proposed_final = proposed
        else:
            # Over budget — drop the oldest enrichment section to make room.
            import re
            parts = re.split(r"(?=\n## )", skill.content)
            if len(parts) > 2:
                parts.pop(1)
            trimmed = "".join(parts).rstrip()
            proposed_final = trimmed + "\n\n" + new_section + "\n"

            if len(proposed_final) > MAX_SKILL_CONTENT_CHARS:
                proposed_final = proposed_final[:MAX_SKILL_CONTENT_CHARS].rstrip() + "\n"

        # Blind comparison: reject if new version is worse
        winner = self._blind_compare(old_content, proposed_final, skill.description)
        if winner == "old":
            logger.info("[Curator] ENRICH rejected for %s: old version better", skill.skill_id)
            return False

        skill.content = proposed_final
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
                    if self.embed_fn:
                        try:
                            child.embedding = self.embed_fn(child.description)
                        except Exception as e:
                            logger.warning("[Curator] Failed to embed child %s: %s", child.name, e)
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


