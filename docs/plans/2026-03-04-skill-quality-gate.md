# Skill Quality Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add content-level quality control for PRISM skills — validate generated content at creation time and periodically refine existing skills via LLM-based quality scoring.

**Architecture:** New `SkillValidator` component (L3) with two modes: `validate()` gates BIRTH/ENRICH/SPECIALIZE output at creation time, `audit()` periodically reviews all active skills and triggers REFINE (full content rewrite). Curator receives an optional validator reference; engine creates validator and wires it in.

**Tech Stack:** Pure Python 3.10+, no new dependencies. Uses existing `llm_fn` callable and `extract_json_from_text` utility.

---

### Task 1: Add quality fields to Skill dataclass

**Files:**
- Modify: `prism/prism/skill_library/skill.py:31-33`
- Modify: `prism/prism/skill_library/skill.py:141-142`
- Modify: `prism/prism/skill_library/skill.py:167-168`
- Test: `tests/test_skill_quality_fields.py`

**Step 1: Write the failing test**

Create `prism/tests/test_skill_quality_fields.py`:

```python
from prism.skill_library.skill import Skill


def test_quality_fields_defaults():
    s = Skill(name="Test", description="desc", content="body")
    assert s.quality_score == 0.0
    assert s.last_validated_step == 0
    assert s.refine_count == 0


def test_quality_fields_roundtrip():
    s = Skill(
        name="Test", description="desc", content="body",
        quality_score=0.85, last_validated_step=42, refine_count=2,
    )
    d = s.to_dict()
    assert d["quality_score"] == 0.85
    assert d["last_validated_step"] == 42
    assert d["refine_count"] == 2

    s2 = Skill.from_dict(d)
    assert s2.quality_score == 0.85
    assert s2.last_validated_step == 42
    assert s2.refine_count == 2


def test_quality_fields_backward_compat():
    """Old dicts without quality fields should load with defaults."""
    old = {"skill_id": "x", "name": "Old", "description": "", "content": ""}
    s = Skill.from_dict(old)
    assert s.quality_score == 0.0
    assert s.last_validated_step == 0
    assert s.refine_count == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_skill_quality_fields.py -v`
Expected: FAIL — `Skill()` does not accept `quality_score` keyword

**Step 3: Add fields to Skill dataclass**

In `prism/prism/skill_library/skill.py`, after line 33 (`pareto_frequency: float = 0.0`), add:

```python
    quality_score: float = 0.0
    last_validated_step: int = 0
    refine_count: int = 0
```

In `to_dict()`, after `"pareto_frequency": self.pareto_frequency,` (line 142), add:

```python
            "quality_score": self.quality_score,
            "last_validated_step": self.last_validated_step,
            "refine_count": self.refine_count,
```

In `from_dict()`, after `pareto_frequency=data.get("pareto_frequency", 0.0),` (line 168), add:

```python
            quality_score=data.get("quality_score", 0.0),
            last_validated_step=data.get("last_validated_step", 0),
            refine_count=data.get("refine_count", 0),
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_skill_quality_fields.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add prism/prism/skill_library/skill.py tests/test_skill_quality_fields.py
git commit -m "feat: add quality_score, last_validated_step, refine_count to Skill"
```

---

### Task 2: Persist quality fields in library save/load

**Files:**
- Modify: `prism/prism/skill_library/library.py:131-132` (save)
- Modify: `prism/prism/skill_library/library.py:181-182` (load)
- Test: `tests/test_library_quality_persist.py`

**Step 1: Write the failing test**

Create `prism/tests/test_library_quality_persist.py`:

```python
import json
import tempfile
from pathlib import Path

from prism.skill_library.library import SkillLibrary
from prism.skill_library.skill import Skill


def test_quality_fields_persist():
    with tempfile.TemporaryDirectory() as tmp:
        lib = SkillLibrary(path=tmp)
        s = Skill(
            name="Test Skill", description="desc", content="# Test\n\nbody",
            quality_score=0.9, last_validated_step=10, refine_count=1,
        )
        lib.add(s)
        lib.save()

        # Verify _meta.json has quality fields
        meta = json.loads((Path(tmp) / "_meta.json").read_text())
        entry = meta[s.skill_id]
        assert entry["quality_score"] == 0.9
        assert entry["last_validated_step"] == 10
        assert entry["refine_count"] == 1

        # Reload and verify
        lib2 = SkillLibrary(path=tmp)
        s2 = lib2.get(s.skill_id)
        assert s2 is not None
        assert s2.quality_score == 0.9
        assert s2.last_validated_step == 10
        assert s2.refine_count == 1


def test_quality_fields_backward_compat_load():
    """Old _meta.json without quality fields loads with defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create skill dir + SKILL.md manually
        skill_dir = Path(tmp) / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: desc\n---\n\nbody\n"
        )
        # Write minimal _meta.json without quality fields
        meta = {
            "abc123": {
                "slug": "test-skill",
                "name": "Test Skill",
                "skill_id": "abc123",
                "status": "active",
                "module_tag": "general",
                "helpful_count": 0,
                "harmful_count": 0,
                "neutral_count": 0,
                "eval_scores": [],
                "parent_id": None,
                "children_ids": [],
                "keywords": [],
                "task_types": [],
                "trigger_conditions": "",
                "strategy_notes": [],
                "pitfalls": [],
                "created_at": "2026-01-01T00:00:00+00:00",
                "embedding": None,
            }
        }
        (Path(tmp) / "_meta.json").write_text(json.dumps(meta))

        lib = SkillLibrary(path=tmp)
        s = lib.get("abc123")
        assert s is not None
        assert s.quality_score == 0.0
        assert s.last_validated_step == 0
        assert s.refine_count == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_library_quality_persist.py -v`
Expected: FAIL — `quality_score` not in saved _meta.json

**Step 3: Add quality fields to save() and _load()**

In `prism/prism/skill_library/library.py`, in `save()` after `"pareto_frequency": skill.pareto_frequency,` (line 132), add:

```python
                "quality_score": skill.quality_score,
                "last_validated_step": skill.last_validated_step,
                "refine_count": skill.refine_count,
```

In `_load()`, after `skill.pareto_frequency = m.get("pareto_frequency", 0.0)` (line 182), add:

```python
                    skill.quality_score = m.get("quality_score", 0.0)
                    skill.last_validated_step = m.get("last_validated_step", 0)
                    skill.refine_count = m.get("refine_count", 0)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_library_quality_persist.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add prism/prism/skill_library/library.py tests/test_library_quality_persist.py
git commit -m "feat: persist quality_score, last_validated_step, refine_count in _meta.json"
```

---

### Task 3: Create SkillValidator with validate()

**Files:**
- Create: `prism/prism/lifecycle/validator.py`
- Test: `tests/test_validator.py`

**Step 1: Write the failing test**

Create `prism/tests/test_validator.py`:

```python
import json

from prism.lifecycle.validator import SkillValidator, ValidationResult
from prism.skill_library.skill import Skill


def _make_mock_llm(score_dict: dict):
    """Return an llm_fn that returns a JSON response with the given scores."""
    def llm_fn(prompt: str) -> str:
        return json.dumps(score_dict)
    return llm_fn


def test_validate_high_quality():
    llm_fn = _make_mock_llm({
        "structural_completeness": 0.9,
        "actionability": 0.8,
        "description_alignment": 0.9,
        "internal_consistency": 0.9,
        "differentiation": 0.8,
        "issues": [],
        "overall_assessment": "Good skill",
    })
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Good Skill", description="Use when X", content="# Title\n\n## Overview\nbody\n\n## Workflow\n1. Step")
    result = validator.validate(skill)
    assert isinstance(result, ValidationResult)
    assert result.score >= 0.7
    assert result.issues == []


def test_validate_low_quality():
    llm_fn = _make_mock_llm({
        "structural_completeness": 0.2,
        "actionability": 0.1,
        "description_alignment": 0.3,
        "internal_consistency": 0.2,
        "differentiation": 0.1,
        "issues": ["Too vague", "No workflow"],
        "overall_assessment": "Poor skill",
    })
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Bad Skill", description="desc", content="stuff")
    result = validator.validate(skill)
    assert result.score < 0.4
    assert len(result.issues) > 0


def test_validate_returns_weighted_score():
    llm_fn = _make_mock_llm({
        "structural_completeness": 1.0,  # weight 0.20
        "actionability": 1.0,            # weight 0.30
        "description_alignment": 0.0,    # weight 0.20
        "internal_consistency": 0.0,     # weight 0.15
        "differentiation": 0.0,         # weight 0.15
        "issues": [],
        "overall_assessment": "Mixed",
    })
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Test", description="desc", content="body")
    result = validator.validate(skill)
    expected = 1.0 * 0.20 + 1.0 * 0.30 + 0.0 * 0.20 + 0.0 * 0.15 + 0.0 * 0.15
    assert abs(result.score - expected) < 0.01


def test_validate_graceful_on_bad_llm_response():
    """If LLM returns garbage, validate should not crash."""
    def bad_llm(prompt: str) -> str:
        return "I don't know what you want"
    validator = SkillValidator(llm_fn=bad_llm)
    skill = Skill(name="Test", description="desc", content="body")
    result = validator.validate(skill)
    assert result.score == 0.0
    assert len(result.issues) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_validator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.lifecycle.validator'`

**Step 3: Create validator.py**

Create `prism/prism/lifecycle/validator.py`:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from prism.skill_library.skill import Skill
from prism.utils import extract_json_from_text

logger = logging.getLogger(__name__)

# Quality thresholds
QUALITY_ACCEPT = 0.70
QUALITY_REVISE = 0.40
REVISION_ACCEPT = 0.60

# Dimension weights
WEIGHT_STRUCTURAL = 0.20
WEIGHT_ACTIONABILITY = 0.30
WEIGHT_ALIGNMENT = 0.20
WEIGHT_CONSISTENCY = 0.15
WEIGHT_DIFFERENTIATION = 0.15

# Audit triggers
AUDIT_QUALITY_THRESHOLD = 0.60
AUDIT_ENRICH_COUNT = 3
AUDIT_MAX_AGE = 50

VALIDATE_PROMPT = """Rate this skill's content quality on 5 dimensions (0.0 to 1.0 each).

Skill name: {name}
Description: {description}

Content:
{content}

{active_context}

Score each dimension:
1. structural_completeness: Has # title, ## sections, 50+ lines, workflow/steps present
2. actionability: Contains concrete steps, methods, formulas — not just vague principles
3. description_alignment: Content teaches what the description claims
4. internal_consistency: No self-contradictions, no redundant repetition
5. differentiation: Sufficiently different from other active skills listed above

Return ONLY a JSON object:
{{
  "structural_completeness": 0.0-1.0,
  "actionability": 0.0-1.0,
  "description_alignment": 0.0-1.0,
  "internal_consistency": 0.0-1.0,
  "differentiation": 0.0-1.0,
  "issues": ["issue1", "issue2"],
  "overall_assessment": "brief text"
}}"""

REVISE_PROMPT = """Rewrite this skill to fix the identified quality issues.
Keep the same purpose and core knowledge, but improve clarity, structure, and actionability.

Skill: {name}
Description (DO NOT CHANGE): {description}

Current content:
{content}

Quality issues to fix:
{issues}

Write a complete replacement for the content field.
Requirements:
- # Title heading
- ## Overview section
- ## Workflow section with numbered steps
- ## Key Principles or ## Examples sections
- 50-200 lines, concrete and actionable
- No contradictions with the description

Return the rewritten content as plain markdown (no JSON wrapper)."""

REFINE_PROMPT = """Rewrite this skill to improve its quality based on performance data.
Keep the same purpose but make it clearer, better structured, and more actionable.

Skill: {name}
Description (DO NOT CHANGE): {description}

Current content:
{content}

Performance data:
- Helpful: {helpful_count}, Harmful: {harmful_count}, Neutral: {neutral_count}
- Pareto frequency: {pareto_frequency:.2f}
- Quality score: {quality_score:.2f}
- Quality issues: {issues}

Write a complete replacement for the content field.
Requirements:
- # Title heading
- ## Overview section
- ## Workflow section with numbered steps
- ## Key Principles or ## Examples sections
- 50-200 lines, concrete and actionable
- Address the quality issues listed above

Return the rewritten content as plain markdown (no JSON wrapper)."""


@dataclass
class ValidationResult:
    score: float
    dimensions: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    revised_content: str | None = None


@dataclass
class AuditAction:
    skill_id: str
    action: str  # "REFINE" or "RETIRE"
    reason: str
    quality_score: float


class SkillValidator:
    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn

    def validate(
        self,
        skill: Skill,
        active_skills: list[Skill] | None = None,
    ) -> ValidationResult:
        """Score skill content quality using LLM. Returns ValidationResult."""
        active_context = ""
        if active_skills:
            summaries = "\n".join(
                f"  - {s.skill_id}: {s.name} — {s.description}"
                for s in active_skills
                if s.skill_id != skill.skill_id
            )
            active_context = f"Existing active skills (for differentiation check):\n{summaries}"

        prompt = VALIDATE_PROMPT.format(
            name=skill.name,
            description=skill.description,
            content=skill.content,
            active_context=active_context,
        )

        try:
            response = self.llm_fn(prompt)
            parsed = extract_json_from_text(response)
            if parsed is None:
                logger.warning("[Validator] Failed to parse validation response")
                return ValidationResult(score=0.0, issues=["LLM response unparseable"])

            dimensions = {
                "structural_completeness": float(parsed.get("structural_completeness", 0.0)),
                "actionability": float(parsed.get("actionability", 0.0)),
                "description_alignment": float(parsed.get("description_alignment", 0.0)),
                "internal_consistency": float(parsed.get("internal_consistency", 0.0)),
                "differentiation": float(parsed.get("differentiation", 0.0)),
            }

            score = (
                dimensions["structural_completeness"] * WEIGHT_STRUCTURAL
                + dimensions["actionability"] * WEIGHT_ACTIONABILITY
                + dimensions["description_alignment"] * WEIGHT_ALIGNMENT
                + dimensions["internal_consistency"] * WEIGHT_CONSISTENCY
                + dimensions["differentiation"] * WEIGHT_DIFFERENTIATION
            )

            issues = parsed.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)]

            return ValidationResult(score=score, dimensions=dimensions, issues=issues)

        except Exception as e:
            logger.warning("[Validator] Error in validate: %s", e)
            return ValidationResult(score=0.0, issues=[f"Validation error: {e}"])

    def revise(self, skill: Skill, issues: list[str]) -> Skill:
        """Ask LLM to rewrite skill content to fix identified issues."""
        prompt = REVISE_PROMPT.format(
            name=skill.name,
            description=skill.description,
            content=skill.content,
            issues="\n".join(f"- {issue}" for issue in issues),
        )

        try:
            revised_content = self.llm_fn(prompt).strip()
            if len(revised_content) < 50:
                logger.warning("[Validator] Revision too short, keeping original")
                return skill

            revised = Skill(
                name=skill.name,
                description=skill.description,
                content=revised_content,
                module_tag=skill.module_tag,
                skill_id=skill.skill_id,
                keywords=skill.keywords.copy(),
                task_types=skill.task_types.copy(),
                parent_id=skill.parent_id,
            )
            return revised

        except Exception as e:
            logger.warning("[Validator] Error in revise: %s", e)
            return skill

    def validate_and_gate(
        self,
        skill: Skill,
        active_skills: list[Skill] | None = None,
    ) -> Skill | None:
        """Validate a skill, revise if marginal, discard if too poor.

        Returns the (possibly revised) skill if it passes, None if discarded.
        """
        result = self.validate(skill, active_skills)
        skill.quality_score = result.score

        if result.score >= QUALITY_ACCEPT:
            logger.info(
                "[Validator] ACCEPT: %s (score=%.2f)", skill.name, result.score
            )
            return skill

        if result.score >= QUALITY_REVISE:
            logger.info(
                "[Validator] REVISE: %s (score=%.2f, issues=%s)",
                skill.name, result.score, result.issues,
            )
            revised = self.revise(skill, result.issues)
            result2 = self.validate(revised, active_skills)
            revised.quality_score = result2.score
            if result2.score >= REVISION_ACCEPT:
                logger.info(
                    "[Validator] ACCEPT after revision: %s (score=%.2f)",
                    revised.name, result2.score,
                )
                return revised
            else:
                logger.info(
                    "[Validator] DISCARD after failed revision: %s (score=%.2f)",
                    skill.name, result2.score,
                )
                return None

        logger.info(
            "[Validator] DISCARD: %s (score=%.2f)", skill.name, result.score
        )
        return None

    def audit(
        self,
        skills: list[Skill],
        step_count: int,
    ) -> list[AuditAction]:
        """Review all active skills, return list of recommended actions."""
        actions: list[AuditAction] = []

        for skill in skills:
            # Skip recently validated skills
            if step_count - skill.last_validated_step < AUDIT_MAX_AGE // 2:
                continue

            result = self.validate(skill)
            skill.quality_score = result.score
            skill.last_validated_step = step_count

            if result.score < AUDIT_QUALITY_THRESHOLD:
                actions.append(AuditAction(
                    skill_id=skill.skill_id,
                    action="REFINE",
                    reason=f"Low quality score ({result.score:.2f}): {', '.join(result.issues[:3])}",
                    quality_score=result.score,
                ))
                continue

            # Check for bloated skills (many enrichments)
            if skill.refine_count == 0 and skill.content.count("\n## ") >= AUDIT_ENRICH_COUNT + 1:
                actions.append(AuditAction(
                    skill_id=skill.skill_id,
                    action="REFINE",
                    reason=f"Bloated: {skill.content.count(chr(10) + '## ')} sections, never refined",
                    quality_score=result.score,
                ))

        return actions

    def execute_refine(self, skill: Skill, reason: str) -> bool:
        """Execute a REFINE operation: rewrite skill content."""
        prompt = REFINE_PROMPT.format(
            name=skill.name,
            description=skill.description,
            content=skill.content,
            helpful_count=skill.helpful_count,
            harmful_count=skill.harmful_count,
            neutral_count=skill.neutral_count,
            pareto_frequency=skill.pareto_frequency,
            quality_score=skill.quality_score,
            issues=reason,
        )

        try:
            revised_content = self.llm_fn(prompt).strip()
            if len(revised_content) < 50:
                logger.warning("[Validator] REFINE output too short, skipping")
                return False

            # Validate the rewrite before applying
            old_content = skill.content
            skill.content = revised_content
            result = self.validate(skill)

            if result.score > skill.quality_score:
                skill.quality_score = result.score
                skill.refine_count += 1
                logger.info(
                    "[Validator] REFINE: %s (%.2f → %.2f)",
                    skill.skill_id, skill.quality_score, result.score,
                )
                return True
            else:
                # Rewrite didn't improve quality — rollback
                skill.content = old_content
                logger.info(
                    "[Validator] REFINE rollback: %s (new score %.2f not better)",
                    skill.skill_id, result.score,
                )
                return False

        except Exception as e:
            logger.warning("[Validator] Error in refine: %s", e)
            return False
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_validator.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add prism/prism/lifecycle/validator.py tests/test_validator.py
git commit -m "feat: add SkillValidator with validate, revise, audit, refine"
```

---

### Task 4: Test validate_and_gate decision logic

**Files:**
- Test: `tests/test_validator_gate.py`

**Step 1: Write the test**

Create `prism/tests/test_validator_gate.py`:

```python
import json

from prism.lifecycle.validator import SkillValidator
from prism.skill_library.skill import Skill


def _make_llm_sequence(responses: list[dict | str]):
    """Return an llm_fn that returns responses in order."""
    idx = [0]
    def llm_fn(prompt: str) -> str:
        r = responses[min(idx[0], len(responses) - 1)]
        idx[0] += 1
        if isinstance(r, dict):
            return json.dumps(r)
        return r
    return llm_fn


HIGH_SCORE = {
    "structural_completeness": 0.9, "actionability": 0.8,
    "description_alignment": 0.9, "internal_consistency": 0.9,
    "differentiation": 0.8, "issues": [], "overall_assessment": "Good",
}

LOW_SCORE = {
    "structural_completeness": 0.1, "actionability": 0.1,
    "description_alignment": 0.1, "internal_consistency": 0.1,
    "differentiation": 0.1, "issues": ["Bad"], "overall_assessment": "Bad",
}

MID_SCORE = {
    "structural_completeness": 0.5, "actionability": 0.5,
    "description_alignment": 0.5, "internal_consistency": 0.5,
    "differentiation": 0.5, "issues": ["Needs work"], "overall_assessment": "OK",
}


def test_gate_accepts_high_quality():
    llm_fn = _make_llm_sequence([HIGH_SCORE])
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Good", description="desc", content="# T\n\n## O\nbody")
    result = validator.validate_and_gate(skill)
    assert result is not None
    assert result.quality_score >= 0.7


def test_gate_discards_low_quality():
    llm_fn = _make_llm_sequence([LOW_SCORE])
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Bad", description="desc", content="bad")
    result = validator.validate_and_gate(skill)
    assert result is None


def test_gate_revises_marginal_then_accepts():
    """Mid score → revise → high score → accept."""
    llm_fn = _make_llm_sequence([
        MID_SCORE,                              # First validate: marginal
        "# Revised\n\n## Overview\nBetter now",  # Revision content
        HIGH_SCORE,                              # Second validate: passes
    ])
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Mid", description="desc", content="meh")
    result = validator.validate_and_gate(skill)
    assert result is not None


def test_gate_revises_marginal_then_discards():
    """Mid score → revise → still low → discard."""
    llm_fn = _make_llm_sequence([
        MID_SCORE,                              # First validate: marginal
        "short",                                 # Revision content (too short, keeps original)
        LOW_SCORE,                               # Second validate: still bad
    ])
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Mid", description="desc", content="meh")
    result = validator.validate_and_gate(skill)
    assert result is None
```

**Step 2: Run test**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_validator_gate.py -v`
Expected: 4 passed (using the validator from Task 3)

**Step 3: Commit**

```bash
git add tests/test_validator_gate.py
git commit -m "test: validate_and_gate decision logic tests"
```

---

### Task 5: Test audit and execute_refine

**Files:**
- Test: `tests/test_validator_audit.py`

**Step 1: Write the test**

Create `prism/tests/test_validator_audit.py`:

```python
import json

from prism.lifecycle.validator import SkillValidator, AUDIT_QUALITY_THRESHOLD
from prism.skill_library.skill import Skill


def _make_llm_fn(score_dict: dict):
    def llm_fn(prompt: str) -> str:
        return json.dumps(score_dict)
    return llm_fn


LOW_SCORE = {
    "structural_completeness": 0.3, "actionability": 0.2,
    "description_alignment": 0.3, "internal_consistency": 0.3,
    "differentiation": 0.3, "issues": ["Vague"], "overall_assessment": "Low",
}

HIGH_SCORE = {
    "structural_completeness": 0.9, "actionability": 0.9,
    "description_alignment": 0.9, "internal_consistency": 0.9,
    "differentiation": 0.9, "issues": [], "overall_assessment": "Great",
}


def test_audit_identifies_low_quality():
    llm_fn = _make_llm_fn(LOW_SCORE)
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Weak", description="desc", content="stuff",
                  last_validated_step=0)
    actions = validator.audit([skill], step_count=100)
    assert len(actions) == 1
    assert actions[0].action == "REFINE"
    assert actions[0].skill_id == skill.skill_id


def test_audit_skips_recently_validated():
    llm_fn = _make_llm_fn(LOW_SCORE)
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Recent", description="desc", content="stuff",
                  last_validated_step=95)
    actions = validator.audit([skill], step_count=100)
    assert len(actions) == 0  # Skipped — validated 5 steps ago


def test_audit_skips_high_quality():
    llm_fn = _make_llm_fn(HIGH_SCORE)
    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Good", description="desc", content="good stuff",
                  last_validated_step=0)
    actions = validator.audit([skill], step_count=100)
    assert len(actions) == 0


def test_execute_refine_improves():
    """Refine replaces content when quality improves."""
    call_count = [0]
    def llm_fn(prompt: str) -> str:
        call_count[0] += 1
        if "Rewrite" in prompt:
            return "# Better\n\n## Overview\nImproved content\n\n## Workflow\n1. Do X"
        return json.dumps(HIGH_SCORE)

    validator = SkillValidator(llm_fn=llm_fn)
    skill = Skill(name="Old", description="desc", content="old stuff",
                  quality_score=0.3)
    result = validator.execute_refine(skill, reason="Low quality")
    assert result is True
    assert skill.refine_count == 1
    assert "Improved" in skill.content
```

**Step 2: Run test**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_validator_audit.py -v`
Expected: 4 passed

**Step 3: Commit**

```bash
git add tests/test_validator_audit.py
git commit -m "test: audit and execute_refine logic tests"
```

---

### Task 6: Wire validator into curator

**Files:**
- Modify: `prism/prism/lifecycle/curator.py:212-219` (SkillCurator.__init__)
- Modify: `prism/prism/lifecycle/curator.py:274-295` (_birth)
- Modify: `prism/prism/lifecycle/curator.py:331-382` (_birth_or_enrich)
- Modify: `prism/prism/lifecycle/curator.py:384-445` (_specialize)
- Test: `tests/test_curator_with_validator.py`

**Step 1: Write the failing test**

Create `prism/tests/test_curator_with_validator.py`:

```python
import json

from prism.lifecycle.curator import SkillCurator
from prism.lifecycle.reflector import ReflectionResult
from prism.lifecycle.validator import SkillValidator
from prism.skill_library.library import SkillLibrary
from prism.skill_library.skill import Skill


LOW_QUALITY = {
    "structural_completeness": 0.1, "actionability": 0.1,
    "description_alignment": 0.1, "internal_consistency": 0.1,
    "differentiation": 0.1, "issues": ["Bad"], "overall_assessment": "Bad",
}

HIGH_QUALITY = {
    "structural_completeness": 0.9, "actionability": 0.9,
    "description_alignment": 0.9, "internal_consistency": 0.9,
    "differentiation": 0.9, "issues": [], "overall_assessment": "Good",
}


def test_birth_rejected_by_validator():
    """If validator rejects birth content, skill should not be added."""
    call_idx = [0]
    def llm_fn(prompt: str) -> str:
        call_idx[0] += 1
        if "CREATE" in prompt or "ENRICH" in prompt:
            return json.dumps({
                "action": "CREATE",
                "name": "Bad Skill",
                "description": "Use when...",
                "content": "bad",
                "keywords": [],
                "task_types": ["math"],
            })
        return json.dumps(LOW_QUALITY)

    lib = SkillLibrary()
    validator = SkillValidator(llm_fn=llm_fn)
    curator = SkillCurator(library=lib, llm_fn=llm_fn, validator=validator)

    reflection = ReflectionResult(
        attributions=[], gaps=["Need a skill for X"], diagnosis=""
    )
    initial_count = len(lib)
    curator.curate(reflection=reflection, module_tag="general", task_type="math")
    # Skill should NOT have been added (validator rejected it)
    assert len(lib) == initial_count


def test_birth_accepted_by_validator():
    """If validator accepts birth content, skill should be added."""
    call_idx = [0]
    def llm_fn(prompt: str) -> str:
        call_idx[0] += 1
        if "CREATE" in prompt or "ENRICH" in prompt:
            return json.dumps({
                "action": "CREATE",
                "name": "Good Skill",
                "description": "Use when...",
                "content": "# Title\n\n## Overview\nGood\n\n## Workflow\n1. Step",
                "keywords": [],
                "task_types": ["math"],
            })
        return json.dumps(HIGH_QUALITY)

    lib = SkillLibrary()
    validator = SkillValidator(llm_fn=llm_fn)
    curator = SkillCurator(library=lib, llm_fn=llm_fn, validator=validator)

    reflection = ReflectionResult(
        attributions=[], gaps=["Need a skill for X"], diagnosis=""
    )
    initial_count = len(lib)
    curator.curate(reflection=reflection, module_tag="general", task_type="math")
    assert len(lib) == initial_count + 1


def test_curator_works_without_validator():
    """Backward compat: curator without validator should still work."""
    def llm_fn(prompt: str) -> str:
        return json.dumps({
            "action": "CREATE",
            "name": "Skill",
            "description": "Use when...",
            "content": "# T\n\nbody",
            "keywords": [],
            "task_types": ["math"],
        })

    lib = SkillLibrary()
    curator = SkillCurator(library=lib, llm_fn=llm_fn)  # No validator
    reflection = ReflectionResult(
        attributions=[], gaps=["gap"], diagnosis=""
    )
    initial_count = len(lib)
    curator.curate(reflection=reflection, module_tag="general", task_type="math")
    assert len(lib) == initial_count + 1  # Added without validation
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_curator_with_validator.py -v`
Expected: FAIL — `SkillCurator.__init__()` does not accept `validator` parameter

**Step 3: Modify curator to accept and use validator**

In `prism/prism/lifecycle/curator.py`:

**3a.** Add import at top (after line 9):
```python
from prism.lifecycle.validator import SkillValidator
```

**3b.** Modify `__init__` (lines 213-219):
```python
class SkillCurator:
    def __init__(
        self,
        library: SkillLibrary,
        llm_fn: Callable[[str], str],
        validator: SkillValidator | None = None,
    ):
        self.library = library
        self.llm_fn = llm_fn
        self.validator = validator
```

**3c.** Modify `_birth()` (lines 274-295) — add validation gate before `library.add()`:
```python
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
    ) -> Skill | None:
        skill = Skill(
            name=name,
            description=description,
            content=content,
            module_tag=module_tag,
            keywords=keywords or [],
            task_types=task_types or [task_type],
        )

        if self.validator:
            active = self.library.list_active(module_tag)
            result = self.validator.validate_and_gate(skill, active)
            if result is None:
                logger.info("[Curator] BIRTH rejected by validator: %s", name)
                return None
            skill = result

        self.library.add(skill)
        logger.info("[Curator] BIRTH: %s (%s)", skill.name, skill.skill_id)
        return skill
```

**3d.** Modify `_enrich()` (line 297-329) — validate after enrichment, reject if quality degrades:
```python
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
            skill.content = proposed
        else:
            import re
            parts = re.split(r"(?=\n## )", skill.content)
            if len(parts) > 2:
                parts.pop(1)
            trimmed = "".join(parts).rstrip()
            skill.content = trimmed + "\n\n" + new_section + "\n"
            if len(skill.content) > MAX_SKILL_CONTENT_CHARS:
                skill.content = skill.content[:MAX_SKILL_CONTENT_CHARS].rstrip() + "\n"

        # Validate enriched content — reject if quality degraded
        if self.validator:
            result = self.validator.validate(skill)
            if result.score < skill.quality_score - 0.05:
                logger.info(
                    "[Curator] ENRICH rejected (quality degraded: %.2f → %.2f): %s",
                    skill.quality_score, result.score, skill_id,
                )
                skill.content = old_content
                return False
            skill.quality_score = result.score

        logger.info("[Curator] ENRICH: %s with content: %s", skill.skill_id, enrich_content[:50])
        return True
```

**3e.** Modify `_specialize()` — validate each child (lines 422-434):

After creating each child Skill, add validation before `library.add()`:
```python
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

                    if self.validator:
                        active = self.library.list_active(module_tag)
                        result = self.validator.validate_and_gate(child, active)
                        if result is None:
                            logger.info("[Curator] SPECIALIZE child rejected: %s", child.name)
                            continue
                        child = result

                    self.library.add(child)
                    child_ids.append(child.skill_id)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_curator_with_validator.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add prism/prism/lifecycle/curator.py tests/test_curator_with_validator.py
git commit -m "feat: wire SkillValidator into curator for BIRTH/ENRICH/SPECIALIZE gating"
```

---

### Task 7: Wire validator into engine + periodic audit

**Files:**
- Modify: `prism/prism/engine.py:19-56` (PRISMEngine.__init__)
- Modify: `prism/prism/engine.py:215-219` (_maintenance)
- Modify: `prism/prism/lifecycle/__init__.py`
- Test: `tests/test_engine_validator.py`

**Step 1: Write the failing test**

Create `prism/tests/test_engine_validator.py`:

```python
from prism.engine import PRISMEngine


def test_engine_creates_validator():
    """Engine should create a SkillValidator and pass it to curator."""
    def dummy_eval(prompt, task):
        return 1.0, "trace", "output"
    def dummy_llm(prompt):
        return '{"attributions": [], "gaps": [], "diagnosis": ""}'

    engine = PRISMEngine(
        evaluate_fn=dummy_eval,
        llm_fn=dummy_llm,
        audit_interval=20,
    )
    assert engine.validator is not None
    assert engine.curator.validator is not None
    assert engine.audit_interval == 20


def test_engine_default_audit_interval():
    """Default audit_interval should be 20."""
    def dummy_eval(prompt, task):
        return 1.0, "trace", "output"
    def dummy_llm(prompt):
        return '{}'

    engine = PRISMEngine(evaluate_fn=dummy_eval, llm_fn=dummy_llm)
    assert engine.audit_interval == 20
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_engine_validator.py -v`
Expected: FAIL — `PRISMEngine` has no `validator` attribute

**Step 3: Modify engine.py**

In `prism/prism/engine.py`:

**3a.** Add import (after line 10):
```python
from prism.lifecycle.validator import SkillValidator
```

**3b.** Add `audit_interval` parameter to `__init__` and create validator (modify lines 19-56):
```python
class PRISMEngine:
    def __init__(
        self,
        evaluate_fn: EvaluateFn,
        llm_fn: Callable[[str], str],
        embed_fn: Callable[[str], list[float]] | None = None,
        library_path: str | None = None,
        index_path: str | None = None,
        base_prompt: str = "",
        top_k: int = 5,
        token_budget: int = 2000,
        maintenance_interval: int = 10,
        enable_differential_eval: bool = True,
        audit_interval: int = 20,
    ):
        self.evaluate_fn = evaluate_fn
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn
        self.base_prompt = base_prompt
        self.top_k = top_k
        self.token_budget = token_budget
        self.maintenance_interval = maintenance_interval
        self.enable_differential_eval = enable_differential_eval
        self.audit_interval = audit_interval

        self.library = SkillLibrary(path=library_path)
        self.task_index = TaskTypeIndex(path=index_path)
        self.assembler = SkillAssembler(
            library=self.library,
            task_index=self.task_index,
            embed_fn=embed_fn,
            llm_fn=llm_fn,
            top_k=top_k,
            token_budget=token_budget,
        )
        self.reflector = PRISMReflector(llm_fn=llm_fn)
        self.validator = SkillValidator(llm_fn=llm_fn)
        self.curator = SkillCurator(
            library=self.library, llm_fn=llm_fn, validator=self.validator
        )

        self._step_count = 0
        self._history: list[dict[str, Any]] = []
```

**3c.** Add audit to `_maintenance()` (modify lines 215-219):
```python
    def _maintenance(self) -> None:
        """Periodic maintenance: save state + audit skills."""
        logger.info("[Maintenance] Saving library and index...")
        self.library.save()
        self.task_index.save()

        # Periodic quality audit
        if self._step_count % self.audit_interval == 0:
            active = self.library.list_active()
            actions = self.validator.audit(active, self._step_count)
            for action in actions:
                if action.action == "REFINE":
                    skill = self.library.get(action.skill_id)
                    if skill:
                        self.validator.execute_refine(skill, action.reason)
                        logger.info(
                            "[Maintenance] REFINE: %s — %s",
                            action.skill_id, action.reason,
                        )
```

**3d.** Update `prism/prism/lifecycle/__init__.py`:
```python
from prism.lifecycle.reflector import PRISMReflector
from prism.lifecycle.curator import SkillCurator
from prism.lifecycle.validator import SkillValidator

__all__ = ["PRISMReflector", "SkillCurator", "SkillValidator"]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/test_engine_validator.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add prism/prism/engine.py prism/prism/lifecycle/__init__.py tests/test_engine_validator.py
git commit -m "feat: wire SkillValidator into PRISMEngine with periodic audit"
```

---

### Task 8: Run all tests + import smoke test

**Files:**
- All test files created above
- All modified source files

**Step 1: Run full test suite**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m pytest tests/ -v`
Expected: All tests pass (approximately 17 tests across 5 files)

**Step 2: Run import smoke test**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -c "from prism import PRISMEngine, SkillLibrary, Skill; from prism.lifecycle.validator import SkillValidator, ValidationResult, AuditAction; print('OK')"`
Expected: `OK`

**Step 3: Run AIME integration test**

Run: `cd /Users/steven/skills-evolve/prism && conda run -n prism python -m evaluate.aime2025 --problems 1-3 --epochs 1`
Expected: Pipeline runs, logs show `[Validator] ACCEPT` or `[Validator] DISCARD` entries for BIRTH/ENRICH operations

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete skill quality gate implementation"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Add quality fields to Skill | `skill.py`, test |
| 2 | Persist quality fields | `library.py`, test |
| 3 | Create SkillValidator | `validator.py`, test |
| 4 | Test validate_and_gate | test |
| 5 | Test audit + refine | test |
| 6 | Wire into curator | `curator.py`, test |
| 7 | Wire into engine | `engine.py`, `__init__.py`, test |
| 8 | Integration test | all files |
