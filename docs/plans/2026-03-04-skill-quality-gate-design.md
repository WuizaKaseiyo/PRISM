# Skill Quality Gate — Design Document

**Date**: 2026-03-04
**Status**: Approved
**Goal**: Add content-level quality control for PRISM skills — currently, helpful/harmful counts measure task outcomes, not whether the skill content itself is well-written, actionable, and internally consistent.

---

## Problem Statement

PRISM's lifecycle operations (BIRTH, ENRICH, SPECIALIZE) generate skill content via LLM, but there is no validation of the generated content's quality. The only feedback signal is task-level performance (helpful/harmful attribution), which conflates skill content quality with task difficulty, model capability, and retrieval accuracy.

**Symptoms**:
- BIRTH creates skills that are vague, repetitive, or overlap with existing skills
- ENRICH appends content that contradicts or bloats the original skill
- SPECIALIZE produces children worse than the parent
- No mechanism to improve existing skill content over time

**Root cause**: No direct measurement or enforcement of skill *content* quality.

---

## Architecture

### New Component: `SkillValidator`

**Location**: `prism/lifecycle/validator.py` (L3, same layer as curator and reflector)

**Responsibility boundary**:

| Component | Evaluates | Decides |
|-----------|-----------|---------|
| `reflector.py` | How well the task was done | Attributions, gaps |
| `curator.py` | Should the skill live or die | BIRTH/RETIRE/SPECIALIZE/GENERALIZE |
| **`validator.py`** | Is the skill content good | Accept/reject/revise/refine |

### Two Core Methods

```
SkillValidator
├── validate(skill, active_skills?) → ValidationResult
│   - Structural checks (headings, length, sections)
│   - LLM quality scoring (5 dimensions, 0-1 score)
│   - If score < threshold: generate revised_content
│   - Returns: score, issues[], revised_content?
│
└── audit(skills, step_count) → list[AuditAction]
    - Batch quality assessment of all active skills
    - Identifies REFINE candidates
    - Returns: list of (skill_id, action, reason)
```

### New Lifecycle Operation: `REFINE`

Unlike ENRICH (append content), REFINE **rewrites** the entire skill content while preserving the skill's identity (ID, stats, lineage).

**When triggered**: During periodic audit, for skills with degraded quality scores or excessive enrichment history.

---

## Detailed Design

### 1. `validate()` — Generation-Time Gate

**Trigger**: Called immediately after BIRTH, ENRICH, or SPECIALIZE produces new content.

#### Evaluation Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Structural completeness | 0.20 | Has `#` title, `##` sections, 50+ lines, workflow/steps present |
| Actionability | 0.30 | Contains concrete steps, methods, formulas — not just vague principles |
| Description alignment | 0.20 | Content actually teaches what the description claims |
| Internal consistency | 0.15 | No self-contradictions, no redundant repetition |
| Differentiation | 0.15 | Sufficiently different from existing active skills |

#### LLM Prompt Structure

```
Rate this skill's content quality on 5 dimensions (0.0 to 1.0 each):

Skill name: {name}
Description: {description}
Content:
{content}

Existing active skills (for differentiation check):
{active_skill_summaries}

Return JSON:
{
  "structural_completeness": 0.0-1.0,
  "actionability": 0.0-1.0,
  "description_alignment": 0.0-1.0,
  "internal_consistency": 0.0-1.0,
  "differentiation": 0.0-1.0,
  "issues": ["issue1", "issue2"],
  "overall_assessment": "brief text"
}
```

#### Decision Logic

```python
QUALITY_ACCEPT = 0.7    # Accept directly
QUALITY_REVISE = 0.4    # Try one revision
QUALITY_REJECT = 0.4    # Below this → discard

result = validator.validate(skill, active_skills)

if result.score >= QUALITY_ACCEPT:
    # Pass → activate skill as-is
    library.add(skill)

elif result.score >= QUALITY_REVISE:
    # Marginal → LLM revises, then re-validate
    revised = validator.revise(skill, result.issues)
    result2 = validator.validate(revised, active_skills)
    if result2.score >= 0.6:  # Lower bar for revision
        library.add(revised)
    else:
        logger.info("Skill discarded after failed revision: %s", skill.name)

else:
    # Too poor → discard immediately
    logger.info("Skill discarded (score=%.2f): %s", result.score, skill.name)
```

#### ENRICH Special Handling

For ENRICH operations, validate the **complete skill content after appending** (not just the new section). If the enriched version scores lower than the pre-enrichment version, reject the enrichment.

```python
pre_score = validator.validate(skill_before_enrich).score
post_score = validator.validate(skill_after_enrich).score
if post_score < pre_score - 0.05:  # Quality degraded
    reject enrichment, keep original
```

### 2. `audit()` — Periodic Deep Review

**Trigger**: Called in `PRISMEngine._maintenance()` every `audit_interval` steps (default: 20).

#### Audit Flow

1. **Collect signals** for each active skill:
   - Content quality score (same LLM scoring as validate)
   - Performance data: `pareto_frequency`, harmful/helpful ratio, `score_matrix` trend
   - Content age: steps since last REFINE or creation
   - Enrichment count: how many times ENRICH has appended content

2. **Identify REFINE candidates** — any of:
   - Content quality score < 0.6
   - `helpful_ratio` high but `harmful_ratio` also non-trivial (content has value but needs improvement)
   - Enriched 3+ times (likely bloated/messy from repeated appends)
   - Content age > 50 steps with no quality check

3. **Execute REFINE**:
   ```
   LLM receives:
   - Current skill content
   - Performance data (helpful/harmful, score_matrix, pareto_frequency)
   - Quality issues identified
   - Original description (must be preserved)

   LLM outputs:
   - Rewritten content (complete replacement)
   - Preserves the skill's core purpose but improves clarity, structure, actionability
   ```

4. **Post-REFINE validation**: The rewritten content goes through `validate()` — only replace if the new version scores higher than the old.

5. **Audit logging**: Record quality scores and actions for trend tracking.

#### REFINE Prompt Structure

```
Rewrite this skill to improve its quality. Keep the same purpose and knowledge,
but make it clearer, better structured, and more actionable.

Skill: {name}
Description (DO NOT CHANGE): {description}
Current content:
{content}

Performance data:
- Helpful: {helpful_count}, Harmful: {harmful_count}
- Pareto frequency: {pareto_frequency:.2f}
- Quality issues: {issues}

Write a complete replacement for the content field.
Requirements:
- # Title heading
- ## Overview section
- ## Workflow section with numbered steps
- ## Key Principles or ## Examples sections
- 50-200 lines, concrete and actionable
- No contradictions with the description

Return the rewritten content as plain markdown (no JSON wrapper).
```

### 3. Skill Dataclass Changes

Add to `Skill` in `skill.py`:

```python
quality_score: float = 0.0           # Latest quality assessment (0-1)
last_validated_step: int = 0         # Step count when last validated
refine_count: int = 0                # Number of times REFINE has rewritten content
```

Update `to_dict()`, `from_dict()`, `save()`, `_load()` with backward-compatible defaults.

### 4. Integration Points

#### In `curator.py`

```python
class SkillCurator:
    def __init__(self, library, llm_fn, validator=None):
        self.validator = validator  # Optional, backward compatible

    def _birth_or_enrich(self, ...):
        # After creating skill via LLM:
        if self.validator:
            result = self.validator.validate(new_skill, active_skills)
            if result.score < QUALITY_ACCEPT:
                # Revision/rejection logic
                ...
        library.add(skill)  # Only if passed validation

    def _specialize(self, ...):
        # After creating child skills:
        if self.validator:
            for child in children:
                result = self.validator.validate(child, active_skills)
                # Same accept/revise/reject logic
```

#### In `engine.py`

```python
class PRISMEngine:
    def __init__(self, ..., audit_interval=20):
        self.validator = SkillValidator(llm_fn=llm_fn)
        self.curator = SkillCurator(library, llm_fn, validator=self.validator)
        self.audit_interval = audit_interval

    def _maintenance(self):
        # Existing: save library and index
        self.library.save()
        self.task_index.save()

        # New: periodic audit
        if self._step_count % self.audit_interval == 0:
            active = self.library.list_active()
            actions = self.validator.audit(active, self._step_count)
            for action in actions:
                if action.type == "REFINE":
                    self._execute_refine(action)
```

---

## Data Structures

### ValidationResult

```python
@dataclass
class ValidationResult:
    score: float                          # Weighted average, 0-1
    dimensions: dict[str, float]          # Per-dimension scores
    issues: list[str]                     # Human-readable issues
    revised_content: str | None = None    # LLM-generated revision (if score was low)
```

### AuditAction

```python
@dataclass
class AuditAction:
    skill_id: str
    action: str          # "REFINE" or "RETIRE"
    reason: str
    quality_score: float
```

---

## Files to Modify

| File | Change |
|------|--------|
| `prism/lifecycle/validator.py` | **NEW** — SkillValidator class, validate(), audit(), revise() |
| `prism/skill_library/skill.py` | Add `quality_score`, `last_validated_step`, `refine_count` fields |
| `prism/skill_library/library.py` | Add new fields to save/load |
| `prism/lifecycle/curator.py` | Accept optional `validator`, gate BIRTH/ENRICH/SPECIALIZE output |
| `prism/engine.py` | Create validator, pass to curator, add audit to maintenance |

---

## Constants

```python
# validator.py
QUALITY_ACCEPT = 0.70           # Accept skill as-is
QUALITY_REVISE = 0.40           # Try one LLM revision
REVISION_ACCEPT = 0.60          # Lower bar for revised content

# Dimension weights
WEIGHT_STRUCTURAL = 0.20
WEIGHT_ACTIONABILITY = 0.30
WEIGHT_ALIGNMENT = 0.20
WEIGHT_CONSISTENCY = 0.15
WEIGHT_DIFFERENTIATION = 0.15

# Audit triggers
AUDIT_QUALITY_THRESHOLD = 0.60  # Below this → REFINE candidate
AUDIT_ENRICH_COUNT = 3          # 3+ enrichments → likely needs REFINE
AUDIT_MAX_AGE = 50              # Steps since last validation → force re-check
```

---

## Backward Compatibility

- `validator` parameter is optional in `SkillCurator.__init__()` — if `None`, all validation is skipped (existing behavior preserved)
- New Skill fields default to `0.0` / `0` — old skills loaded from disk work unchanged
- `_meta.json` uses `.get()` with defaults — old format is forward-compatible
- `REFINE` is a new operation name in `curate()` return list — downstream consumers treat ops list as informational

---

## Verification Plan

1. **Import check**: `from prism.lifecycle.validator import SkillValidator` works
2. **Unit test validate()**: Create a skill with poor content → score < 0.7; create one with good content → score >= 0.7
3. **Integration test**: Run `python -m evaluate.aime2025 --problems 1-5 --epochs 2` → check logs for validation accept/reject/revise events
4. **REFINE test**: After 20+ steps, verify audit triggers REFINE on low-quality skills; check `_meta.json` for `refine_count > 0`
5. **Backward compat**: Load existing `data/aime_skills/` → no errors, new fields default correctly
6. **Quality trend**: After 2+ epochs, compare average `quality_score` of active skills — should trend upward
