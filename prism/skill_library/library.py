from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from prism.skill_library.skill import Skill

logger = logging.getLogger(__name__)


class SkillLibrary:
    """Skill library with Claude Code directory-per-skill persistence.

    Storage layout:
        skills/
        ├── basic-arithmetic/
        │   └── SKILL.md              # name + description frontmatter, rich body
        ├── exact-format-matching/
        │   └── SKILL.md
        └── _meta.json                # ALL PRISM tracking data keyed by skill_id
    """

    def __init__(self, path: str | Path | None = None):
        self._skills: dict[str, Skill] = {}
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self._load()

    def add(self, skill: Skill) -> str:
        self._skills[skill.skill_id] = skill
        return skill.skill_id

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def update(self, skill_id: str, **kwargs: Any) -> None:
        skill = self._skills.get(skill_id)
        if skill is None:
            logger.warning("Skill %s not found for update", skill_id)
            return
        for key, value in kwargs.items():
            if hasattr(skill, key):
                setattr(skill, key, value)

    def retire(self, skill_id: str) -> None:
        skill = self._skills.get(skill_id)
        if skill:
            skill.status = "retired"

    def filter(
        self,
        module_tag: str | None = None,
        task_type: str | None = None,
        status: str | None = None,
    ) -> list[Skill]:
        results = list(self._skills.values())
        if module_tag is not None:
            results = [s for s in results if s.module_tag == module_tag]
        if task_type is not None:
            results = [s for s in results if task_type in s.task_types]
        if status is not None:
            results = [s for s in results if s.status == status]
        return results

    def list_active(self, module_tag: str | None = None) -> list[Skill]:
        return self.filter(module_tag=module_tag, status="active")

    def save(self) -> None:
        """Save each skill as a directory with SKILL.md + all tracking data in _meta.json."""
        if self._path is None:
            return
        self._path.mkdir(parents=True, exist_ok=True)

        # Track which slug dirs belong to current skills
        existing_dirs = {
            d for d in self._path.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        }
        current_dirs: set[Path] = set()

        # Detect slug collisions and resolve with skill_id suffix
        slug_counts: dict[str, list[Skill]] = {}
        for skill in self._skills.values():
            slug_counts.setdefault(skill.slug, []).append(skill)

        slug_map: dict[str, str] = {}  # skill_id → final directory name
        for slug, skills in slug_counts.items():
            if len(skills) == 1:
                slug_map[skills[0].skill_id] = slug
            else:
                for s in skills:
                    slug_map[s.skill_id] = f"{slug}-{s.skill_id[:6]}"

        # Write each skill directory
        for skill in self._skills.values():
            dir_name = slug_map[skill.skill_id]
            skill_dir = self._path / dir_name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(skill.to_markdown())
            current_dirs.add(skill_dir)

        # Remove stale skill directories
        import shutil
        for stale in existing_dirs - current_dirs:
            shutil.rmtree(stale)

        # Write _meta.json with ALL PRISM tracking fields
        meta: dict[str, Any] = {}
        for skill in self._skills.values():
            meta[skill.skill_id] = {
                "slug": slug_map[skill.skill_id],
                "name": skill.name,
                "skill_id": skill.skill_id,
                "status": skill.status,
                "module_tag": skill.module_tag,
                "helpful_count": skill.helpful_count,
                "harmful_count": skill.harmful_count,
                "neutral_count": skill.neutral_count,
                "eval_scores": skill.eval_scores,
                "parent_id": skill.parent_id,
                "children_ids": skill.children_ids,
                "keywords": skill.keywords,
                "task_types": skill.task_types,
                "trigger_conditions": skill.trigger_conditions,
                "strategy_notes": skill.strategy_notes,
                "pitfalls": skill.pitfalls,
                "created_at": skill.created_at,
                "embedding": skill.embedding,
                "score_matrix": dict(skill.score_matrix),
                "pareto_frequency": skill.pareto_frequency,
            }
        meta_path = self._path / "_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

    def _load(self) -> None:
        """Load skills from */SKILL.md + merge all tracking fields from _meta.json."""
        if self._path is None or not self._path.exists():
            return

        # Load tracking metadata first
        meta: dict[str, Any] = {}
        meta_path = self._path / "_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError as e:
                logger.warning("Failed to load _meta.json: %s", e)

        # Build reverse lookup: slug → skill_id from meta
        slug_to_id: dict[str, str] = {}
        for skill_id, entry in meta.items():
            slug_to_id[entry.get("slug", "")] = skill_id

        # Load each SKILL.md from subdirectories
        for skill_md in sorted(self._path.glob("*/SKILL.md")):
            try:
                skill = Skill.from_markdown(skill_md.read_text())
                dir_name = skill_md.parent.name

                # Resolve skill_id from meta via directory slug
                skill_id = slug_to_id.get(dir_name)
                if skill_id and skill_id in meta:
                    m = meta[skill_id]
                    skill.skill_id = skill_id
                    skill.name = m.get("name", skill.name)
                    skill.status = m.get("status", "active")
                    skill.module_tag = m.get("module_tag", "general")
                    skill.helpful_count = m.get("helpful_count", 0)
                    skill.harmful_count = m.get("harmful_count", 0)
                    skill.neutral_count = m.get("neutral_count", 0)
                    skill.eval_scores = m.get("eval_scores", [])
                    skill.parent_id = m.get("parent_id")
                    skill.children_ids = m.get("children_ids", [])
                    skill.keywords = m.get("keywords", [])
                    skill.task_types = m.get("task_types", [])
                    skill.trigger_conditions = m.get("trigger_conditions", "")
                    skill.strategy_notes = m.get("strategy_notes", [])
                    skill.pitfalls = m.get("pitfalls", [])
                    skill.created_at = m.get("created_at", skill.created_at)
                    skill.embedding = m.get("embedding")
                    skill.score_matrix = m.get("score_matrix", {})
                    skill.pareto_frequency = m.get("pareto_frequency", 0.0)

                self._skills[skill.skill_id] = skill
            except (ValueError, KeyError) as e:
                logger.warning("Failed to load skill from %s: %s", skill_md, e)

        # Backward compat: also try loading old JSON format
        json_path = self._path.parent / (self._path.name + ".json") if self._path.suffix == "" else None
        if json_path and json_path.exists():
            self._load_legacy_json(json_path)

    def _load_legacy_json(self, json_path: Path) -> None:
        """Load skills from legacy single-file JSON format."""
        try:
            data = json.loads(json_path.read_text())
            for entry in data:
                skill = Skill.from_dict(entry)
                if skill.skill_id not in self._skills:
                    self._skills[skill.skill_id] = skill
            logger.info("Loaded %d skills from legacy JSON: %s", len(data), json_path)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load legacy JSON: %s", e)

    def to_playbook_text(self) -> str:
        lines = []
        for skill in self._skills.values():
            if skill.status == "active":
                lines.append(
                    f"[{skill.skill_id}] helpful={skill.helpful_count} "
                    f"harmful={skill.harmful_count} :: {skill.content}"
                )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_module: dict[str, int] = {}
        for skill in self._skills.values():
            by_status[skill.status] = by_status.get(skill.status, 0) + 1
            by_module[skill.module_tag] = by_module.get(skill.module_tag, 0) + 1
        return {
            "total": len(self._skills),
            "by_status": by_status,
            "by_module": by_module,
        }

    def __len__(self) -> int:
        return len(self._skills)
