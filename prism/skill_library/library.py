from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from prism.skill_library.skill import Skill

logger = logging.getLogger(__name__)


class SkillLibrary:
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
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [skill.to_dict() for skill in self._skills.values()]
        self._path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for entry in data:
                skill = Skill.from_dict(entry)
                self._skills[skill.skill_id] = skill
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load skill library: %s", e)

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
