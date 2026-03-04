from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Skill:
    name: str
    description: str
    content: str
    module_tag: str = "general"
    skill_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    trigger_conditions: str = ""
    strategy_notes: list[str] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    helpful_count: int = 0
    harmful_count: int = 0
    neutral_count: int = 0
    eval_scores: list[float] = field(default_factory=list)
    status: str = "active"  # active | retired
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    score_matrix: dict[str, float] = field(default_factory=dict)
    pareto_frequency: float = 0.0

    @staticmethod
    def task_key(question: str) -> str:
        """8-char md5 hash of question text for per-instance score tracking."""
        return hashlib.md5(question.encode()).hexdigest()[:8]

    @property
    def slug(self) -> str:
        """Kebab-case directory name derived from skill name."""
        s = self.name.lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        return s.strip("-")

    @property
    def score_variance(self) -> float:
        if len(self.eval_scores) < 2:
            return 0.0
        mean = sum(self.eval_scores) / len(self.eval_scores)
        return sum((s - mean) ** 2 for s in self.eval_scores) / len(self.eval_scores)

    @property
    def net_value(self) -> int:
        return self.helpful_count - self.harmful_count

    @property
    def total_evals(self) -> int:
        return self.helpful_count + self.harmful_count + self.neutral_count

    # --- Markdown serialization (Claude Code SKILL.md format) ---

    def to_markdown(self) -> str:
        """Serialize to Claude Code SKILL.md format.

        Only name + description go in YAML frontmatter.
        All operational content goes in the markdown body.
        Internal tracking fields are stored separately in _meta.json.
        """
        lines = ["---"]
        lines.append(f"name: {self.slug}")
        lines.append(f"description: {_yaml_escape(self.description)}")
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, text: str) -> Skill:
        """Parse a Claude Code SKILL.md file.

        Expects only name + description in frontmatter.
        Everything below the closing --- is the content body.
        """
        match = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
        if not match:
            raise ValueError("Invalid skill markdown: missing YAML frontmatter")

        frontmatter_text = match.group(1)
        body = match.group(2).strip()

        # Parse YAML frontmatter (simple key: value parser)
        meta: dict[str, Any] = {}
        for line in frontmatter_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            meta[key] = value

        # Derive display name from slug: "basic-arithmetic" → "Basic Arithmetic"
        slug = meta.get("name", "unnamed")
        display_name = slug.replace("-", " ").title()

        return cls(
            name=display_name,
            description=meta.get("description", ""),
            content=body,
        )

    # --- Dict serialization (kept for backward compat / index metadata) ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "module_tag": self.module_tag,
            "trigger_conditions": self.trigger_conditions,
            "strategy_notes": list(self.strategy_notes),
            "pitfalls": list(self.pitfalls),
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "neutral_count": self.neutral_count,
            "eval_scores": list(self.eval_scores),
            "status": self.status,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "created_at": self.created_at,
            "task_types": list(self.task_types),
            "keywords": list(self.keywords),
            "embedding": self.embedding,
            "score_matrix": dict(self.score_matrix),
            "pareto_frequency": self.pareto_frequency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            description=data["description"],
            content=data["content"],
            module_tag=data.get("module_tag", "general"),
            trigger_conditions=data.get("trigger_conditions", ""),
            strategy_notes=data.get("strategy_notes", []),
            pitfalls=data.get("pitfalls", []),
            helpful_count=data.get("helpful_count", 0),
            harmful_count=data.get("harmful_count", 0),
            neutral_count=data.get("neutral_count", 0),
            eval_scores=data.get("eval_scores", []),
            status=data.get("status", "active"),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            task_types=data.get("task_types", []),
            keywords=data.get("keywords", []),
            embedding=data.get("embedding"),
            score_matrix=data.get("score_matrix", {}),
            pareto_frequency=data.get("pareto_frequency", 0.0),
        )


def _yaml_escape(s: str) -> str:
    """Escape a string for YAML value if it contains special characters."""
    if any(c in s for c in ":#{}[]|>&*!?,"):
        return f'"{s}"'
    return s
