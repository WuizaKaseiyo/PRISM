from __future__ import annotations

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
        )
