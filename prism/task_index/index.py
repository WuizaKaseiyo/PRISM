from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_KEYWORD_MAP: dict[str, list[str]] = {
    "math": ["calculate", "equation", "solve", "arithmetic", "algebra", "math", "number", "sum", "product"],
    "code": ["code", "program", "function", "debug", "implement", "python", "javascript", "algorithm"],
    "writing": ["write", "essay", "story", "summarize", "paraphrase", "draft", "compose", "article"],
    "reasoning": ["reason", "logic", "deduce", "infer", "analyze", "think", "evaluate", "argue"],
    "qa": ["question", "answer", "what", "who", "when", "where", "why", "how", "explain"],
}


class TaskTypeIndex:
    def __init__(
        self,
        path: str | Path | None = None,
        keyword_map: dict[str, list[str]] | None = None,
        alpha: float = 0.3,
    ):
        self._index: dict[str, dict[str, float]] = {}
        self._path = Path(path) if path else None
        self._keyword_map = keyword_map or DEFAULT_KEYWORD_MAP
        self._alpha = alpha
        if self._path and self._path.exists():
            self._load()

    def classify(self, text: str) -> str:
        text_lower = text.lower()
        best_type = "general"
        best_count = 0
        for task_type, keywords in self._keyword_map.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_type = task_type
        return best_type

    def update(self, task_type: str, skill_id: str, score: float) -> None:
        if task_type not in self._index:
            self._index[task_type] = {}
        current = self._index[task_type].get(skill_id, score)
        self._index[task_type][skill_id] = (1 - self._alpha) * current + self._alpha * score

    def top_skills(self, task_type: str, n: int = 10) -> list[tuple[str, float]]:
        skills = self._index.get(task_type, {})
        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        return sorted_skills[:n]

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._index, indent=2))

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            self._index = json.loads(self._path.read_text())
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load task index: %s", e)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._index)
