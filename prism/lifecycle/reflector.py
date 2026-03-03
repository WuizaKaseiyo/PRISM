from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from prism.utils import extract_json_from_text

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    attributions: list[dict[str, str]]  # [{"skill_id": ..., "tag": "helpful"|"harmful"|"neutral"}]
    gaps: list[str]  # descriptions of missing knowledge
    diagnosis: str  # overall analysis

    @classmethod
    def empty(cls) -> ReflectionResult:
        return cls(attributions=[], gaps=[], diagnosis="")


REFLECTOR_PROMPT = """Analyze the following execution trace and determine:
1. Which skills were helpful, harmful, or neutral
2. What knowledge gaps exist that new skills could fill
3. An overall diagnosis of performance

Task: {task}
Score: {score}
Trace: {trace}
Output: {output}
Skills used: {skill_ids}

Return ONLY a JSON object with this structure:
{{
  "attributions": [
    {{"skill_id": "...", "tag": "helpful|harmful|neutral"}}
  ],
  "gaps": ["description of missing knowledge", ...],
  "diagnosis": "overall analysis"
}}"""


class PRISMReflector:
    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn

    def reflect(
        self,
        task: dict[str, Any],
        score: float,
        trace: str,
        output: str,
        skill_ids: list[str],
    ) -> ReflectionResult:
        prompt = REFLECTOR_PROMPT.format(
            task=json.dumps(task),
            score=score,
            trace=trace,
            output=output,
            skill_ids=json.dumps(skill_ids),
        )

        try:
            response = self.llm_fn(prompt)
            parsed = extract_json_from_text(response)
            if parsed is None:
                logger.warning("[Reflector] Warning: failed to parse JSON response")
                return ReflectionResult.empty()

            attributions = parsed.get("attributions", [])
            gaps = parsed.get("gaps", [])
            diagnosis = parsed.get("diagnosis", "")

            return ReflectionResult(
                attributions=attributions,
                gaps=gaps,
                diagnosis=diagnosis,
            )
        except Exception as e:
            logger.warning("[Reflector] Error during reflection: %s", e)
            return ReflectionResult.empty()


