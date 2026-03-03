from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from prism.assembler.assembler import SkillAssembler
from prism.lifecycle.curator import SkillCurator
from prism.lifecycle.reflector import PRISMReflector
from prism.skill_library.library import SkillLibrary
from prism.task_index.index import TaskTypeIndex

logger = logging.getLogger(__name__)

try:
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: F401
except ImportError:
    # Graceful degradation: define minimal stubs so the module can be imported
    # without gepa installed. Actual usage will fail at runtime.
    class EvaluationBatch:  # type: ignore[no-redef]
        def __init__(
            self,
            outputs: list,
            scores: list[float],
            trajectories: list | None = None,
            objective_scores: list[dict[str, float]] | None = None,
        ):
            self.outputs = outputs
            self.scores = scores
            self.trajectories = trajectories
            self.objective_scores = objective_scores

    class GEPAAdapter:  # type: ignore[no-redef]
        pass


def _batch_to_task_repr(batch: list[Any]) -> dict[str, Any]:
    """Extract a meaningful text representation from batch items for skill retrieval.

    Handles common GEPA DataInst patterns:
    - TypedDict/dict with 'input' key (DefaultAdapter pattern)
    - TypedDict/dict with 'question' key
    - Plain strings
    - Other dicts: serialized as JSON
    """
    texts: list[str] = []
    for item in batch[:5]:  # Sample first 5 items for efficiency
        if isinstance(item, dict):
            # Try common keys in order of specificity
            text = item.get("input") or item.get("question") or item.get("text", "")
            if not text:
                text = json.dumps(item, default=str)[:200]
            texts.append(str(text)[:200])
        elif isinstance(item, str):
            texts.append(item[:200])
        else:
            texts.append(str(item)[:200])
    return {"batch_text": " | ".join(texts), "batch_size": len(batch)}


class PRISMWrappedAdapter:
    """Wraps any GEPAAdapter, intercepting evaluate() and make_reflective_dataset()
    to integrate PRISM's skill assembly and lifecycle management.

    Usage with gepa.optimize():
        inner = MyAdapter(...)
        wrapped = PRISMWrappedAdapter(inner, library, task_index, llm_fn)
        result = gepa.optimize(seed_candidate, trainset, adapter=wrapped, ...)
    """

    def __init__(
        self,
        inner_adapter: Any,  # GEPAAdapter instance
        library: SkillLibrary,
        task_index: TaskTypeIndex,
        llm_fn: Callable[[str], str],
        embed_fn: Callable[[str], list[float]] | None = None,
        module_tag: str = "general",
        top_k: int = 5,
        token_budget: int = 2000,
        inject_components: list[str] | None = None,
    ):
        self.inner = inner_adapter
        self.library = library
        self.task_index = task_index
        self.llm_fn = llm_fn
        self.module_tag = module_tag
        self._inject_components = inject_components  # None = inject into all

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

        self._last_skill_ids: list[str] = []
        self._last_skill_context: str = ""

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Intercept evaluate: assemble skills, inject into candidate, delegate.

        Follows GEPAAdapter.evaluate() contract:
        - Returns EvaluationBatch with len(outputs) == len(scores) == len(batch)
        - If capture_traces=True, trajectories are populated
        - Never raises for individual example failures
        """
        # Build task representation from actual batch data
        task_repr = _batch_to_task_repr(batch)
        skill_context, skill_ids = self.assembler.assemble(task_repr, module_tag=self.module_tag)
        self._last_skill_ids = skill_ids
        self._last_skill_context = skill_context

        # Inject skill context into candidate components (copy, never mutate original)
        augmented_candidate = dict(candidate)
        if skill_context:
            target_keys = self._inject_components or list(augmented_candidate.keys())
            for key in target_keys:
                if key in augmented_candidate:
                    augmented_candidate[key] = (
                        augmented_candidate[key] + "\n\n## PRISM Skills\n" + skill_context
                    )

        # Delegate to inner adapter
        eval_batch = self.inner.evaluate(batch, augmented_candidate, capture_traces)

        # Reflect and curate on traces (only when capturing, to avoid double cost)
        if capture_traces and eval_batch.trajectories:
            self._reflect_and_curate(task_repr, eval_batch, skill_ids)

        logger.info(
            "[PRISMWrappedAdapter] evaluate: %d examples, %d skills injected, avg_score=%.3f",
            len(batch),
            len(skill_ids),
            sum(eval_batch.scores) / len(eval_batch.scores) if eval_batch.scores else 0.0,
        )

        return eval_batch

    def _reflect_and_curate(
        self,
        task_repr: dict[str, Any],
        eval_batch: EvaluationBatch,
        skill_ids: list[str],
    ) -> None:
        """Run reflection and curation on evaluation results."""
        avg_score = sum(eval_batch.scores) / len(eval_batch.scores) if eval_batch.scores else 0.0

        # Build a richer trace summary from trajectories
        trace_parts: list[str] = []
        for i, traj in enumerate(eval_batch.trajectories or []):
            if i >= 3:
                break
            if isinstance(traj, dict):
                # Handle GEPA DefaultTrajectory format: {data, full_assistant_response, feedback}
                feedback = traj.get("feedback", "")
                response = str(traj.get("full_assistant_response", ""))[:200]
                trace_parts.append(f"Example {i}: feedback={feedback}, output={response}")
            else:
                trace_parts.append(f"Example {i}: {str(traj)[:200]}")
        trace_summary = "\n".join(trace_parts)

        # Build output summary
        output_parts: list[str] = []
        for i, out in enumerate(eval_batch.outputs[:3]):
            if isinstance(out, dict):
                output_parts.append(str(out.get("full_assistant_response", out))[:200])
            else:
                output_parts.append(str(out)[:200])
        output_summary = " | ".join(output_parts)

        reflection = self.reflector.reflect(
            task=task_repr,
            score=avg_score,
            trace=trace_summary,
            output=output_summary,
            skill_ids=skill_ids,
        )

        task_type = self.task_index.classify(task_repr.get("batch_text", ""))
        self.curator.curate(
            reflection=reflection,
            module_tag=self.module_tag,
            task_type=task_type,
        )

        # Update index with per-skill EMA scores
        for sid in skill_ids:
            self.task_index.update(task_type, sid, avg_score)
            skill = self.library.get(sid)
            if skill:
                skill.eval_scores.append(avg_score)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Delegate to inner adapter, then augment records with skill context info.

        The PRISM metadata is added to each record so the proposal LLM can see
        which skills were active during evaluation.
        """
        base_dataset = self.inner.make_reflective_dataset(
            candidate, eval_batch, components_to_update
        )

        # Augment each record with PRISM skill information
        augmented: dict[str, list[dict[str, Any]]] = {}
        for component, records in base_dataset.items():
            augmented_records = []
            for record in records:
                aug_record = dict(record)
                aug_record["prism_skills_used"] = list(self._last_skill_ids)
                aug_record["prism_skill_context"] = self._last_skill_context[:500]
                augmented_records.append(aug_record)
            augmented[component] = augmented_records

        return augmented

    @property
    def propose_new_texts(self) -> Any:
        """Forward propose_new_texts from inner adapter.

        The GEPA engine checks `adapter.propose_new_texts is not None` to decide
        whether to use the adapter's custom proposal logic. We forward whatever
        the inner adapter provides.
        """
        return getattr(self.inner, "propose_new_texts", None)
