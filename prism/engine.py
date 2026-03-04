from __future__ import annotations

import logging
from typing import Any, Callable

from prism.assembler.assembler import SkillAssembler
from prism.lifecycle.curator import SkillCurator
from prism.lifecycle.reflector import PRISMReflector
from prism.skill_library.library import SkillLibrary
from prism.skill_library.skill import Skill
from prism.task_index.index import TaskTypeIndex

logger = logging.getLogger(__name__)

# Type alias: (prompt, task) -> (score, trace, output)
EvaluateFn = Callable[[str, dict[str, Any]], tuple[float, str, str]]


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
    ):
        self.evaluate_fn = evaluate_fn
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn
        self.base_prompt = base_prompt
        self.top_k = top_k
        self.token_budget = token_budget
        self.maintenance_interval = maintenance_interval
        self.enable_differential_eval = enable_differential_eval

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
        self.curator = SkillCurator(library=self.library, llm_fn=llm_fn, embed_fn=embed_fn)

        self._step_count = 0
        self._history: list[dict[str, Any]] = []

    def step(self, task: dict[str, Any], module_tag: str = "general") -> dict[str, Any]:
        """Execute a single iteration of the 6-step PRISM loop."""
        self._step_count += 1
        result: dict[str, Any] = {"step": self._step_count, "task": task}

        # Step 1: RETRIEVE & ASSEMBLE
        skill_context, skill_ids = self.assembler.assemble(task, module_tag=module_tag)
        if skill_context:
            augmented_prompt = f"{self.base_prompt}\n\n## Relevant Skills\n{skill_context}"
        else:
            augmented_prompt = self.base_prompt
        result["skills_injected"] = len(skill_ids)
        result["skill_ids"] = skill_ids
        logger.info("[Step %d] RETRIEVE: %d skills assembled", self._step_count, len(skill_ids))

        # Step 2: EXECUTE
        try:
            score, trace, output = self.evaluate_fn(augmented_prompt, task)
        except Exception as e:
            logger.warning("[Step %d] EXECUTE failed: %s", self._step_count, e)
            score, trace, output = 0.0, str(e), ""
        result["score"] = score
        result["output"] = output
        logger.info("[Step %d] EXECUTE: score=%.3f", self._step_count, score)

        # Step 3: REFLECT
        reflection = self.reflector.reflect(
            task=task,
            score=score,
            trace=trace,
            output=output,
            skill_ids=skill_ids,
        )
        result["attributions"] = reflection.attributions
        result["gaps"] = reflection.gaps
        result["diagnosis"] = reflection.diagnosis
        logger.info(
            "[Step %d] REFLECT: %d attributions, %d gaps",
            self._step_count,
            len(reflection.attributions),
            len(reflection.gaps),
        )

        # Step 4: CURATE
        task_type = self.task_index.classify_task(task)
        ops = self.curator.curate(
            reflection=reflection,
            module_tag=module_tag,
            task_type=task_type,
        )
        result["operations"] = ops
        logger.info("[Step %d] CURATE: %s", self._step_count, ops or "none")

        # Step 5: DIFFERENTIAL EVALUATION + score_matrix update
        task_key = Skill.task_key(task.get("question", ""))

        if self.enable_differential_eval and skill_ids:
            bare_score = self._evaluate_without_skills(task)
            result["bare_score"] = bare_score
            result["differential"] = score - bare_score
            effective_score = score - bare_score
            logger.info(
                "[Step %d] DIFFERENTIAL: with_skills=%.3f, without=%.3f, delta=%.3f",
                self._step_count, score, bare_score, score - bare_score,
            )

            # Update eval_scores and score_matrix on used skills
            for sid in skill_ids:
                skill = self.library.get(sid)
                if skill:
                    skill.eval_scores.append(score)
                    skill.score_matrix[task_key] = effective_score
        else:
            # Still record scores even without differential
            for sid in skill_ids:
                skill = self.library.get(sid)
                if skill:
                    skill.eval_scores.append(score)
                    skill.score_matrix[task_key] = score

        # Step 6: INDEX UPDATE (EMA)
        for sid in skill_ids:
            self.task_index.update(task_type, sid, score)
        logger.info("[Step %d] INDEX: updated %d entries for type=%s", self._step_count, len(skill_ids), task_type)

        self._history.append(result)
        return result

    def _evaluate_without_skills(self, task: dict[str, Any]) -> float:
        try:
            score, _, _ = self.evaluate_fn(self.base_prompt, task)
            return score
        except Exception as e:
            logger.warning("Bare evaluation failed: %s", e)
            return 0.0

    def train(
        self,
        trainset: list[dict[str, Any]],
        num_epochs: int = 1,
        eval_every: int = 5,
        valset: list[dict[str, Any]] | None = None,
        module_tag: str = "general",
    ) -> dict[str, Any]:
        """Full training loop with periodic maintenance."""
        results: list[dict[str, Any]] = []
        val_scores: list[float] = []

        for epoch in range(num_epochs):
            logger.info("=== Epoch %d/%d ===", epoch + 1, num_epochs)
            epoch_scores: list[float] = []

            for i, task in enumerate(trainset):
                result = self.step(task, module_tag=module_tag)
                results.append(result)
                epoch_scores.append(result["score"])

                # Periodic maintenance
                if self._step_count % self.maintenance_interval == 0:
                    self._maintenance()

            avg_train = sum(epoch_scores) / len(epoch_scores) if epoch_scores else 0.0
            logger.info("Epoch %d train avg: %.3f", epoch + 1, avg_train)

            # Validation
            if valset and (epoch + 1) % eval_every == 0:
                val_score = self._validate(valset, module_tag)
                val_scores.append(val_score)
                logger.info("Epoch %d val avg: %.3f", epoch + 1, val_score)

        # Final save
        self.library.save()
        self.task_index.save()

        return {
            "total_steps": self._step_count,
            "results": results,
            "val_scores": val_scores,
            "library_summary": self.library.summary(),
        }

    def _validate(self, valset: list[dict[str, Any]], module_tag: str) -> float:
        scores = []
        for task in valset:
            skill_context, _ = self.assembler.assemble(task, module_tag=module_tag)
            if skill_context:
                prompt = f"{self.base_prompt}\n\n## Relevant Skills\n{skill_context}"
            else:
                prompt = self.base_prompt
            try:
                score, _, _ = self.evaluate_fn(prompt, task)
                scores.append(score)
            except Exception as e:
                logger.warning("Validation error: %s", e)
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def _maintenance(self) -> None:
        """Periodic maintenance: save state."""
        logger.info("[Maintenance] Saving library and index...")
        self.library.save()
        self.task_index.save()
