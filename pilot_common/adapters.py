"""Domain adapters keep private task fields out of model messages."""

from __future__ import annotations

import re
from typing import Any


_FENCE = re.compile(r"```(?:python|py|sql)?[ \t]*\n(.*?)```", re.I | re.S)


def _extract_fence(text: str) -> str:
    match = _FENCE.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


class PythonTaskAdapter:
    """Adapter for MBPP-style tasks; hidden asserts never enter messages."""

    def public_task(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": task.get("task_id"),
            "prompt": task["prompt"],
            "test_imports": list(task.get("test_imports", [])),
            "visible_test": (task.get("test_list") or [None])[0],
        }

    def initial_user(self, task, config, examples=None) -> str:
        parts = []
        if examples:
            parts.append(f"Examples from memory:\n{examples}")
        parts.append(task["prompt"])
        if config.use_visible_verifier and task.get("test_list"):
            parts.append(f"Your code should pass this visible test:\n{task['test_list'][0]}")
        return "\n\n".join(parts)

    def refine_user(self, task, previous: str) -> str:
        return f"Task:\n{task['prompt']}\n\nPrevious solution:\n{previous}\n\nReview it and return a corrected solution only."

    def repair_user(self, task, previous: str, error: str | None) -> str:
        return f"Task:\n{task['prompt']}\n\nPrevious solution:\n{previous}\n\nVerifier error:\n{error or 'failed'}\n\nReturn a corrected solution only."

    def extract_output(self, text: str) -> str:
        return _extract_fence(text)


class SQLTaskAdapter:
    """Adapter for pilot_sql; reference_sql is deliberately inaccessible."""

    PRIVATE_FIELDS = {"reference_sql"}

    def public_task(self, task: dict[str, Any]) -> dict[str, Any]:
        return {k: task[k] for k in ("task_id", "description", "schema", "allowed_operations", "features", "template_group") if k in task}

    def initial_user(self, task, config, examples=None) -> str:
        public = self.public_task(task)
        parts = []
        if examples:
            parts.append(f"Examples from memory:\n{examples}")
        parts += [
            f"Task:\n{public['description']}",
            f"SQLite schema:\n{public['schema']}",
            "Return exactly one read-only SQLite SELECT or WITH query. Do not modify data.",
        ]
        return "\n\n".join(parts)

    def refine_user(self, task, previous: str) -> str:
        return f"Task:\n{task['description']}\n\nPrevious SQL:\n{previous}\n\nReview it and return one corrected read-only SQLite query only."

    def repair_user(self, task, previous: str, error: str | None) -> str:
        return f"Task:\n{task['description']}\n\nPrevious SQL:\n{previous}\n\nVerifier error:\n{error or 'failed'}\n\nReturn one corrected read-only SQLite query only."

    def extract_output(self, text: str) -> str:
        return _extract_fence(text).rstrip(";").strip()

