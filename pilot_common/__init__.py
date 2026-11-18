"""Shared, backend-agnostic harness execution and trajectory schemas."""

from .schema import HarnessConfig, ModelConfig, Trajectory, Step, ToolResult
from .clients import OllamaClient, OpenAICompatibleClient
from .journal import AppendOnlyJournal, completed_pairs
from .runner import run_task
from .adapters import PythonTaskAdapter, SQLTaskAdapter

__all__ = [
    "HarnessConfig", "ModelConfig", "Trajectory", "Step", "ToolResult",
    "OllamaClient", "OpenAICompatibleClient", "AppendOnlyJournal",
    "completed_pairs", "run_task", "PythonTaskAdapter", "SQLTaskAdapter",
]
