"""AutoSaddler optimizer provider backed by a local Ollama model.

AutoSaddler ships Claude / Copilot / Codex transports only.  This transport is a
single-shot, tool-free session: it inlines the rendered workspace files into one
prompt, asks the local model for a JSON object and lets the stock BaseAgentProvider
validate it against the session schema.  Retries/timeouts stay in the engine.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from harness_lab import _paths  # noqa: F401  (adds AutoSaddler to sys.path)
from autosaddler.v2.prompting.models import Usage
from autosaddler.v2.providers.base import BaseAgentProvider, TransportOutcome, observe_usage
from autosaddler.v2.providers.workspace_renderer import RenderedSession, codex_renderer
from harness_lab.llm import LLMSettings, OllamaClient

_SKIP_PREFIXES = (".autosaddler", ".agents", "AGENTS.md")


@dataclass(frozen=True)
class OllamaProviderConfig:
    model: str = "qwen2.5:7b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.3
    seed: int = 0
    num_ctx: int = 8192
    max_tokens: int = 1024
    max_file_chars: int = 9000
    api_key_env: str = ""


class OllamaTransport:
    def __init__(self, config: OllamaProviderConfig) -> None:
        self.config = config

    def _prompt(self, session: RenderedSession) -> list[dict[str, str]]:
        parts = []
        for path in sorted(session.workspace.rglob("*")):
            rel = path.relative_to(session.workspace).as_posix()
            if not path.is_file() or rel.startswith(_SKIP_PREFIXES):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > self.config.max_file_chars:
                text = text[: self.config.max_file_chars] + "\n...[truncated]"
            parts.append(f"### FILE {rel}\n{text}")
        schema = session.output_schema_path.read_text(encoding="utf-8")
        system = (
            session.system_context.rstrip()
            + "\n\nYou have no tools. All workspace files are inlined below. Reply with ONE JSON "
            "object and nothing else. It must satisfy this JSON Schema:\n" + schema
        )
        user = session.task_prompt + "\n\n" + "\n\n".join(parts)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def run(self, session: RenderedSession, timeout_seconds: float) -> TransportOutcome:
        c = self.config
        client = OllamaClient(LLMSettings(
            model=c.model, base_url=c.base_url, temperature=c.temperature, seed=c.seed,
            num_ctx=c.num_ctx, max_tokens=c.max_tokens, timeout_s=timeout_seconds,
            api_key_env=c.api_key_env))
        reply = await asyncio.to_thread(client.chat, self._prompt(session), json_format="json")
        usage = Usage(
            input_tokens=reply["prompt_tokens"], output_tokens=reply["completion_tokens"],
            model=c.model, duration_seconds=reply["latency_s"],
            configured_settings={"model": c.model, "temperature": c.temperature, "num_ctx": c.num_ctx},
        )
        observe_usage(usage)
        try:
            json.loads(reply["text"])
        except json.JSONDecodeError:
            pass  # BaseAgentProvider reports "without the required structured output"; engine retries
        return TransportOutcome(raw_response=reply["text"], usage=(usage,), usage_streamed=True)


class OllamaAgentProvider(BaseAgentProvider):
    def __init__(self, config: OllamaProviderConfig) -> None:
        super().__init__(codex_renderer(), OllamaTransport(config))


def make_provider(*, ledger, settings):
    del ledger
    return OllamaAgentProvider(OllamaProviderConfig(**{k: v for k, v in dict(settings).items()}))
