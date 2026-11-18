# Реализация общего harness-слоя

Дата: 8 октября 2026 г.

Добавлен изолированный каталог `pilot_common/` без изменения `pilot/bandit` и
без запуска сбора данных. `legacy.py` делегирует старые patterns напрямую в
`pilot/bandit/patterns.py`; поэтому существующая семантика и cache не меняются.

Реализовано:

- `HarnessConfig` и `ModelConfig` на stdlib dataclass, JSON Schema и hash канонической конфигурации;
- единый `run_task(task, harness_config, model_client, verifier, memory)`;
- полная `Trajectory` с model/tool/verifier шагами, входами/выходами, токенами, calls, latency, ошибкой и остановкой;
- source hash для legacy harness и конфигурационный hash;
- append-only JSONL через `AppendOnlyJournal`, resume через `completed_pairs`;
- Ollama и OpenAI-compatible endpoint (включая LM Studio);
- unit tests с mock-клиентом и пример полной траектории;
- latency только логируется и не используется в reward.

Проверка:

- `python -m compileall -q pilot_common` — успешно;
- локальный smoke test `pilot_common` с mock model/verifier — успешно;
- `python -m pytest pilot_common/tests -q` — после исправления двойной verifier-проверки остаётся 3 passed и 1 ошибка окружения pytest из-за `WinError 5` при разрешении системного temp/cwd.
- полный `pilot/tests` не запускал массовых задач; pytest в том же окружении блокируется тем же `WinError 5` на `pilot/tests/conftest.py`.

Сетевые клиенты и Ollama не вызывались. Результаты существующего pilot не перезаписывались.
