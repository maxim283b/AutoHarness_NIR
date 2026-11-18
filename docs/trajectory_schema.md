# Единая схема harness и траектории

`pilot_common` — изолированный общий слой для Python-задач и SQL-задач. Он не меняет
верификатор и не изменяет веса target-модели. Harness — внешняя схема работы агента:
инструкции, порядок шагов, вызовы инструментов, исправления, память и остановка.

## Конфигурация

`HarnessConfig` сериализуется через `to_dict()` и хешируется каноническим JSON.
Обязательные поля:

- `system_prompt` — системная инструкция;
- `pattern` — `direct`, `self_refine`, `verifier_loop` или `memory_assisted`;
- `max_repairs`, `memory_k`, `use_visible_verifier`, `stop_rule`, `tool_policy`;
- `model` — backend (`ollama` или `openai_compatible`), model, URL и параметры генерации;
- `seed` — явное зерно случайности;
- `harness_source_hash` — hash исходника harness, если он фиксируется вызывающей программой.

## Запуск

```python
from pilot_common import HarnessConfig, ModelConfig, OpenAICompatibleClient, SQLTaskAdapter, run_task

cfg = HarnessConfig(
    system_prompt="Return one correct SQL query.", pattern="direct",
    model=ModelConfig(backend="openai_compatible", model="ministral-3-8b-instruct-2512",
                      base_url="http://127.0.0.1:1234"), seed=0,
)
client = OpenAICompatibleClient(cfg.model)
trajectory = run_task(task, cfg, client, verifier=lambda answer: verifier(task, answer),
                      adapter=SQLTaskAdapter())
```

Основной backend — LM Studio: используется `OpenAICompatibleClient` и `ModelConfig(backend="openai_compatible", base_url="http://127.0.0.1:1234")`.
Вызов клиента через HTTP является единственным изменением backend; verifier и harness остаются теми же.

## Траектория

Одна запись журнала — одна завершённая пара `(task_id, pattern)`. `steps` сохраняет в порядке
все шаги модели, инструментов и verifier-а; у model-step сохраняются сообщения, вход, полный
выход, токены и latency. У tool/verifier-step сохраняются вход, результат, ошибка и длительность.
На верхнем уровне есть `success`, `error_type`, `stop_reason`, hashes конфигурации и исходника,
суммарные токены, число model/tool calls и время.

Latency логируется, но намеренно не входит в reward: существующий pilot показывает нестабильность
локального измерения. Это должно быть доказано отдельным тестом до включения в reward.

## Append-only и resume

`AppendOnlyJournal.append_trajectory()` только добавляет JSONL-строку и не редактирует старые
строки. `completed_pairs(journal, config_hash=...)` возвращает уже рассчитанные пары; вызывающий
сборщик должен пропускать их. При изменении конфигурации старые записи сохраняются и считаются
отдельным запуском. Тесты используют mock-клиент и не запускают массовый сбор данных.
## Доменные адаптеры и приватные поля

Общий runner не формирует prompt из произвольного словаря задачи. Для Python
используется `PythonTaskAdapter`, для SQL — `SQLTaskAdapter`. Адаптер определяет
публичное представление задачи, сообщения solve/refine/repair и извлечение
ответа. Поля `reference_sql` и скрытые asserts не входят в model messages или
`task_input` траектории. `memory_assisted` выполняет один модельный вызов с
извлечёнными примерами; предварительный direct-вызов не выполняется.
