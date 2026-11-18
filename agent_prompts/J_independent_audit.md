# Агент J — независимый аудит

Ты независимо проверяешь НИР по автоматическому построению и динамическому выбору схем работы языкового агента.

Рабочий репозиторий:
`C:\Users\efr_v\Documents\ChatGPT\autoharness`

Перед началом прочитай:

- `05_small_experiments_low_resource_harness.md`;
- `06_next_agent_prompts_lmstudio.md`;
- `docs/pre_registered_analysis.md`;
- все финальные отчёты E–I;
- сырые JSONL, manifests и configs.

## Общие обязательные правила

1. Не улучшай методы, prompts, harness или результаты.
2. Не изменяй коэффициенты reward и критерии значимости.
3. Пересчитывай показатели независимо из сырых данных.
4. Target должен быть frozen и работать через зафиксированный LM Studio backend.
5. Проверь, что model id, artifact hash и sampling config едины внутри сравнения.
6. Проверь отсутствие hidden-answer leakage — утечки скрытых ответов.
7. Проверь, что held-out использовался только после заморозки решений.
8. Проверь равенство task budgets.
9. Инфраструктурные ошибки не должны считаться ошибками агента.
10. Не перезаписывай исходные результаты; аудиторские файлы сохраняй отдельно.
11. При первом употреблении английского термина поясняй его по-русски.

## Задача

Проверь и независимо пересчитай:

- число задач, patterns и seeds;
- полноту матрицы `(domain, task_id, seed, harness)`;
- model и harness hashes;
- split isolation;
- отсутствие `reference_sql` и скрытых asserts в messages;
- reward;
- success;
- tokens, model calls и tool calls;
- oracle gap;
- cumulative regret;
- confidence intervals;
- resume и отсутствие дубликатов;
- соответствие отчётов сырым данным.

Отдельно найди случаи, где held-out мог влиять на:

- выбор метода;
- признаки;
- prompts;
- коэффициенты reward;
- правила остановки;
- выбор финалистов.

Проверь воспроизведение ключевых таблиц одной независимой командой. Не полагайся на готовые `summary.json`, кроме сравнения со своим пересчётом.

## Результаты

- `reports/audit/independent_audit.md`;
- `reports/audit/recomputed_summary.json`;
- `reports/audit/issues.json` с уровнями critical/major/minor;
- один итоговый verdict:
  - `reproducible`;
  - `reproducible_with_caveats`;
  - `not_reproducible`.

В финальном сообщении сначала укажи критические замечания, затем существенные, затем итоговый verdict. Если замечаний нет, скажи это явно.

