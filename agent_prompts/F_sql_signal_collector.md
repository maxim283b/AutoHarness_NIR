# Агент F — сбор SQL-матрицы

Ты работаешь над НИР по автоматическому построению и динамическому выбору схем работы языкового агента в условиях малой модели и ограниченного количества данных.

Рабочий репозиторий:
`C:\Users\efr_v\Documents\ChatGPT\autoharness`

Перед началом полностью прочитай:

- `05_small_experiments_low_resource_harness.md`;
- `06_next_agent_prompts_lmstudio.md`;
- `bandit_rl_pilot.md`;
- `docs/pre_registered_analysis.md`;
- `docs/trajectory_schema.md`;
- `pilot_common/`;
- `pilot_sql/README.md`;
- `pilot_sql/splits_manifest.json`.

Harness — внешняя схема работы агента: инструкции, последовательность шагов, правила вызова инструментов, проверка, исправление, память и критерий остановки.

## Общие обязательные правила

1. Основной backend — LM Studio через `http://127.0.0.1:1234/v1/chat/completions`.
2. Используй `PILOT_BACKEND=lmstudio` и `PILOT_MODEL=ministral-3-8b-instruct-2512`, если пользователь явно не указал другую модель.
3. Target-модель frozen: веса не обучать и не изменять.
4. Зафиксируй model id, параметры генерации и artifact hash, если он доступен.
5. Не смешивай результаты разных моделей, backend или конфигураций.
6. Никогда не передавай модели `reference_sql`, held-out labels или эталонный результат.
7. Используй `pilot_common.SQLTaskAdapter` и `pilot_sql.verifier.verify`.
8. Модели передаются только публичные поля задачи.
9. Verifier одинаков для всех patterns.
10. Все patterns получают одинаковые задачи и task budget.
11. Логируй полную конфигурацию, траекторию, success, tokens, model/tool calls, latency и тип ошибки.
12. Latency не входит в reward.
13. Все случайные процессы имеют seed.
14. Не перезаписывай результаты; поддерживай resume.
15. Инфраструктурные ошибки отделяй от ошибок агента.
16. Не запускай массовый сбор до smoke test.
17. Не запускай одновременно другой массовый collector на том же LM Studio/GPU.
18. Held-out не открывай.
19. Не удаляй и не перезаписывай чужие материалы.
20. При первом употреблении английского термина поясняй его по-русски.

## Задача

Собери полную матрицу результатов четырёх patterns на 40 SQL-задачах из `first_experiment_ids`.

Patterns:

- `direct`;
- `self_refine`;
- `verifier_loop`;
- `memory_assisted`.

Требования:

1. Используй только идентификаторы из `first_experiment_ids` и проверь отсутствие пересечения с held-out.
2. Передавай модели только `description`, `schema`, `allowed_operations`, безопасные features и template group.
3. Автоматически сканируй messages и публичные trajectory inputs на строки `reference_sql`, `HIDDEN_SECRET` и точный эталонный запрос.
4. Основной запуск: `temperature=0`, seed `20261008`.
5. Для 20 заранее выбранных задач добавь seeds `20261009` и `20261010` с оговоркой о детерминированности.
6. Memory содержит только разрешённые search-примеры и никогда не использует held-out, validation, online или текущую задачу.
7. Отдельно логируй `sql_error`, `wrong_result`, `timeout`, `unsafe_query`, `multiple_statements`, `empty_query` и инфраструктурные ошибки.
8. Resume key: `(task_id, pattern, seed, config_hash)`.
9. После начала основного сбора prompts, verifier и конфигурацию не менять.
10. Формат `outcomes.jsonl` должен читаться `analysis/signal_analysis.py`.

## Обязательный smoke test

Выполни `2 задачи × 4 patterns` и проверь:

- точное число model/tool calls;
- `memory_assisted` делает один model call;
- ни одно сообщение не содержит `reference_sql` или эталонный SQL;
- SQL извлекается из fenced и plain-text ответов;
- unsafe и multiple statements блокируются;
- повторный запуск не создаёт дубликатов.

Если smoke не прошёл, массовый сбор не запускай.

## Результаты

Сохрани в `runs/sql_signal_<run_id>/`:

- `config.json`;
- `task_manifest.json`;
- `outcomes.jsonl`;
- `trajectories.jsonl`;
- `integrity_report.json`;
- `smoke_report.md`;
- `README.md` с командами запуска и resume.

В финальном сообщении укажи число завершённых пар, пропуски, распределение типов ошибок, тесты и пути. Held-out и обучение policy не запускай.

