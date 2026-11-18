# Агент E — сбор Python-матрицы

Ты работаешь над НИР по автоматическому построению и динамическому выбору схем работы языкового агента в условиях малой модели и ограниченного количества данных.

Рабочий репозиторий:
`C:\Users\efr_v\Documents\ChatGPT\autoharness`

Перед началом полностью прочитай:

- `05_small_experiments_low_resource_harness.md`;
- `06_next_agent_prompts_lmstudio.md`;
- `bandit_rl_pilot.md`;
- `docs/pre_registered_analysis.md`;
- `docs/trajectory_schema.md`;
- `pilot/README.md`;
- `pilot_common/`;
- `pilot/bandit/`.

Harness — внешняя схема работы агента: инструкции, последовательность шагов, правила вызова инструментов, проверка, исправление, память и критерий остановки.

## Общие обязательные правила

1. Основной backend — LM Studio через `http://127.0.0.1:1234/v1/chat/completions`.
2. Используй `PILOT_BACKEND=lmstudio` и `PILOT_MODEL=ministral-3-8b-instruct-2512`, если пользователь явно не указал другую модель.
3. Target-модель frozen — неизменяемая: не обучай и не изменяй её веса.
4. Перед запуском сохрани точный model id, параметры генерации и `PILOT_MODEL_ARTIFACT_HASH`, если hash доступен.
5. Не смешивай результаты разных моделей, model hash, backend или параметров генерации.
6. Никогда не передавай модели скрытые asserts, held-out labels или эталонные ответы.
7. Используй `pilot_common.PythonTaskAdapter`; не формируй prompt через `str(task)`.
8. Не изменяй verifier между patterns и методами.
9. Все методы получают одинаковые задачи и одинаковый task budget — число оценок на задачах.
10. Логируй task id, split, seed, model config, harness config/hash, выбранный pattern, полную траекторию, success, tokens, model calls, tool calls, latency и инфраструктурные ошибки.
11. Latency только логируется и не входит в reward.
12. Все случайные процессы имеют явный seed.
13. Не перезаписывай существующие результаты; поддерживай resume.
14. Инфраструктурная ошибка не считается ошибкой агента.
15. Не запускай массовый сбор до smoke test.
16. Не запускай одновременно другой массовый collector на том же LM Studio/GPU.
17. Held-out не открывай и не используй.
18. Не удаляй и не перезаписывай чужие файлы и результаты.
19. При первом употреблении английского термина поясняй его по-русски.
20. В конце запусти тесты и сохрани точные команды воспроизведения.

## Задача

Реализуй и проведи малый эксперимент полезных сигналов на MBPP-sanitized через LM Studio. Собери полную матрицу исходов четырёх harness patterns, не обучая policy — стратегию выбора.

Patterns:

- `direct`;
- `self_refine`;
- `verifier_loop`;
- `memory_assisted`.

Требования:

1. Создай collector поверх `pilot_common`, не дублируя общий runner.
2. Используй существующие split и verifier Python-пилота.
3. До модельных запусков зафиксируй список 40 non-heldout задач и его hash.
4. На каждой задаче выполни все четыре patterns при одной target-модели.
5. Основной запуск: `temperature=0`, seed `20261008`.
6. Для 20 заранее выбранных задач добавь seeds `20261009` и `20261010`. Если LM Studio остаётся детерминированным, отметь это и не называй идентичные повторы независимыми стохастическими измерениями.
7. Memory строится только из разрешённого search pool; текущая задача, validation, online и held-out в memory не попадают.
8. После начала основного сбора prompts и конфигурации не изменяются.
9. Результат должен читаться `analysis/signal_analysis.py` без ручной конвертации.

## Обязательный smoke test

Сначала выполни `2 задачи × 4 patterns` и автоматически проверь:

- `direct`: 1 model call, 1 final verifier;
- `self_refine`: 2 model calls, 1 final verifier;
- `verifier_loop`: 1–2 model calls и соответствующие verifier calls;
- `memory_assisted`: 1 model call, 1 retrieval, 1 final verifier;
- скрытые asserts отсутствуют во всех model messages;
- все записи имеют config hash и source hash;
- повторный запуск пропускает уже готовые пары.

Если smoke не прошёл, массовый сбор не запускай. Исправь причину, повтори smoke с новым run id и сохрани неудачную попытку как инфраструктурную.

## Результаты

Сохрани в новом каталоге `runs/python_signal_<run_id>/`:

- `config.json`;
- `task_manifest.json`;
- `outcomes.jsonl`;
- `trajectories.jsonl`;
- `integrity_report.json`;
- `smoke_report.md`;
- `README.md` с командами запуска и resume.

В финальном сообщении укажи число завершённых пар `(task, pattern, seed)`, пропуски, инфраструктурные ошибки, тесты и пути к результатам. Held-out оценку и обучение выбора не запускай.

