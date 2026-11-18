# Агент I — автоматическое создание harness

Ты работаешь над НИР по автоматическому построению и динамическому выбору схем работы языкового агента в условиях малой модели и ограниченного количества данных.

Рабочий репозиторий:
`C:\Users\efr_v\Documents\ChatGPT\autoharness`

Перед началом полностью прочитай:

- `05_small_experiments_low_resource_harness.md`;
- `06_next_agent_prompts_lmstudio.md`;
- `docs/framework_scout.md`;
- `docs/framework_matrix.csv`;
- `docs/pre_registered_analysis.md`;
- результаты E, F и G;
- существующий GEPA adapter.

## Общие обязательные правила

1. Target запускается через LM Studio и остаётся frozen.
2. Target model и proposer/reflection model — модель, предлагающая изменения, — фиксируются и логируются отдельно.
3. Не смешивай разные модели, backend или sampling config.
4. Не передавай оптимизатору hidden tests, `reference_sql` или held-out labels.
5. Verifier, safety rules и скрытые ответы изменять запрещено.
6. Все методы получают одинаковое число task-evaluations.
7. Proposer calls не считаются target evaluations, но их токены, вызовы и стоимость логируются отдельно.
8. Held-out открывается только после заморозки кандидатов и метода выбора.
9. Latency не входит в reward.
10. Все seeds явные.
11. Не перезаписывай результаты; поддерживай resume.
12. До оптимизации проведи smoke на 3–5 задачах.
13. Не запускай одновременно другой массовый процесс на LM Studio.
14. При первом употреблении английского термина поясняй его по-русски.

## Задача

Сравни автоматическое создание harness при бюджетах `25, 50, 100` task-evaluations.

Методы:

1. random mutation — случайные допустимые изменения;
2. BetterHarness, только если отдельный Agent Harness Optimizer adapter smoke прошёл;
3. GEPA;
4. MIPROv2;
5. гибрид: GEPA создаёт 3–5 кандидатов, затем контекстная policy выбирает кандидата по текущей задаче.

Разрешённые изменения:

- `system_prompt`;
- `pattern`;
- `max_repairs`;
- `memory_k`;
- `use_visible_verifier`;
- `stop_rule`.

На первом этапе запрещено изменять код инструментов, SQL authorizer, Python sandbox и структуру скрытых tests.

Для каждого метода сохрани:

- seed candidate;
- всех кандидатов и родительские связи;
- текст/конфигурацию изменения;
- задачи, на которых кандидат оценивался;
- target и proposer budgets;
- причину принятия или отклонения;
- validation score;
- Pareto frontier — набор вариантов без одновременного проигрыша по качеству и стоимости.

Сравни:

- лучшую найденную fixed схему;
- динамический выбор между кандидатами;
- перенос на второй домен;
- validation improvement и однократное held-out improvement после заморозки.

## Результаты

- `reports/harness_optimization.md`;
- `results/harness_optimization/candidates/`;
- `results/harness_optimization/optimization_log.jsonl`;
- `results/harness_optimization/final_configs/`;
- `results/harness_optimization/summary.json`;
- команды воспроизведения и integrity report.

