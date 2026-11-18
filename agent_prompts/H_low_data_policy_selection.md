# Агент H — малоданный выбор harness

Ты работаешь над НИР по автоматическому построению и динамическому выбору схем работы языкового агента в условиях малой модели и ограниченного количества данных.

Рабочий репозиторий:
`C:\Users\efr_v\Documents\ChatGPT\autoharness`

Перед началом полностью прочитай:

- `05_small_experiments_low_resource_harness.md`;
- `06_next_agent_prompts_lmstudio.md`;
- `docs/pre_registered_analysis.md`;
- `reports/signal_report.md`;
- `reports/go_no_go.json`;
- сырые результаты E и F;
- `pilot/bandit/`.

## Общие обязательные правила

1. Начинай работу только если `go_no_go.json` разрешает динамический выбор. При `NO_SIGNAL` остановись с отчётом и не запускай RL.
2. Основной backend реальной проверки — LM Studio; target frozen.
3. Не смешивай разные модели и конфигурации.
4. Held-out labels никогда не обновляют policy — стратегию выбора.
5. Все методы внутри seed получают один поток задач и одинаковый task budget.
6. Контекст содержит только признаки, доступные до решения задачи.
7. Latency не входит в reward.
8. Логируй состояние policy после каждого update.
9. Все seeds явные и воспроизводимые.
10. Сначала replay — проигрывание по собранной матрице; реальные LM Studio вызовы только для двух финалистов.
11. Не перезаписывай результаты и поддерживай resume.
12. Не запускай одновременно другой массовый collector на LM Studio.
13. При первом употреблении английского термина поясняй его по-русски.

## Задача

Сравни при бюджетах `0, 5, 10, 25, 50` наблюдений:

- random;
- best fixed с честным round-robin бюджетом;
- правило `direct -> verifier` при неуспехе;
- epsilon-greedy;
- UCB1;
- Thompson sampling;
- LinUCB;
- linear Thompson sampling;
- supervised reward model только как диагностическую верхнюю базовую линию.

Этап 1:

- replay на полной nonheldout матрице;
- минимум 100 фиксированных перестановок;
- отдельные и объединённые домены;
- перенос Python → SQL и SQL → Python;
- cumulative regret, held-out-free validation reward, success и стоимость.

Этап 2:

- выбрать не более двух финалистов по validation;
- заморозить их конфигурации;
- проверить на новом online потоке LM Studio, где наблюдается только выбранный harness;
- не использовать контрфактические результаты для update.

## Результаты

- `reports/policy_comparison.md`;
- `results/policy/online_log.jsonl.gz`;
- `results/policy/per_seed.jsonl`;
- `results/policy/summary.json`;
- `results/policy/frozen_finalists/`;
- команды воспроизведения и integrity report.

