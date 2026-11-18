# Gate A: официальный baseline и реальная интеграция

Дата проверки: 12 октября 2026 года.

Основной baseline — официальный **Microsoft AutoSaddler V2**, MIT, commit
`e841543ce6cf6bdf690e65134b12a829ec9f66d1`. Proposer — официальный
`CodexAgentProvider`; frozen target — `ministral-3-8b-instruct-2512` через LM Studio.

## Gate A1 — `official_infrastructure_reproduction_passed`

Run `codex-official-smoke-v5` действительно исполнил официальный AutoSaddler V2 с
реальными Codex sessions. Завершены `evolve`, `diagnose_patch`, `reflect` и получено
событие `RunCompleted`. Полный неизменённый run сохранён отдельно, для каждого из 66
файлов записан SHA-256.

Но это только infrastructure fixture: scenario был `fake`, scorer не вызывал
Ministral, SQL benchmark не использовался, а patch `{"instruction":"improved"}` не
применялся к `HarnessGenome`. Его score 1.0 не является качеством агента, исключён из
научных таблиц и не является научным финалистом.

## Gate A2 — `real_target_benchmark_integration_passed`

Run `gate-a2-sql-v1` использовал официальный неизменённый цикл и реальный adapter:

- scenario `sql_harness`, не fake/mock;
- пять task IDs из фиксированного SQL search manifest: `sql-01-00`, `sql-01-01`,
  `sql-01-02`, `sql-02-00`, `sql-02-01`;
- реальные LM Studio calls к frozen `ministral-3-8b-instruct-2512`;
- существующие project runner и SQL verifier;
- официальный content-addressed `ComponentMapHarnessSpace` и официальный patch application;
- proposer получал только безопасный агрегированный verifier feedback; поле
  `reference_sql` отсутствует во всех трёх proposer requests;
- parent candidate `sha256:bcbc7b7d63c174a95933f34f0dada2b1a2ea25ec3c76dc4d5528d36a8ace56f2`;
- child candidate `sha256:482a423cb4c76819bd1e7d0793ab747d35ef670e12f0596f74167a0c56a9afaf`;
- derived HarnessGenome config hash изменился с `f1c3afe2f39c9bf825bda5263ed9c0935fcd01932b074aa68e4d625190698547`
  на `56c3e575d6fe63290ca5dfbd56b11ed4672e00c7e85a4fd0f7db774822a547b8`;
- child был фактически передан LM Studio в train-after evaluation;
- patch не прошёл официальный strict gate: mean reward ухудшился с `-0.0916667` до
  `-0.1174333`; выбран seed harness;
- independently recomputed development reward seed harness равен `0.9488`;
- 3 proposer calls: 249318 input и 4506 output tokens;
- 14 target calls: 3869 input и 628 output tokens;
- infrastructure errors: 0.

Тонкий adapter только преобразует публичные шесть текстовых компонентов в реальный
сериализуемый `HarnessGenome`, вызывает существующий runner/verifier и строит safe
evidence. Он не генерирует patch и не заменяет `evolve → diagnose_patch → reflect`.

Screening и validation до прохождения Gate A2 не запускались. Held-out закрыт.
