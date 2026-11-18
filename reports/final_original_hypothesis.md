# Финальная проверка исходной гипотезы

Вердикт: `partially_tested_with_caveats`.

## Результат текущего этапа

Gate A1 имеет статус `official_infrastructure_reproduction_passed`. Официальный
AutoSaddler V2 на pinned commit выполнен через реальные Codex sessions; завершены
`evolve`, `diagnose_patch`, `reflect`, получен `RunCompleted`. Однако scenario был
`fake`: Ministral и SQL benchmark не запускались, patch
`{"instruction":"improved"}` не применялся к реальному `HarnessGenome`, а score 1.0
является только fixture result. Он не включён ни в одну научную таблицу.

Gate A2 имеет статус `real_target_benchmark_integration_passed`. Один официальный
candidate прошёл на пяти SQL search-задачах через frozen target
`ministral-3-8b-instruct-2512` в LM Studio и настоящий SQL verifier. Цепочка
`official session → patch → новый hash → target outputs → verifier → score` полностью
восстанавливается из raw events и evaluation rows.

AutoSaddler предложил содержательный patch к `planning_instruction`,
`verification_instruction` и `repair_rule`. Candidate hash изменился с `bcbc7b7d…`
на `482a423c…`, runtime config hash — с `f1c3afe2…` на `56c3e575…`. Child оценён
target-моделью. Официальный strict matched-valid gate отклонил patch: train mean
reward ухудшился с −0.09167 до −0.11743. Это корректный отрицательный smoke-результат,
а не integration failure. Seed development reward 0.9488 независимо пересчитан из
raw rows.

Использовано 3 official proposer calls (249318 input, 4506 output tokens) и 14 target
calls (3869 input, 628 output tokens). Infrastructure errors — 0. Validation и
held-out не открывались.

## Что ещё не проверено

Optimizer screening не запускался согласно прямому запрету до Gate A2. Поэтому пока
нет сравнения original/random/RL/tail-aware RL, нет подтверждения преимущества RL,
нет результата short-context/weak-model transfer. Реализованный
`tail_aware_rl_variant` не является воспроизведением официального Tail-Likelihood RL.

После Gate A2 создан новый пустой versioned scientific run `staged-scientific-v1`.
Он не переиспользует raw rows, score или candidate из fake scenario либо A2 smoke.

## Воспроизводимость

```powershell
$env:PYTHONPATH='experiments\final_harness_experiment\windows_compat;third_party\AutoSaddler\src;.'
$py = '.\pilot\.venv\Scripts\python.exe'
& $py -m pytest experiments\final_harness_experiment\tests -q
& $py -m experiments.final_harness_experiment.run_gate_a2 --config experiments\final_harness_experiment\autosaddler_sql_gate_a2.yaml --run-id gate-a2-sql-v1
& $py -m experiments.final_harness_experiment.gate_a2_audit results\final_harness_experiment\gate_a2_sql\gate-a2-sql-v1
```

Повторный запуск с тем же run id использует официальный resume; новый научный опыт
обязан иметь новый run id.
