# Разведка методов автоматического построения harness

Дата среза: 2026-10-08, часовой пояс Europe/Moscow. Под harness понимается внешняя схема работы агента: инструкция, loop, middleware, инструменты, память, проверка и остановка. Target-модель и verifier в этой разведке считаются фиксированными.

## Итог

Рекомендация: **основная платформа — существующий GEPA adapter в `pilot/` с LM Studio как OpenAI-compatible backend**; **лёгкая контрольная платформа — прямой prompt-only loop из `pilot/run_eval.py`**; методы сравнения — **GEPA и DSPy MIPROv2**. AHO следует подключить как следующий кандидат после маленького adapter smoke: его API хорошо совпадает с benchmark, но он требует outer/proposer model и фактически расширяет mutable surface до middleware. Критерий «подключить AHO за 1–2 дня» пока не подтверждён: сетевой доступ к GitHub из текущего окружения не позволил получить checkout и lock-файл, а live LM Studio endpoint сейчас недоступен.

Не запускалась полноценная оптимизация. Held-out задачи не использовались для настройки.

## Источники, назначение и лицензии

- [Airbnb Agent Harness Optimizer](https://github.com/airbnb/agent-harness-optimizer) — benchmark-agnostic framework; README описывает две роли модели, редактирование `system_prompt` и middleware, `Benchmark` subclass и resume. Репозиторий MIT; публичный initial release наблюдался на commit prefix `0d7652b`.
- [GEPA](https://github.com/gepa-ai/gepa) и [статья](https://arxiv.org/abs/2507.19457) — reflective text optimization с Pareto-отбором и полными трассами; MIT, зафиксирован в этом проекте на `fb1ed589fd83372caef499cffc2c73173d3b096b`.
- [DSPy MIPROv2](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md) — Bayesian search по инструкциям и few-shot примерам; это prompt optimizer, не полноценный workflow optimizer.
- [AFlow](https://github.com/FoundationAgents/AFlow) и [ICLR paper](https://arxiv.org/abs/2410.10762) — MCTS по code-represented workflow graph. Название иногда ошибочно связывают с AutoFlow; это другой проект.
- [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) — framework/orchestration layer с несколькими оптимизаторами, а не один сопоставимый алгоритм.
- [Microsoft AutoSaddler](https://github.com/microsoft/AutoSaddler) и [paper](https://arxiv.org/abs/2608.23041) — отдельный durable trace-driven harness optimizer; MIT, initial release commit prefix `30e20ce`.
- [Stanford IRIS Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) и [paper](https://arxiv.org/abs/2603.28052) — filesystem-mediated search по исходникам harness и трассам; MIT. Это официальный проект, не одноимённые независимые `ManagementMO/Meta-Harness` или `SuperagenticAI/metaharness`.

Короткие prefixes приведены честно как наблюдаемые GitHub references. Для строгого воспроизведения перед установкой нужно разрешить их до полных SHA командой `git ls-remote <url> <ref>` в окружении с сетью и записать результат вместе с lock-файлом. Не следует подменять короткий prefix вымышленным полным SHA.

## Что реально меняют методы

AHO — самый близкий к определению harness: prompt плюс middleware вокруг tool calls; PRISM/BH/MIPROv2/GEPA доступны через единый CLI. Но AHO предполагает отдельную outer model, а его штатные benchmarks — BFCL/tau, поэтому нужен custom `Benchmark` adapter для MBPP и явное ограничение именно числа task-evaluations.

GEPA меняет текстовые параметры через adapter. В текущем проекте это `solver_prompt` и `repair_prompt`; adapter уже передаёт teacher видимый результат проверки, сохраняет шаги, token counts и не показывает скрытые asserts. Это наиболее безопасное соответствие правилам НИР.

MIPROv2 совместно ищет инструкции и демонстрации Bayesian optimization. Он годится как prompt-only comparison при заранее заданном малом `num_trials`, но число реальных вызовов надо считать в собственном ledger, а не только по параметру trials.

AFlow ищет код графа workflow методом Monte Carlo Tree Search: меняется топология, узлы, ребра, промпты и форматы. Это научно интересный поздний метод, но его validation score и генерация графа делают бюджет 5–50 наблюдений трудным.

EvoAgentX — удобная оболочка для интеграции, но из-за нескольких встроенных стратегий и изменяемого workflow не является чистым сравнением без фиксации конкретного backend.

AutoSaddler меняет кандидатный harness через scenario/plugin boundary и сохраняет durable события, но его текущий V2 требует `uv`, Python 3.12–3.14 и собственный plugin. Локальные Ollama/LM Studio в README не являются штатным путём.

Meta-Harness эволюционирует произвольный код harness через proposer-agent, который читает filesystem с предыдущими кандидатами, score и traces. Это сильный ориентир для позднего end-to-end эксперимента, но дорогой outer loop и reference wrapper на Claude Code плохо соответствуют 5–50 наблюдениям и локальной малой модели.

## Выбор и go/no-go

| Система | Решение | Причина |
|---|---|---|
| GEPA | GO | Уже есть адаптер, локальная модель, JSONL traces, `max_metric_calls`, run state и тесты. |
| DSPy MIPROv2 | GO как сравнение | Prompt-only, есть явные knobs бюджета; нужен отдельный ledger. |
| Airbnb AHO | CONDITIONAL GO | Adapter реалистичен за 1–2 дня, если подтвердятся local OpenAI-compatible backend и полноценные hashes; сначала offline/mock smoke, затем 3–5 задач. |
| AFlow | NO-GO сейчас | Слишком большой graph-search surface и неочевидный task-eval budget. |
| EvoAgentX | NO-GO сейчас | Оболочка, а не однозначный метод; добавляет интеграционный шум. |
| AutoSaddler | NO-GO сейчас | Durable инфраструктура полезна позднее, но не минимальная связка и не локальный backend из коробки. |
| Meta-Harness | NO-GO сейчас | Сильный end-to-end ориентир, но дорогой proposer и не строгий маленький бюджет без доработки. |

## Интеграция AHO с текущим benchmark

Минимальный путь:

1. Сделать checkout AHO в отдельном каталоге, не смешивая его с `pilot/.venv`.
2. Реализовать `MBPPBenchmark(Benchmark)`: `score_async` вызывает существующий `run_task`/sandbox, возвращает `SplitScore` с per-case pass/fail, tokens и trajectory metadata; `build_asi` формирует только feedback по visible assert.
3. Запретить AHO менять verifier, sandbox, target model и tool implementations. Для первого прогона использовать `--bh-prompt-only` или `--prism-prompt-only`; middleware сравнивать отдельным экспериментом.
4. Ввести `--max-steps` равный числу task-evaluations и отдельный hard counter в adapter. `wall-time` и latency не включать в reward.
5. Сохранить immutable split manifest, seed, model endpoint, candidate diff, exact input, complete trajectory, token/tool counts и resume checkpoint в новом `runs/aho_<timestamp>/`.
6. Выполнить mock smoke, затем 3–5 задач; не использовать held-out и не перезаписывать `pilot/runs`.

Ожидаемая проблема: AHO outer model должна читать failure files и писать патчи; малая target-модель не обязана быть пригодной для этой роли. Поэтому reflection/proposer лучше вынести на отдельную локальную или удалённую модель, явно зафиксировав её в config и не включая её вызовы в task-evaluation budget target-модели.

## Воспроизводимые команды

```powershell
cd C:\Users\efr_v\Documents\ChatGPT\autoharness\pilot
& .\.venv\Scripts\python.exe -m pytest tests -q
& .\.venv\Scripts\python.exe -m pytest tests/test_gepa_e2e.py tests/test_adapter.py -q
& .\.venv\Scripts\python.exe run_eval.py --split train --limit 5 --run-id scout_smoke_20261008
```

Последняя команда требует запущенного LM Studio Server на `http://127.0.0.1:1234` и уже загруженной модели. После разрешения доступа endpoint был проверен, модель `ministral-3-8b-instruct-2512` успешно прошла smoke на 3 train-задачах: 3/3 verifier pass, без repair. Результат сохранён в новом каталоге `pilot/runs/lmstudio_smoke_20261008/`. Mock GEPA smoke запускается первой командой и не расходует модельные вызовы. Backend выбирается через `PILOT_BACKEND=lmstudio`, модель — через `PILOT_MODEL`.

Для будущего AHO после разрешения сети:

```powershell
git clone https://github.com/airbnb/agent-harness-optimizer.git external/agent-harness-optimizer
git -C external/agent-harness-optimizer rev-parse HEAD
git -C external/agent-harness-optimizer log -1 --format=%H
uv pip install -e external/agent-harness-optimizer
python -m agent_harness_optimizer.cli --help
```

Запуск оптимизации намеренно не приводится: сначала нужен smoke с собственным benchmark и hard task-evaluation counter.

## Протокол smoke и ограничения

Тестовый smoke должен иметь 3–5 train задач, seed=0, одинаковый frozen target/verifier и отдельный output directory. Успех smoke означает только end-to-end совместимость, не улучшение качества. Любая оценка на held-out запрещена до выбора кандидата. Latency логируется, но не входит в reward. Существующие результаты не изменялись.
