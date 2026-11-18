# Краткий отчёт технической разведки

Дата: 2026-10-08.

Проверены материалы проекта `05_small_experiments_low_resource_harness.md`, `bandit_rl_pilot.md`, `pilot/README.md` и код `pilot/bandit/`. Запускались только read-only проверки окружения и тесты; полноценная оптимизация не запускалась. `git status` до изменений уже показывал набор неотслеживаемых пользовательских материалов; они не удалялись и не перезаписывались.

## Результат

Основной стенд: существующий GEPA adapter с LM Studio через OpenAI-compatible API. Контроль: prompt-only evaluation без optimizer. Сравнение: GEPA против DSPy MIPROv2; AHO — условный следующий кандидат после adapter smoke. Остальные системы отложить.

Причина: текущий GEPA имеет frozen target, общий verifier, train/val/test split manifest, JSONL trajectories, exact token/tool/latency logging, seeded optimizer call и resume state. AHO концептуально лучше покрывает middleware и tool-loop, но требует внешней proposer model и отдельного benchmark adapter. Для 5–50 наблюдений GEPA сохраняет меньшую дисперсию инфраструктурных изменений.

## Проверка версий

Локальная GEPA-зависимость зафиксирована в `pilot/requirements.lock.txt`: `gepa @ ...@fb1ed589fd83372caef499cffc2c73173d3b096b`; Python в README проекта — 3.13.9, Ollama — v0.35.0, target digest — `d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668`. Для внешних репозиториев GitHub в доступном web-срезе были видны только commit prefixes (`AHO 0d7652b`, AutoSaddler `30e20ce`, AFlow `3f45721`); полные SHA не выдумывались и должны быть получены clone/`rev-parse` перед установкой.

## Smoke

Повторная проверка подтвердила LM Studio endpoint и модель `ministral-3-8b-instruct-2512`. Live smoke выполнен на 3 train-задачах в новом каталоге `pilot/runs/lmstudio_smoke_20261008/`: **3/3 passed**, `pass_at_1=1.0`, repair rate `0`, 3 target calls, 440 total tokens, 47.58 s LM latency и 48.2 s wall time. Mock end-to-end, adapter smoke и тест формата OpenAI-compatible ответа также прошли: **24 passed in 41.24s** (`PILOT_BACKEND=lmstudio pilot/.venv/Scripts/python.exe -m pytest tests -q`). Held-out задачи не использовались.

## Контрольные инварианты перед первой оптимизацией

Нужно проверять: `target_model` и digest не меняются; verifier hash одинаков у методов; held-out IDs не попадают в optimizer feedback; total task-evaluations одинаков; every random source имеет seed; каждый output directory новый; interrupted run восстанавливается из checkpoint; latency остаётся только диагностикой.
