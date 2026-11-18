# Pilot baseline: frozen Qwen2.5-Coder-1.5B + GEPA adapter on MBPP-sanitized

Baseline-этап из [`../pilot_selection.md`](../pilot_selection.md), раздел 3. Результаты описаны в [`baseline_results.md`](baseline_results.md).

Framework: **GEPA** (`gepa-ai/gepa@fb1ed58`, v0.1.4). Он установлен как pip-зависимость, его код не менялся.
Наш код — только адаптер (`GEPAAdapter`) для MBPP-sanitized и Ollama, а также evaluator и обвязка для запуска.

> Папка отделена от `../pilot/`, где параллельно работает другая сессия со своей версией harness. Эти две папки не связаны.

| Файл | Назначение |
|---|---|
| `data.py` | загружает MBPP-sanitized, фиксирует сплиты 30/20/50 (`seed=0`), пишет `data/*.jsonl` и `splits_manifest.json` с SHA-256 |
| `sandbox.py` | **evaluator**: запускает код и asserts в отдельном процессе `python -I` во временной директории, timeout 10 с |
| `harness_adapter.py` | `MBPPHarnessAdapter(GEPAAdapter)`: SOLVE → CHECK (видимый assert) → REPAIR ×1 → SCORE (все asserts). Реализует `evaluate` и `make_reflective_dataset`, пишет JSONL-лог шагов |
| `seed_harness.json` | baseline candidate: `solver_prompt`, `repair_prompt`, написаны вручную |
| `run_eval.py` | прогоняет кандидата на сплите и пишет результат в `runs/<run_id>/` |
| `run_optimize.py` | `gepa.optimize(...)` с нашим адаптером и reflection-моделью → `runs/<run_id>/best_candidate.json` |
| `compare_runs.py` | попарно сравнивает два прогона (воспроизводимость, B0 против baseline) |
| `tests/` | тесты sandbox, адаптера (mock LM, проверка, что скрытые asserts не утекают) и сплитов |
| `requirements.lock.txt` | точные версии окружения |

## Установка (Windows, PowerShell)

Нужны Python ≥ 3.10 (использовался 3.13.9) и Ollama. На машине нет `winget` и `uv`, поэтому окружение собрано через `venv` + `pip`, а Ollama поставлена из portable-архива.

```powershell
cd C:\Users\efr_v\Documents\ChatGPT\autoharness\pilot_baseline
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

Ollama v0.35.0: распакуйте `ollama-windows-amd64.zip` со страницы https://github.com/ollama/ollama/releases/tag/v0.35.0 в `%LOCALAPPDATA%\Programs\ollama` (или установите `OllamaSetup.exe`). Сервер запускается в отдельном окне:

```powershell
& "$env:LOCALAPPDATA\Programs\ollama\ollama.exe" serve
```

```powershell
& "$env:LOCALAPPDATA\Programs\ollama\ollama.exe" pull qwen2.5-coder:1.5b
```

Ожидаемый digest модели: `d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668`. Он записывается в `config.json` каждого прогона.

## Запуск

```powershell
.\.venv\Scripts\python.exe data.py
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

```powershell
.\.venv\Scripts\python.exe run_eval.py --split train --limit 5 --run-id smoke
```

```powershell
.\.venv\Scripts\python.exe run_eval.py --split test --run-id baseline
```

```powershell
.\.venv\Scripts\python.exe run_eval.py --split test --run-id baseline_rep
```

```powershell
.\.venv\Scripts\python.exe compare_runs.py runs/baseline runs/baseline_rep
```

```powershell
.\.venv\Scripts\python.exe run_eval.py --split test --no-repair --run-id direct
```

```powershell
.\.venv\Scripts\python.exe run_eval.py --split val --run-id baseline_val
```

`data.py` детерминированно пересоздаёт те же файлы: `manifest_sha256 = 16c91aba…`, а `load_split` падает при несовпадении SHA. `run_eval.py` не перезаписывает существующий `runs/<id>` без флага `--overwrite`.

## Оптимизация GEPA (локальный teacher)

```powershell
& "$env:LOCALAPPDATA\Programs\ollama\ollama.exe" pull qwen2.5-coder:7b
```

```powershell
.\.venv\Scripts\python.exe run_optimize.py --run-id gepa_s0 --seed 0 --max-metric-calls 400 --reflection-lm ollama_chat/qwen2.5-coder:7b
```

```powershell
.\.venv\Scripts\python.exe run_eval.py --candidate runs/gepa_s0/best_candidate.json --split test --run-id gepa_s0_test
```

```powershell
.\.venv\Scripts\python.exe compare_runs.py runs/baseline runs/gepa_s0_test
```

Повторный запуск `run_optimize.py` с тем же `--run-id` продолжает работу из `runs/<id>/gepa_state`. Параметры: `reflection_minibatch_size=3`, `pareto`, `round_robin`, `cache_evaluation=True`; target-декодинг такой же, как в baseline. Помимо файлов GEPA сохраняются `trajectories.jsonl` (все шаги target, phase=optimize), `reflection_calls.jsonl` (все вызовы teacher: prompt, ответ, токены, latency), `candidates.json`, `best_candidate.json`, `summary.json`.

## Что сохраняется в `runs/<run_id>/`

| Файл | Содержимое |
|---|---|
| `config.json` | конфигурация harness (модель, `temperature=0`, `max_tokens=512`, `num_ctx=4096`, `seed`, repair вкл/выкл, таймауты); candidate и его hash; digest модели и версия Ollama; версии gepa и litellm; SHA манифеста; список task id; команда запуска; платформа |
| `results.jsonl` | по строке на задачу: `task_id`, `answer` (финальный код), `verifier` (`ok`, `n_passed`, `n_total`, `error`, `timed_out`), `score`, `prompt_tokens`, `completion_tokens`, `latency_s`, `seed`, `harness_config`, `trajectory` (все LM-шаги) |
| `results.csv` | то же плоской таблицей, без trajectory |
| `trajectories.jsonl` | по строке на LM-шаг (схема из `pilot_selection.md` 3.4): `messages`, `raw_response`, `extracted_code`, `visible_test_ok`, `visible_error`, `final_pass`, токены, latency, хэши |
| `summary.json` | pass@1 и агрегаты |
| `run.log` | stdout прогона |

`latency_s` — суммарное время LM-вызовов на задачу, без sandbox. `wall_time_s` в `summary.json` — полное время прогона.

## Ограничения

- Sandbox на Windows: отдельный процесс, timeout и временная директория. Лимитов `resource` и сетевой изоляции нет. Для MBPP это приемлемо, но надёжнее запускать в WSL или Docker.
- Ollama работает на CPU (Intel Core 7 240H, без дискретного GPU). Greedy-декодинг на CPU воспроизводится не побайтово, подробности в `baseline_results.md`.
- `analyze.py` (парный bootstrap/McNemar) пока не написан; для парного сравнения на test используется `compare_runs.py`.
