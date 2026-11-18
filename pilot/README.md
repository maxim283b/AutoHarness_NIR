# Pilot: frozen small LLM + GEPA harness — baseline stage

Реализация раздела 3 из [`../pilot_selection.md`](../pilot_selection.md).
Framework — **GEPA** (`gepa-ai/gepa@fb1ed58`, v0.1.4), установлен как зависимость; его код не меняется.
Наш код — только адаптер (`GEPAAdapter`) для MBPP-sanitized и frozen target через локальный OpenAI-compatible endpoint LM Studio. Текущий target по умолчанию — `ministral-3-8b-instruct-2512`; его можно менять только между отдельными экспериментами, не внутри одного сравнения. Ollama сохранён как совместимый backend для старых запусков.

| Файл | Назначение |
|---|---|
| `data.py` | загрузка MBPP-sanitized, фиксированные сплиты 30/20/50 (`seed=0`), `splits_manifest.json` с SHA-256 |
| `sandbox.py` | **evaluator**: код + asserts в отдельном `python -I` процессе, temp cwd, timeout 10 s |
| `harness_adapter.py` | `MBPPHarnessAdapter(GEPAAdapter)`: SOLVE → CHECK (видимый assert) → REPAIR ×1 → SCORE (все asserts); `evaluate`, `make_reflective_dataset`, JSONL-лог шагов |
| `seed_harness.json` | baseline candidate: `solver_prompt`, `repair_prompt` (ручные) |
| `run_eval.py` | прогон кандидата на сплите → `runs/<run_id>/` |
| `compare_runs.py` | попарное сравнение двух прогонов (воспроизводимость, B0 vs baseline) |
| `tests/` | sandbox, адаптер (mock LM, отсутствие утечки скрытых asserts), сплиты |
| `requirements.lock.txt` | точные версии окружения (`pip freeze`) |

`run_optimize.py` / `analyze.py` из плана ещё не написаны — это следующий этап (GEPA-оптимизация), он требует reflection-модели.

## Установка (Windows, PowerShell)

Нужны Python ≥ 3.10 (здесь 3.13.9) и LM Studio с загруженной моделью. `winget`/`uv` на машине нет, поэтому используется существующий `venv` + `pip`.

LM Studio: запустить локальный сервер на `http://127.0.0.1:1234`, загрузить target-модель и узнать её точный identifier в LM Studio. Перед запуском задать:

```powershell
$env:PILOT_BACKEND = "lmstudio"
$env:LM_STUDIO_CHAT_URL = "http://127.0.0.1:1234/v1/chat/completions"
$env:PILOT_MODEL = "ministral-3-8b-instruct-2512"
$env:PILOT_MODEL_ARTIFACT_HASH = "<sha256-файла-модели-если-доступен>"
```

`run_eval.py` сохраняет backend, точный model id, параметры генерации и необязательный hash файла модели. Результаты разных model id или artifact hash нельзя смешивать в одном сравнении.

```powershell
cd C:\Users\efr_v\Documents\ChatGPT\autoharness\pilot
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

Старый Ollama backend можно включить явно:

```powershell
 $env:PILOT_BACKEND = "ollama"
```

```powershell
 # далее используется установленный ранее Ollama endpoint
```

Для исторических Ollama-прогонов использовался `qwen2.5-coder:1.5b` с digest `d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668`. Он не относится к текущему LM Studio target и не должен указываться как его hash.

## Запуск

```powershell
.\.venv\Scripts\python.exe data.py --out splits_manifest.json --seed 0
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

`data.py` пересоздаёт те же файлы (проверяется SHA в манифесте; `load_split` падает при расхождении). `run_eval.py` не перезаписывает существующий `runs/<id>` без `--overwrite`.

## Что сохраняется в `runs/<run_id>/`

| Файл | Содержимое |
|---|---|
| `config.json` | конфигурация harness (модель, `temperature=0`, `max_tokens=512`, `num_ctx=4096`, `seed`, repair on/off, timeout), сам candidate и его hash, digest модели и версия Ollama, версии gepa/litellm, SHA манифеста сплитов, список task id, команда запуска, платформа |
| `results.jsonl` | по строке на задачу: `task_id`, `answer` (финальный код), `verifier` (`ok`, `n_passed`, `n_total`, `error`, `timed_out`), `score`, `prompt_tokens`, `completion_tokens`, `latency_s`, `seed`, `harness_config`, `trajectory` (все LM-шаги) |
| `results.csv` | то же плоской таблицей, без trajectory |
| `trajectories.jsonl` | по строке на LM-шаг (схема из `pilot_selection.md` 3.4): `messages`, `raw_response`, `extracted_code`, `visible_test_ok`, `visible_error`, `final_pass`, токены, latency, hashes |
| `summary.json` | pass@1 и агрегаты |
| `run.log` | stdout прогона |

`latency_s` — суммарное время LM-вызовов на задачу (без sandbox); `wall_time_s` в `summary.json` — полное время прогона.

## Ограничения

- Sandbox на Windows: отдельный процесс + timeout + временная директория, без `resource`-лимитов и без сетевой изоляции. Для MBPP приемлемо; надёжнее — WSL/Docker.
- Ollama на CPU (Intel Core 7 240H, без дискретного GPU). Детерминизм проверяется повторным прогоном (`baseline_rep`).
