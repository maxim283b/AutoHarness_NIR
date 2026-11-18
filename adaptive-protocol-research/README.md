# Динамический выбор протоколов LLM-агентов: воспроизводимый пилот

Статус: архитектура проверена на scripted mock и в ограниченном **реальном** пилоте через LM Studio (`ministral-3-8b-instruct-2512`, 2 train / 2 test, 28 task-runs). Все методы решили две простые test-задачи без переключений; это не подтверждает и не опровергает гипотезу о полезности динамики. Дополнительная диагностика P1 на 12 train-задачах дала 12/12 без видимых ошибок. Исследовательский дизайн — в [research/experiment_design.md](research/experiment_design.md), аналоги — в [research/related_work.md](research/related_work.md), итоги — в [research/results.md](research/results.md) и [research/conclusions.md](research/conclusions.md).

## Что реализовано

- P1: исполнитель; P2: планировщик → исполнитель; P3: исполнитель → reviewer; P4: планировщик → исполнитель → reviewer. Каждая роль использует один и тот же `OPENAI_MODEL` через OpenAI-compatible chat API.
- Общая доступная инструментальная проверка: запуск публичного assert в отдельном процессе после каждого исполнения. Скрытые assert доступны только финальному evaluator. До трёх логических этапов и двух переключений.
- Fixed, rule, LLM-controller со строгой Pydantic-валидацией, LinUCB с train-историей, холодный LinUCB, random и rule с выбором только в начале.
- JSONL traces, ограничения вызовов/токенов/времени, отчёт с Wilson interval и парными расхождениями. При заданных тарифах записывается USD cost.

## Данные

В `data/` уже лежат 12 train и 20 test задач MBPP-sanitized из существующего [локального пилота](../pilot/README.md). Их ID и SHA-256 зафиксированы в [manifest](data/manifest.json). При наличии соседнего `pilot/data` файлы можно пересоздать через `python scripts/prepare_data.py`. Первоисточник: [Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp), [dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp).

## Запуск (Python 3.11+)

```powershell
python -m pip install -e ".[test,plot]"
python -m pytest -q
python scripts/run_experiment.py --mode mock --limit 2 --out results/my_mock_smoke
python scripts/analyze.py results/my_mock_smoke
```

Реальный прогон требует `OPENAI_API_KEY`, `OPENAI_BASE_URL` (например, адрес с `/v1`) и `OPENAI_MODEL`. Можно использовать любой совместимый endpoint. Команда печатает верхнюю оценку бюджета **до запуска**. Для реальной модели дополнительно нужен явный флаг `--real-budget-approved`; это защита от случайного платного запуска. Для малого пробного запуска после оценки цены:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "http://127.0.0.1:1234/v1"
$env:OPENAI_MODEL = "<model-id>"
python scripts/run_experiment.py --mode real --real-budget-approved --limit 1 --out results/real_smoke_01
python scripts/analyze.py results/real_smoke_01
```

Полный план (12 train, 20 test, 248 task-runs) консервативно ограничен сверху 2 976 вызовами и 2,48 млн токенов. Настройки по умолчанию блокируют такой реальный запуск; после пробного измерения укажите обоснованные `--max-total-calls`, `--max-total-tokens` и `--max-total-seconds`. Для денежной оценки задайте `OPENAI_INPUT_USD_PER_MTOK` и `OPENAI_OUTPUT_USD_PER_MTOK`.

Длительные запуски сохраняют строку после каждой задачи; `--resume` продолжает прерванный run без повторения уже выполненных строк при том же split, модели и лимитах на задачу. Глобальные лимиты при возобновлении можно только повысить. Локальная диагностика трудности train: `python scripts/diagnose_train.py --out results/train_p1_diagnostic.jsonl --reuse results/real_paired_2/traces.jsonl` (запускать с теми же переменными LM Studio).

## Ограничения

Mock служит только проверкой кода и не даёт evidence о полезности динамического выбора. Локальный subprocess с AST-фильтром и timeout не является защищённой песочницей. MBPP ограничивается короткими задачами Python; при досрочном прохождении публичного assert переключение не происходит. Обучение LinUCB использует награду после задачи и не решает оптимальное последовательное управление. При 12 train задачах выводы об обучении будут особенно ненадёжны.
