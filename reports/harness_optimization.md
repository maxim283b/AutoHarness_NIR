# Малоресурсная оптимизация фиксированного harness

Статус: `complete_core`. Held-out — окончательно отложенная выборка — не открывался.

## Validation результаты

| метод | success | reward | task-bootstrap 95% CI | tokens/task | tail reward | worst failure |
|---|---:|---:|---:|---:|---:|---:|
| `literature_inspired_baseline` | 0.200 | 0.153 | [-0.050, 0.454] | 274.9 | -0.056 | 1.000 |
| `random_mutation_search` | 0.200 | 0.153 | [-0.050, 0.454] | 274.9 | -0.056 | 1.000 |
| `seed_harness` | 0.200 | 0.153 | [-0.050, 0.454] | 274.9 | -0.056 | 1.000 |
| `sequential_evolution_tail` | 0.200 | 0.152 | [-0.050, 0.453] | 281.7 | -0.056 | 1.000 |
| `bandit_mutation_operator` | 0.200 | 0.151 | [-0.051, 0.451] | 290.5 | -0.057 | 1.000 |
| `sequential_evolution_mean` | 0.200 | 0.139 | [-0.067, 0.444] | 353.6 | -0.076 | 1.000 |

## Бюджеты

- Логических search task-evaluations: 480.
- Логических validation task-evaluations: 180.
- Фактических target calls после cache: 453.
- Proposer calls: 0.
- Фактических токенов: 110605.

## Вывод

Автоматическая оптимизация не улучшила seed harness на validation: одинаковый лучший результат разделили `literature_inspired_baseline`, `random_mutation_search`, `seed_harness`. Поле `best_method=literature_inspired_baseline` — лишь детерминированный порядок внутри этой ничьей, а не доказательство превосходства.
Финалисты по preregistered правилу: `literature_inspired_baseline`, `random_mutation_search`.
Tail-оптимизация не улучшила нижний хвост относительно seed harness; усложнённый verifier loop увеличил расход токенов и вызовов, не повысив success.
Интервалы bootstrap пересэмплируют задачи; диапазоны по algorithm seeds в summary имеют другую интерпретацию и не являются CI переноса на новые задачи.

## Ограничения и отклонения

- Артефактный hash модели LM Studio недоступен; фиксирован только model id и параметры запроса.
- `seed_harness` не выполняет поиск кандидатов и получил 10 search task-evaluations на seed против 30 у оптимизаторов; это отмечено независимым аудитом как асимметрия бюджета.
- Core заранее ограничен тремя кандидатами на seed, а не рекомендованными 25 кандидатами; это зафиксировано до validation.
- Context 4096, режим 25 observations, перенос на Python, strong-model reference и held-out оставлены optional extension.
- `literature_inspired_baseline` не является точным AutoSaddler/AutoHarness reproduction.
- `core_20261009_v1` прерван во время разработки resumability и исключён; его append-only сырые логи не переписывались. Основной результат — отдельный `core_20261009_v2`.
- Первичный audit ошибочно трактовал оптимизируемый `generation_max_tokens` как смену модели. Исправлен только пост-hoc анализ с регрессионным тестом; raw JSONL `v2` не изменялся.

## Воспроизведение

```powershell
$env:TEMP=(Resolve-Path '.\.test_tmp').Path
$env:TMP=$env:TEMP
& '.\pilot\.venv\Scripts\python.exe' -m experiments.harness_optimization.cli audit --run-dir 'C:\\Users\\efr_v\\Documents\\ChatGPT\\autoharness\\results\\harness_optimization\\core_20261009_v2'
```
