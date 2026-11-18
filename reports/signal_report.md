# Анализ полезных сигналов Python и SQL

**Основное решение: `CONTEXT_SIGNAL`.** Контекстная модель ожидаемой награды улучшает validation reward над заранее выбранной fixed-схемой на предзарегистрированный порог, а non-heldout oracle gap подтверждает гетерогенность; переходить к ограниченному contextual-bandit replay.

Анализ выполнен по заранее зарегистрированным коэффициентам: `reward = success - 0.10 × tokens/1000 - 0.02 × tool_calls`. Held-out — окончательная отложенная выборка — не открывался; решение принято по validation/gate — проверочной выборке для допуска.

## Аудит входов

- В анализ качества вошло 640 строк и 160 полных независимых единиц `(task_id, seed)`.
- Дубликаты resume: 0; неполные единицы: 0; held-out строк: 0.
- Первый Python-run исключён целиком: 328 инфраструктурных ошибок. Python retry и SQL содержат по 320 валидных строк; инфраструктурных ошибок в них нет.
- Backend, model id, temperature, max tokens и context совпадают: LM Studio, `ministral-3-8b-instruct-2512`, 0, 512 и 4096. Artifact hash модели не записан, поэтому идентичность весов нельзя доказать криптографически.
- Явных маркеров `reference_sql`, `HIDDEN_SECRET`, hidden tests/asserts и gold SQL в model messages не найдено. Это отрицательная проверка маркеров, а не формальное доказательство отсутствия любой семантической утечки.
- Task ID не пересекаются между split. У Python все задачи помечены одним `task_template=mbpp-sanitized`, который пересекает search и validation; поэтому внутридоменная GroupKFold — групповая перекрёстная проверка по шаблонам — для Python невозможна.

## Итоговые метрики

### PYTHON

| harness | success | reward | tokens | tool calls |
|---|---:|---:|---:|---:|
| `direct` | 0.688 | 0.653 | 140.3 | 1.00 |
| `memory_assisted` | 0.762 | 0.679 | 433.2 | 2.00 |
| `self_refine` | 0.375 | 0.323 | 322.8 | 1.00 |
| `verifier_loop` | 0.762 | 0.716 | 212.8 | 1.27 |

Best fixed — лучшая единая схема — выбрана на search+online: `verifier_loop`. На validation oracle gap — потенциальный выигрыш идеального выбора — 0.020 (95% bootstrap CI 0.005, 0.036).

Unique-best / practically-best — доли уникально / практически лучших схем (не дальше 0.05 reward):

| harness | unique | practical |
|---|---:|---:|
| `direct` | 0.400 | 1.000 |
| `memory_assisted` | 0.000 | 0.500 |
| `self_refine` | 0.000 | 0.500 |
| `verifier_loop` | 0.600 | 0.900 |

Модель ожидаемой награды пары task–harness на validation: reward 0.546 против 0.546 у best fixed, эффект 0.000 (95% CI 0.000, 0.000); success-эффект 0.000.

### SQL

| harness | success | reward | tokens | tool calls |
|---|---:|---:|---:|---:|
| `direct` | 0.400 | 0.356 | 240.7 | 1.00 |
| `memory_assisted` | 0.312 | 0.199 | 736.5 | 2.00 |
| `self_refine` | 0.412 | 0.353 | 393.5 | 1.00 |
| `verifier_loop` | 0.438 | 0.371 | 341.6 | 1.60 |

Best fixed — лучшая единая схема — выбрана на search+online: `verifier_loop`. На validation oracle gap — потенциальный выигрыш идеального выбора — 0.059 (95% bootstrap CI 0.012, 0.161).

Unique-best / practically-best — доли уникально / практически лучших схем (не дальше 0.05 reward):

| harness | unique | practical |
|---|---:|---:|
| `direct` | 0.520 | 0.960 |
| `memory_assisted` | 0.040 | 0.040 |
| `self_refine` | 0.000 | 0.960 |
| `verifier_loop` | 0.000 | 0.920 |

Модель ожидаемой награды пары task–harness на validation: reward 0.413 против 0.374 у best fixed, эффект 0.039 (95% CI -0.011, 0.143); success-эффект 0.040.

### Объединённые данные

| harness | success | reward | tokens | tool calls |
|---|---:|---:|---:|---:|
| `direct` | 0.544 | 0.505 | 190.5 | 1.00 |
| `memory_assisted` | 0.537 | 0.439 | 584.9 | 2.00 |
| `self_refine` | 0.394 | 0.338 | 358.1 | 1.00 |
| `verifier_loop` | 0.600 | 0.544 | 277.2 | 1.44 |

Best fixed — лучшая единая схема — выбрана на search+online: `verifier_loop`. На validation oracle gap — потенциальный выигрыш идеального выбора — 0.048 (95% bootstrap CI 0.013, 0.116).

Unique-best / practically-best — доли уникально / практически лучших схем (не дальше 0.05 reward):

| harness | unique | practical |
|---|---:|---:|
| `direct` | 0.486 | 0.971 |
| `memory_assisted` | 0.029 | 0.171 |
| `self_refine` | 0.000 | 0.829 |
| `verifier_loop` | 0.171 | 0.914 |

Модель ожидаемой награды пары task–harness на validation: reward 0.454 против 0.423 у best fixed, эффект 0.031 (95% CI -0.005, 0.101); success-эффект 0.029.

## Устойчивость по seeds

Повторы при `temperature=0` не интерпретируются как гарантированно независимая стохастика. Для каждой области ниже использованы только задачи, повторённые на всех трёх seeds.

| область | повторённых задач | согласие порядка | средняя корреляция reward между seeds | доля полностью идентичных векторов |
|---|---:|---:|---:|---:|
| python | 20 | 1.000 | 1.000 | 0.900 |
| sql | 20 | 1.000 | 1.000 | 0.967 |
| combined | 40 | 1.000 | 1.000 | 0.933 |

## Предсказуемость и зависимость от признаков

- **python:** GroupKFold недоступен — fewer than two task_template groups.
- **sql:** групп=16; logistic regression — логистическая регрессия — accuracy 0.875, selected reward 0.388; decision tree — дерево решений — accuracy 0.887, selected reward 0.356; pair ridge — модель ожидаемой награды — selected reward 0.343; честный fold-best-fixed 0.311; Brier 0.245, ECE 0.111.
- **combined:** групп=17; logistic regression — логистическая регрессия — accuracy 0.531, selected reward 0.509; decision tree — дерево решений — accuracy 0.512, selected reward 0.509; pair ridge — модель ожидаемой награды — selected reward 0.514; честный fold-best-fixed 0.544; Brier 0.913, ECE 0.455.

Accuracy лучшего класса приведена только диагностически. Основная проверка — reward-эффект на validation относительно best fixed; ни одна accuracy сама по себе не используется для решения.

## Перенос между доменами

Использованы только общие признаки `n_chars` и `n_words`. Это zero-shot — перенос без наблюдений целевого домена; предзарегистрированное сравнение адаптации на 5/10/25/50 наблюдениях здесь ещё не выполнено.

| направление | policy reward | target best-fixed reward | эффект | success-эффект |
|---|---:|---:|---:|---:|
| Python → SQL | 0.374 | 0.374 | 0.000 | 0.000 |
| SQL → Python | 0.566 | 0.546 | 0.020 | 0.000 |

## Чувствительность reward

| область | режим | best fixed (все non-heldout, описательно) | oracle gap |
|---|---|---|---:|
| python | main | `verifier_loop` | 0.119 |
| python | low_cost | `verifier_loop` | 0.116 |
| python | high_cost | `verifier_loop` | 0.126 |
| python | quality_first | `memory_assisted` | 0.113 |
| sql | main | `verifier_loop` | 0.092 |
| sql | low_cost | `verifier_loop` | 0.083 |
| sql | high_cost | `direct` | 0.101 |
| sql | quality_first | `verifier_loop` | 0.075 |
| combined | main | `verifier_loop` | 0.105 |
| combined | low_cost | `verifier_loop` | 0.099 |
| combined | high_cost | `verifier_loop` | 0.120 |
| combined | quality_first | `verifier_loop` | 0.094 |

## Решение

`CONTEXT_SIGNAL` — Контекстная модель ожидаемой награды улучшает validation reward над заранее выбранной fixed-схемой на предзарегистрированный порог, а non-heldout oracle gap подтверждает гетерогенность; переходить к ограниченному contextual-bandit replay.

Это разведочный, а не подтверждающий результат: на домен имеется 40 задач, но validation содержит 10 Python-задач и 15 SQL-задач (25 task-seed единиц SQL из-за повторов), а Python не имеет пригодных template groups. Следующий допустимый шаг — провести предзарегистрированное replay-сравнение contextual bandit и затем проверить не более двух финалистов на новом online-потоке; held-out остаётся закрытым.

## Воспроизведение

Из корня репозитория:

```powershell
python -m pytest analysis/tests -q
$env:PILOT_BACKEND='lmstudio'
$env:PILOT_MODEL='ministral-3-8b-instruct-2512'
python pilot/python_signal_collect.py --run-id 20261008_retry
python pilot/sql_signal_collect.py --run-id 20261008
python analysis/generate_signal_report.py --output-dir reports/reproduction_20261009
```

Машиночитаемые результаты: `reports/signal_summary.json`, `reports/tables.csv`, `reports/go_no_go.json`. Фигуры: `reports/figures/aggregate_reward.svg`, `reports/figures/validation_oracle_gap.svg`, `reports/figures/validation_policy_effect.svg`.
