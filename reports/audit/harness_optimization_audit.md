# Независимый аудит harness optimization

Вердикт: `reproducible_with_caveats`.

Пересчёт выполнен из `optimization_log.jsonl` и `evaluations.jsonl`, без использования сохранённого summary как источника истины.

## Ошибки

- Нет.

## Предупреждения

- logical search budget ('seed_harness', 0) is 10, expected 30
- logical search budget ('seed_harness', 1) is 10, expected 30
- logical search budget ('seed_harness', 2) is 10, expected 30
- model artifact hash unavailable from LM Studio

## Проверки

```json
{
  "unique_cache_keys": true,
  "reward_recomputed": true,
  "held_out_closed": true,
  "no_validation_updates": true,
  "model_ids": [
    "ministral-3-8b-instruct-2512"
  ],
  "logical_search_evaluations": 480,
  "logical_validation_evaluations": 180
}
```
