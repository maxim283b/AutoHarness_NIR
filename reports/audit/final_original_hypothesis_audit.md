# Независимый аудит финальной проверки

Вердикт: `partially_tested_with_caveats`.

## Gate A1

- official AutoSaddler commit: `e841543ce6cf6bdf690e65134b12a829ec9f66d1`;
- run: `codex-official-smoke-v5`;
- реальные official Codex sessions: 3;
- `RunCompleted`: да;
- классификация: `official_infrastructure_reproduction_passed`;
- scenario: `fake`;
- LM Studio target calls: 0;
- SQL tasks: 0;
- score 1.0: исключён как infrastructure fixture.

Архив содержит 66 исходных файлов; каждый copied file совпал с source SHA-256.

## Gate A2

Independent audit прочитал 51 official events и raw target rows, не summary:

- scenario `sql_harness`;
- 5 IDs принадлежат search manifest;
- target client `RecordingLMStudioClient` является тонкой регистрирующей оболочкой
  реального project `LMStudioClient`, не mock;
- model и endpoint совпадают с протоколом;
- parent/child candidate и runtime config hashes различаются;
- raw child rows доказывают использование patched harness;
- raw model outputs и verifier events сохранены;
- каждая строка reward повторно вычисляется формулой
  `success - 0.10*tokens/1000 - 0.02*tool_calls`;
- development reward 0.9488 независимо пересчитан;
- в proposer request отсутствует `reference_sql`;
- sessions: ровно `evolve`, `diagnose_patch`, `reflect`;
- `RunCompleted`: да;
- proposer и target usage разделены;
- infrastructure errors: 0 и не переклассифицированы в task failures.

Integration tests намеренно падают для fake scenario, mock target, неизменившегося
hash, неиспользованного child, непересчитываемого score и leakage. Результат тестов:
12 passed.

Validation, screening и held-out не запускались. A2 smoke не переносится в новый
scientific run как observation, score или finalist.
