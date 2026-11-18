# Post-hoc amendment to the harness optimization audit

Date: 2026-10-09. Run: `core_20261009_v2`.

This amendment changes no experimental observation, candidate, finalist, or raw
event. It corrects one false-positive check in the report generator after the
core run had finished.

## Correction

The first audit compared every field in the model request except `seed`. That
incorrectly classified variation in `generation_max_tokens` (256 versus 384) as
an inconsistent target model. `generation_max_tokens` is explicitly a frozen,
preregistered harness gene. The corrected audit compares the invariant target
backend fields: model id, endpoint, temperature, timeout, and artifact hash.
Algorithm seed, generation budget, and harness context budget remain in the
evaluation cache identity but are not interpreted as a model switch.

A regression test was added to prevent this false positive. The corrected
verdict is `reproducible_with_caveats`, with no audit errors.

## Unchanged caveats

- LM Studio does not expose an artifact hash for the loaded model.
- `seed_harness` performs no candidate search and therefore consumed 10 search
  task-evaluations per algorithm seed, versus 30 for each optimizer. The
  preregistered budget text said 30 per method, so the audit retains this as an
  explicit asymmetry.
- The held-out split remains closed.

## Integrity anchors

- Executed `analysis.py` hash recorded in the run manifest:
  `302f448a2545b000e5a25709f625982c20b781ca0031b9cedcf0c3b9038e9d77`.
- Corrected post-hoc `analysis.py` SHA-256:
  `899ec1008fd370f6e53ea1f842d84cdb231de553a2efbc2d3d52216e0111d920`.
- `evaluations.jsonl` SHA-256:
  `ed45e0706c2a725106ccaf275719eda0e868e11c015521e17e7658cff019a5b5`.
- `optimization_log.jsonl` SHA-256:
  `e55a18f48f6a432bc64a3554d607faaa9897213c25e62d7fdd5029086e109ba3`.

The last two hashes are the immutable raw-data anchors for independent
recomputation.
