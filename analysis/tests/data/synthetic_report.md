# Signal analysis report (synthetic or supplied JSONL)

Input rows: 97; tasks: 24

## Main reward

| harness | success | reward | tokens | tools |
|---|---:|---:|---:|---:|
| direct | 0.500 | 0.490 | 100.0 | 0.00 |
| verifier_loop | 0.500 | 0.460 | 200.0 | 1.00 |

Mean oracle gap: **0.485** (95% bootstrap CI 0.287, 0.683).

## Sensitivity and stability

* `low_cost`: best fixed `direct`, oracle gap `0.492`
* `high_cost`: best fixed `direct`, oracle gap `0.465`
* `quality_first`: best fixed `verifier_loop`, oracle gap `0.500`

Seeds: 2; order agreement: 1.000.

## Predictability

```json
{
  "status": "unavailable",
  "reason": "scikit-learn is not installed"
}
```

Held-out rows, if present, were scored after fitting and were not used for model selection.
