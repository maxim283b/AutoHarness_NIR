## Run `provisional`: reward = success − 0.1·tokens/1000 − 0.02·tool_calls − 0.0·latency_s; 50 seeds; online pool 100, held-out 41

### Patterns (full information, each pattern on every task)

| pattern | online reward | online success | held-out reward | held-out success | tokens/task | LM calls/task | tool calls/task |
|---|---|---|---|---|---|---|---|
| direct | 0.664 | 0.68 | 0.520 | 0.54 | 163 | 1.00 | 0.00 |
| plan_execute | 0.515 | 0.59 | 0.393 | 0.46 | 745 | 2.00 | 0.00 |
| self_refine | 0.636 | 0.68 | 0.492 | 0.54 | 442 | 2.00 | 0.00 |
| verifier_loop | 0.666 | 0.73 | 0.504 | 0.59 | 348 | 1.44 | 1.44 |
| memory | 0.552 | 0.61 | 0.479 | 0.54 | 377 | 1.00 | 1.00 |

Best fixed pattern on the online pool (regret comparator): **verifier_loop**.

### Online cumulative regret and held-out quality (mean ± 95% CI over seeds)

| T | policy | regret vs best fixed | regret vs per-task oracle | online success | held-out reward | held-out success | held-out tokens | P(deploy best) | deployed |
|---|---|---|---|---|---|---|---|---|---|
| 10 | random | 0.61 ± 0.05 | 1.72 ± 0.25 | 0.632 | 0.476 ± 0.009 | 0.530 | 431 | 0.00 | random:50 |
| 10 | best_fixed | 0.60 ± 0.00 | 1.39 ± 0.26 | 0.664 | 0.490 ± 0.012 | 0.536 | 371 | 0.22 | direct:21, verifier_loop:11, self_refine:8, plan_execute:8, memory:2 |
| 10 | eps_greedy | 0.59 ± 0.13 | 1.70 ± 0.30 | 0.630 | 0.486 ± 0.011 | 0.535 | 387 | 0.18 | direct:15, memory:10, self_refine:9, verifier_loop:9, plan_execute:7 |
| 10 | ucb1 | 0.53 ± 0.03 | 1.51 ± 0.29 | 0.652 | 0.498 ± 0.010 | 0.541 | 337 | 0.24 | direct:24, verifier_loop:12, self_refine:6, plan_execute:5, memory:3 |
| 10 | thompson | 0.52 ± 0.06 | 1.52 ± 0.23 | 0.650 | 0.484 ± 0.013 | 0.537 | 425 | 0.22 | self_refine:13, direct:12, verifier_loop:11, plan_execute:9, memory:5 |
| 10 | ts_per_type | 0.55 ± 0.06 | 1.77 ± 0.28 | 0.628 | 0.479 ± 0.012 | 0.533 | 426 | 0.24 | self_refine:13, verifier_loop:12, direct:11, plan_execute:9, memory:5 |
| 25 | random | 1.53 ± 0.09 | 3.97 ± 0.40 | 0.651 | 0.474 ± 0.012 | 0.526 | 419 | 0.00 | random:50 |
| 25 | best_fixed | 1.49 ± 0.00 | 3.80 ± 0.42 | 0.658 | 0.494 ± 0.009 | 0.538 | 357 | 0.14 | direct:18, self_refine:14, memory:7, verifier_loop:7, plan_execute:4 |
| 25 | eps_greedy | 1.34 ± 0.32 | 3.94 ± 0.49 | 0.650 | 0.489 ± 0.011 | 0.540 | 393 | 0.24 | direct:14, verifier_loop:12, self_refine:11, memory:7, plan_execute:6 |
| 25 | ucb1 | 1.36 ± 0.05 | 3.69 ± 0.44 | 0.661 | 0.491 ± 0.011 | 0.540 | 385 | 0.24 | direct:16, verifier_loop:12, self_refine:11, plan_execute:6, memory:5 |
| 25 | thompson | 1.24 ± 0.13 | 3.95 ± 0.52 | 0.650 | 0.499 ± 0.007 | 0.549 | 373 | 0.32 | verifier_loop:16, direct:15, self_refine:12, memory:5, plan_execute:2 |
| 25 | ts_per_type | 1.37 ± 0.10 | 4.00 ± 0.42 | 0.648 | 0.487 ± 0.010 | 0.539 | 409 | 0.24 | self_refine:14, verifier_loop:12, direct:12, plan_execute:7, memory:5 |
| 50 | random | 2.98 ± 0.11 | 8.03 ± 0.61 | 0.652 | 0.475 ± 0.011 | 0.528 | 429 | 0.00 | random:50 |
| 50 | best_fixed | 2.98 ± 0.00 | 7.93 ± 0.60 | 0.655 | 0.495 ± 0.009 | 0.543 | 383 | 0.26 | direct:15, self_refine:15, verifier_loop:13, plan_execute:4, memory:3 |
| 50 | eps_greedy | 1.89 ± 0.46 | 7.05 ± 0.62 | 0.668 | 0.496 ± 0.008 | 0.549 | 388 | 0.34 | verifier_loop:17, direct:13, self_refine:11, memory:6, plan_execute:3 |
| 50 | ucb1 | 2.70 ± 0.11 | 7.33 ± 0.52 | 0.666 | 0.494 ± 0.009 | 0.546 | 384 | 0.32 | verifier_loop:16, direct:14, memory:8, self_refine:8, plan_execute:4 |
| 50 | thompson | 2.35 ± 0.26 | 7.59 ± 0.65 | 0.660 | 0.494 ± 0.009 | 0.550 | 399 | 0.40 | verifier_loop:20, direct:12, self_refine:8, memory:6, plan_execute:4 |
| 50 | ts_per_type | 2.59 ± 0.16 | 7.63 ± 0.48 | 0.660 | 0.498 ± 0.009 | 0.545 | 366 | 0.24 | direct:19, verifier_loop:12, self_refine:8, memory:6, plan_execute:5 |

References on held-out: best fixed (full info) = verifier_loop 0.504; per-type oracle 0.502 ({'math': 'direct', 'other': 'plan_execute', 'sequence': 'verifier_loop', 'string': 'direct'}); per-task oracle 0.635 (success 0.66); random expected 0.478.

### Mean reward by task type × pattern (full information)

| pool/type | n | direct | plan_execute | self_refine | verifier_loop | memory | best |
|---|---|---|---|---|---|---|---|
| online/math | 27 | 0.465 (0.48) | 0.323 (0.41) | 0.436 (0.48) | 0.442 (0.52) | 0.314 (0.37) | direct |
| online/other | 7 | 0.553 (0.57) | 0.637 (0.71) | 0.523 (0.57) | 0.470 (0.57) | 0.368 (0.43) | plan_execute |
| online/sequence | 50 | 0.783 (0.80) | 0.666 (0.74) | 0.755 (0.80) | 0.827 (0.88) | 0.721 (0.78) | verifier_loop |
| online/string | 16 | 0.673 (0.69) | 0.316 (0.38) | 0.648 (0.69) | 0.631 (0.69) | 0.509 (0.56) | direct |
| heldout/math | 7 | 0.269 (0.29) | 0.359 (0.43) | 0.240 (0.29) | 0.314 (0.43) | 0.224 (0.29) | plan_execute |
| heldout/sequence | 28 | 0.519 (0.54) | 0.358 (0.43) | 0.491 (0.54) | 0.492 (0.57) | 0.478 (0.54) | direct |
| heldout/string | 6 | 0.819 (0.83) | 0.595 (0.67) | 0.792 (0.83) | 0.780 (0.83) | 0.781 (0.83) | direct |

### Online selection share by task type, T=50

| policy | type | direct | plan_execute | self_refine | verifier_loop | memory |
|---|---|---|---|---|---|---|
| random | math | 0.22 | 0.19 | 0.20 | 0.20 | 0.19 |
| random | other | 0.24 | 0.21 | 0.20 | 0.15 | 0.20 |
| random | sequence | 0.21 | 0.19 | 0.19 | 0.19 | 0.22 |
| random | string | 0.20 | 0.20 | 0.20 | 0.17 | 0.22 |
| best_fixed | math | 0.20 | 0.21 | 0.19 | 0.19 | 0.21 |
| best_fixed | other | 0.27 | 0.15 | 0.21 | 0.15 | 0.21 |
| best_fixed | sequence | 0.20 | 0.21 | 0.21 | 0.19 | 0.20 |
| best_fixed | string | 0.19 | 0.18 | 0.19 | 0.25 | 0.19 |
| eps_greedy | math | 0.24 | 0.10 | 0.26 | 0.26 | 0.14 |
| eps_greedy | other | 0.24 | 0.08 | 0.25 | 0.26 | 0.17 |
| eps_greedy | sequence | 0.24 | 0.08 | 0.29 | 0.25 | 0.14 |
| eps_greedy | string | 0.23 | 0.08 | 0.34 | 0.24 | 0.10 |
| ucb1 | math | 0.21 | 0.19 | 0.20 | 0.21 | 0.20 |
| ucb1 | other | 0.30 | 0.15 | 0.16 | 0.18 | 0.21 |
| ucb1 | sequence | 0.21 | 0.16 | 0.20 | 0.26 | 0.18 |
| ucb1 | string | 0.21 | 0.19 | 0.21 | 0.19 | 0.20 |
| thompson | math | 0.24 | 0.13 | 0.22 | 0.24 | 0.17 |
| thompson | other | 0.21 | 0.14 | 0.21 | 0.26 | 0.18 |
| thompson | sequence | 0.22 | 0.15 | 0.21 | 0.26 | 0.15 |
| thompson | string | 0.25 | 0.16 | 0.23 | 0.22 | 0.13 |
| ts_per_type | math | 0.23 | 0.17 | 0.22 | 0.20 | 0.18 |
| ts_per_type | other | 0.21 | 0.25 | 0.17 | 0.17 | 0.19 |
| ts_per_type | sequence | 0.26 | 0.16 | 0.20 | 0.21 | 0.16 |
| ts_per_type | string | 0.20 | 0.16 | 0.18 | 0.28 | 0.16 |

Latency reliability: 90 identical-request pairs (direct vs verifier_loop passing first try): median relative |Δlatency| = 0.37, p90 = 0.99; corr(latency, tokens) = 0.70.
