## Run `final`: reward = success − 0.1·tokens/1000 − 0.02·tool_calls − 0.0·latency_s; 100 seeds; online pool 100, held-out 50

### Patterns (full information, each pattern on every task)

| pattern | online reward | online success | held-out reward | held-out success | tokens/task | LM calls/task | tool calls/task |
|---|---|---|---|---|---|---|---|
| direct | 0.664 | 0.68 | 0.565 | 0.58 | 163 | 1.00 | 0.00 |
| plan_execute | 0.515 | 0.59 | 0.451 | 0.52 | 745 | 2.00 | 0.00 |
| self_refine | 0.636 | 0.68 | 0.538 | 0.58 | 442 | 2.00 | 0.00 |
| verifier_loop | 0.666 | 0.73 | 0.566 | 0.64 | 348 | 1.44 | 1.44 |
| memory | 0.552 | 0.61 | 0.544 | 0.60 | 377 | 1.00 | 1.00 |

Best fixed pattern on the online pool (regret comparator): **verifier_loop**.

### Online cumulative regret and held-out quality (mean ± 95% CI over seeds)

| T | policy | regret vs best fixed | regret vs per-task oracle | online success | held-out reward | held-out success | held-out tokens | P(deploy best) | deployed |
|---|---|---|---|---|---|---|---|---|---|
| 10 | random | 0.62 ± 0.04 | 1.66 ± 0.19 | 0.640 | 0.534 ± 0.006 | 0.585 | 406 | 0.00 | random:100 |
| 10 | best_fixed | 0.60 ± 0.00 | 1.31 ± 0.18 | 0.672 | 0.550 ± 0.006 | 0.593 | 321 | 0.28 | direct:45, verifier_loop:28, self_refine:12, plan_execute:9, memory:6 |
| 10 | eps_greedy | 0.55 ± 0.09 | 1.64 ± 0.21 | 0.639 | 0.540 ± 0.007 | 0.588 | 377 | 0.21 | direct:26, memory:21, verifier_loop:21, self_refine:18, plan_execute:14 |
| 10 | ucb1 | 0.55 ± 0.02 | 1.52 ± 0.19 | 0.652 | 0.546 ± 0.007 | 0.588 | 336 | 0.23 | direct:42, verifier_loop:23, self_refine:14, plan_execute:12, memory:9 |
| 10 | thompson | 0.54 ± 0.04 | 1.40 ± 0.18 | 0.663 | 0.539 ± 0.008 | 0.590 | 399 | 0.27 | self_refine:27, verifier_loop:27, direct:23, plan_execute:17, memory:6 |
| 10 | ts_per_type | 0.59 ± 0.04 | 1.71 ± 0.20 | 0.634 | 0.538 ± 0.007 | 0.588 | 394 | 0.26 | verifier_loop:26, self_refine:24, direct:23, plan_execute:16, memory:11 |
| 25 | random | 1.54 ± 0.06 | 4.01 ± 0.31 | 0.650 | 0.531 ± 0.007 | 0.582 | 403 | 0.00 | random:100 |
| 25 | best_fixed | 1.49 ± 0.00 | 3.72 ± 0.30 | 0.660 | 0.549 ± 0.006 | 0.591 | 330 | 0.23 | direct:40, verifier_loop:23, self_refine:19, memory:10, plan_execute:8 |
| 25 | eps_greedy | 1.29 ± 0.22 | 3.85 ± 0.34 | 0.653 | 0.544 ± 0.007 | 0.594 | 370 | 0.29 | verifier_loop:29, direct:27, self_refine:19, memory:14, plan_execute:11 |
| 25 | ucb1 | 1.37 ± 0.04 | 3.55 ± 0.31 | 0.666 | 0.548 ± 0.006 | 0.595 | 356 | 0.29 | direct:29, verifier_loop:29, self_refine:24, memory:11, plan_execute:7 |
| 25 | thompson | 1.27 ± 0.09 | 3.94 ± 0.34 | 0.651 | 0.547 ± 0.006 | 0.595 | 355 | 0.27 | direct:31, verifier_loop:27, self_refine:21, memory:13, plan_execute:8 |
| 25 | ts_per_type | 1.44 ± 0.07 | 3.92 ± 0.32 | 0.651 | 0.543 ± 0.006 | 0.593 | 378 | 0.25 | verifier_loop:25, memory:23, self_refine:21, direct:19, plan_execute:12 |
| 50 | random | 2.94 ± 0.08 | 7.95 ± 0.47 | 0.657 | 0.534 ± 0.007 | 0.586 | 411 | 0.00 | random:100 |
| 50 | best_fixed | 2.98 ± 0.00 | 7.77 ± 0.43 | 0.661 | 0.545 ± 0.007 | 0.591 | 361 | 0.28 | direct:32, verifier_loop:28, self_refine:23, plan_execute:11, memory:6 |
| 50 | eps_greedy | 1.99 ± 0.36 | 7.09 ± 0.51 | 0.673 | 0.550 ± 0.006 | 0.600 | 361 | 0.37 | verifier_loop:37, direct:27, self_refine:18, memory:11, plan_execute:7 |
| 50 | ucb1 | 2.73 ± 0.07 | 7.28 ± 0.39 | 0.670 | 0.551 ± 0.005 | 0.598 | 340 | 0.30 | direct:32, verifier_loop:30, self_refine:18, memory:15, plan_execute:5 |
| 50 | thompson | 2.48 ± 0.19 | 7.56 ± 0.42 | 0.664 | 0.549 ± 0.006 | 0.598 | 360 | 0.34 | verifier_loop:34, direct:27, self_refine:21, memory:12, plan_execute:6 |
| 50 | ts_per_type | 2.66 ± 0.13 | 7.68 ± 0.34 | 0.662 | 0.550 ± 0.006 | 0.595 | 349 | 0.21 | direct:37, verifier_loop:21, self_refine:18, memory:12, plan_execute:12 |

References on held-out: best fixed (full info) = verifier_loop 0.566; per-type oracle 0.549 ({'math': 'direct', 'other': 'plan_execute', 'sequence': 'verifier_loop', 'string': 'direct'}); per-task oracle 0.678 (success 0.70); random expected 0.533.

### Mean reward by task type × pattern (full information)

| pool/type | n | direct | plan_execute | self_refine | verifier_loop | memory | best |
|---|---|---|---|---|---|---|---|
| online/math | 27 | 0.465 (0.48) | 0.323 (0.41) | 0.436 (0.48) | 0.442 (0.52) | 0.314 (0.37) | direct |
| online/other | 7 | 0.553 (0.57) | 0.637 (0.71) | 0.523 (0.57) | 0.470 (0.57) | 0.368 (0.43) | plan_execute |
| online/sequence | 50 | 0.783 (0.80) | 0.666 (0.74) | 0.755 (0.80) | 0.827 (0.88) | 0.721 (0.78) | verifier_loop |
| online/string | 16 | 0.673 (0.69) | 0.316 (0.38) | 0.648 (0.69) | 0.631 (0.69) | 0.509 (0.56) | direct |
| heldout/math | 10 | 0.485 (0.50) | 0.535 (0.60) | 0.457 (0.50) | 0.510 (0.60) | 0.442 (0.50) | plan_execute |
| heldout/sequence | 29 | 0.535 (0.55) | 0.377 (0.45) | 0.508 (0.55) | 0.509 (0.59) | 0.494 (0.55) | direct |
| heldout/string | 11 | 0.714 (0.73) | 0.569 (0.64) | 0.690 (0.73) | 0.766 (0.82) | 0.767 (0.82) | memory |

### Online selection share by task type, T=50

| policy | type | direct | plan_execute | self_refine | verifier_loop | memory |
|---|---|---|---|---|---|---|
| random | math | 0.20 | 0.19 | 0.20 | 0.21 | 0.19 |
| random | other | 0.26 | 0.20 | 0.18 | 0.17 | 0.19 |
| random | sequence | 0.20 | 0.19 | 0.20 | 0.20 | 0.21 |
| random | string | 0.21 | 0.20 | 0.20 | 0.19 | 0.21 |
| best_fixed | math | 0.20 | 0.20 | 0.21 | 0.19 | 0.20 |
| best_fixed | other | 0.21 | 0.18 | 0.22 | 0.17 | 0.21 |
| best_fixed | sequence | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| best_fixed | string | 0.20 | 0.20 | 0.20 | 0.22 | 0.19 |
| eps_greedy | math | 0.23 | 0.10 | 0.22 | 0.29 | 0.16 |
| eps_greedy | other | 0.20 | 0.09 | 0.21 | 0.32 | 0.18 |
| eps_greedy | sequence | 0.22 | 0.10 | 0.24 | 0.30 | 0.15 |
| eps_greedy | string | 0.23 | 0.10 | 0.25 | 0.30 | 0.13 |
| ucb1 | math | 0.22 | 0.18 | 0.20 | 0.22 | 0.19 |
| ucb1 | other | 0.26 | 0.18 | 0.16 | 0.20 | 0.21 |
| ucb1 | sequence | 0.22 | 0.17 | 0.20 | 0.23 | 0.18 |
| ucb1 | string | 0.21 | 0.19 | 0.20 | 0.20 | 0.19 |
| thompson | math | 0.24 | 0.15 | 0.21 | 0.23 | 0.17 |
| thompson | other | 0.22 | 0.15 | 0.21 | 0.26 | 0.17 |
| thompson | sequence | 0.22 | 0.16 | 0.22 | 0.23 | 0.17 |
| thompson | string | 0.23 | 0.18 | 0.22 | 0.24 | 0.13 |
| ts_per_type | math | 0.21 | 0.18 | 0.23 | 0.19 | 0.18 |
| ts_per_type | other | 0.22 | 0.24 | 0.16 | 0.19 | 0.19 |
| ts_per_type | sequence | 0.26 | 0.17 | 0.21 | 0.21 | 0.16 |
| ts_per_type | string | 0.21 | 0.17 | 0.18 | 0.26 | 0.19 |

Latency reliability: 96 identical-request pairs (direct vs verifier_loop passing first try): median relative |Δlatency| = 0.35, p90 = 0.96; corr(latency, tokens) = 0.70.
