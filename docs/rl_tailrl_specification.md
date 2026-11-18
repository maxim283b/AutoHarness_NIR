# Gate B: спецификация RL и TailRL

## RL optimizer

RL — reinforcement learning, обучение с подкреплением. В этом проекте `rl_optimizer` определяется как последовательная обучаемая policy:

- state: hash текущего harness, дискретизированный текущий score, число шагов, последние actions/rewards и cumulative budget;
- action: выбор компонента harness и типа содержательного изменения, после чего общий LLM proposer создаёт новый текст;
- transition: оценка нового candidate на заранее определённой search mini-batch;
- reward: изменение preregistered objective относительно parent с учётом стоимости;
- policy: softmax по обучаемым предпочтениям `theta[state, action]`;
- update: on-policy REINFORCE update с экспоненциальным baseline после каждого перехода;
- horizon: не менее четырёх зависимых шагов, а в planned core — 14 mutation steps после seed.

Это не bandit: state и следующее распределение действий зависят от текущего harness и всей истории, а policy обновляется по последовательным переходам.

## Опубликованный TailRL

Однозначно идентифицирован **Tail-Likelihood Reinforcement Learning** (TailRL), arXiv:2609.02987, официальный код `Zanette-Labs/TailRL`, commit `5682c6ac03387355e017ce966693266bb148fa10`.

Для группы из `N` rollout rewards алгоритм сортирует rewards, берёт gaps между соседними order statistics и делит каждый gap на число surviving rollouts. Официальная формула advantage до mean-centering:

`A_i = N * sum_{j <= i} (r_(j) - r_(j-1)) / (N - j + 1)`.

Это policy-gradient estimator для обучаемой генеративной policy. Он повышает вес редких high-reward rollouts и оптимизирует вероятность превысить случайный reward threshold.

## Что реализуемо здесь

Точный TailRL **не воспроизведён**. Причины:

1. frozen target нельзя обновлять;
2. общий LM Studio proposer также должен оставаться frozen для равенства методов;
3. табличная action policy не совпадает с генеративной token policy из статьи;
4. candidate budget 15 и одиночные action outcomes не образуют сопоставимые on-policy rollout groups официальных экспериментов.

Допустимое имя реализации: `tail_aware_rl_variant`. Она использует тот же настоящий sequential RL, но update reward заменяется заранее заданным сочетанием среднего reward и среднего reward худших 20% задач. Это проверка tail-aware objective, а не TailRL reproduction.

## Запреты именования

- UCB/Thompson sampling над mutation operators — bandit, не RL.
- evolutionary selection — не RL.
- `tail_aware_rl_variant` нельзя сокращать до TailRL в таблицах или выводах.
- Пока Gate A заблокирован, обе реализации разрешены только для unit/integration smoke, не для confirmatory claim.
