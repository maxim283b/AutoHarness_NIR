# Актуальные методы автоматической оптимизации harness

Срез источников: 9 октября 2026 года. Harness — внешняя оболочка агента: системная инструкция, цикл рассуждения, инструменты, память, проверка, исправление и остановка. Веса target-модели в этом проекте заморожены.

## Вывод для пилота

Для малого локального эксперимента выбран типизированный поиск по `HarnessGenome` и честно обозначенная `literature_inspired_baseline`. Это не заявляется воспроизведением AutoSaddler, AutoHarness или Airbnb Agent Harness Optimizer: их исходные outer loops требуют отдельного proposer-agent, иных benchmark adapters или изменяют исполняемый middleware. GEPA уже интегрирована локально, но не включена в обязательное ядро, чтобы не смешивать её proposer-бюджет с полностью детерминированными оптимизаторами при очень малом бюджете.

Последовательный эволюционный поиск выбран вместо байесовской оптимизации и RL. Причина — только три оцениваемых кандидата на один algorithm seed и смешанное дискретное пространство: этого недостаточно для устойчивой surrogate-модели, а RL означал бы обучение policy и потребовал бы гораздо больше переходов.

## Проверенные названия и источники

| Метод | Что меняет | Поиск | Отдельный proposer | Код и лицензия | Последнее существенное состояние, наблюдаемое 2026-10-09 | Решение |
|---|---|---|---|---|---|---|
| [AutoHarness: improving LLM agents by automatically synthesizing a code harness](https://arxiv.org/abs/2603.03329) | Исполняемый code-as-harness для допустимости действий, вплоть до полной policy | Итеративный синтез и исправление кода по feedback среды | Да, генератор кода | В статье не найден официальный открытый reference repository или лицензированный artifact | arXiv v1: 2026-02-10; ICLR RSI Workshop poster | Исключён: домен TextArena и исполняемый код не совпадают с безопасным типизированным пространством пилота |
| [AIMing Lab AutoHarness](https://github.com/aiming-lab/AutoHarness) | Governance/runtime: context, tool policy, audit, session persistence | Не является алгоритмом поиска кандидатов | Нет | Открытый код, MIT | v0.1.0 от 2026-04-01 | Не считать baseline оптимизации; возможный runtime-компонент |
| [Airbnb Agent Harness Optimizer](https://github.com/airbnb/agent-harness-optimizer) | System prompt и tool-call middleware | PRISM Genetic-Pareto, BetterHarness propose/accept, MIPROv2, GEPA | Да, outer model | Открытый код, MIT | EMNLP 2026 artifact; активный README и reproducibility configs на дату среза | Близкий framework baseline, но нужен custom benchmark и локальный outer-model smoke |
| [AutoSaddler](https://github.com/microsoft/AutoSaddler) | Prompts, tools, middleware и agent-loop через scenario plugin | Diagnosis-Patch, Reflection, Evolution/EvoDAG, dev gate | Да | Открытый код, MIT | V2 plugin engine; NeurIPS 2026 poster, обновление V2 в августе 2026 | Главный литературный ориентир; точное воспроизведение исключено из малого пилота |
| [GEPA](https://github.com/gepa-ai/gepa) | Текстовые компоненты, определённые adapter | Reflective mutation и Pareto-aware selection | Обычно да | Открытый код, MIT | Активный официальный repository; исходная статья 2025 | Дополнительный baseline; локальный adapter и mock smoke уже существуют |
| [DSPy MIPROv2](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md) | Instructions и few-shot examples | Bayesian optimization по предложенным вариантам | Да для instruction proposal | DSPy открыт под MIT | Официальная документация активна на дату среза | Исключён из core: три кандидата недостаточны для содержательной BO-модели |
| [HARBOR: Automated Harness Optimization](https://arxiv.org/abs/2604.20938) | Смешанные harness flags и resource settings | Constrained noisy Bayesian optimization, SAAS/TuRBO, multi-fidelity acquisition | Не обязательно для категориальных flags | Статья открыта; официальный reference solver/repository и его лицензия не обнаружены | arXiv v1: 2026-04-22 | Методически релевантен, но невоспроизводим локально как открытый baseline |
| [AFlow](https://github.com/FoundationAgents/AFlow) | Код и топология workflow | Monte Carlo Tree Search — поиск по дереву Монте-Карло | Да | Открытый repository; лицензию надо фиксировать по checkout | ICLR 2025 | Исключён: evaluation budget умножается на workflow scoring |
| [EvoAgentX](https://github.com/ANative-Lab/EvoAgentX) | Prompts, tools и workflow topology | Framework с TextGrad, MIPRO и AFlow | Зависит от optimizer | Открытый код, MIT | Активный repository в октябре 2026 | Не один алгоритм; интеграционный слой, не baseline |
| [TailRL](https://github.com/Zanette-Labs/TailRL) | Веса generative policy через advantage estimator | Policy-gradient RL по верхним хвостам reward | Нет отдельного proposer; нужна обучаемая policy | Официальный код открыт; явная лицензия на корне не подтверждена страницей repository | arXiv:2609.02987, 2026-09-02 | Не применим: нарушает frozen-model и требует GPU-training |

## Что означает TailRL в этом эксперименте

`TailRL` больше не неоднозначное название: это **Tail-Likelihood Reinforcement Learning**, метод, который повышает вероятность редких высоконаградных rollout и обновляет веса policy. Он не оптимизирует нижний хвост задач и не является harness-search методом.

В пилоте используется другая, заранее определённая `tail-aware objective` — цель, чувствительная к худшим задачам:

```text
objective_tail = 0.5 * mean_reward + 0.5 * mean(worst ceil(0.20 * n) task rewards)
```

При `n=10` нижний компонент — среднее двух худших task rewards. Это сравнение среднего качества с устойчивостью на сложных задачах, а не реализация TailRL.

## Отличия literature-inspired baseline

Локальная базовая линия заимствует у BetterHarness/AutoSaddler только проверяемую механику: один текущий champion, типизированный patch, оценка на общей frozen task sequence, принятие при улучшении и append-only provenance. Она не использует LLM-proposer, не редактирует verifier/tool code, не выполняет Diagnosis/Reflection/EvoDAG и потому называется `literature_inspired_baseline`, а не AutoSaddler.

## Ограничения поиска

- Даты GitHub обозначают наблюдаемое существенное публичное состояние, а не обещание, что последний commit был именно в этот день.
- Для AutoHarness-paper и HARBOR не найден лицензированный официальный reference implementation; статья сама по себе не даёт права приписывать сторонний код авторам.
- Никакие результаты из validation или held-out не использованы для выбора коэффициентов цели, пространства mutations или stopping rules.

