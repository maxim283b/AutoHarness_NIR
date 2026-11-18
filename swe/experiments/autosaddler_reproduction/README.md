# Общий OpenCode-харнес и AutoSaddler

## Продолжение после исправления таймаута (4 ноября 2026)

Активный запуск: `go-terminal-recovery-20261104`, конфигурация
`configs/terminal_go_plus_20261104.yaml`. Пользователь поручил довести эксперимент
до завершения. Старый запуск остановился на AgentTimeoutError в initial dev;
его артефакты сохранены как диагностические. Исправление адаптера зафиксировано
отдельной версией, initial dev измеряется заново.

Только штатный AgentTimeoutError с подтверждённой очисткой активного дерева
процессов и результатом официального verifier считается исходом задачи.
Остальные инфраструктурные/API ошибки остаются ошибками. При завершении агента
останавливается его ещё активная команда с потомками, включая отдельные сессии;
сервисы от уже завершённых команд остаются доступными оценщику. Проверка проведена
в настоящем Docker и через настоящий Harbor с искусственным таймаутом без модели.
61 локальный тест прошёл. Таймауты официальных задач не увеличены.

Старые 91 запрос, 1468561 токен и 0.1725182 allowance включены в общий бюджет
через `budget_prior_runs`; 20 allowance/2000 запросов относятся ко всей цепочке.
`gateway/status.json` показывает расходы с учётом истории, поэтому прибавлять
к нему прошлый расход повторно нельзя. Полный итог по ролям —
`campaign_accounting.json`; источником всех списаний служат provider usage,
включая ответы, которые могли не попасть в локальные CLI-логи при таймауте.
Первичный локальный отчёт сохраняется отдельно как `native_comparison.json`.
Снимок исходников до запуска — `source_snapshot.zip` в каталоге нового запуска.

## Текущий первый этап: Terminal-Bench 2.0 (3 ноября 2026)

Финальная конфигурация первого ограниченного пилота:
`configs/terminal_go_plus_20261103.yaml`. Используется только Terminal-Bench 2.0.
Целевой агент — OpenCode 1.19.5 / Qwen3.7 Plus; оптимизатор — OpenCode ACP /
Kimi K2.7 Code. Обе модели работают через нативные API Go Plus. OAuth берётся
из существующего Desktop-подключения и остаётся на хосте. В контейнеры передаётся
только временный ключ локального посредника учёта. Codex ACP здесь не используется.

Размер: 4 train, 3 dev, батч 2, максимум 4 итерации (2 capability + 2 steering),
до 31 search rollout. Итоговое сравнение: 5 test-задач × H0/H1 × один повтор =
10 выполнений. Это пилот метода V2 на OpenCode, не полное воспроизведение статьи.
Локальный test содержит 40 задач; точное авторское разбиение не найдено.

Калибровка на train `log-summary-date-ranges`: валидный результат 1.0, 4 вызова
Qwen, 41065 токенов. Проверка инструментов Kimi: 4 вызова, 22306 токенов.
Отдельный полный цикл настоящего оптимизатора с синтетическими оценками прошёл
за 22 запроса. Общий расход проверок — 0.14950607 allowance; API подтвердил
Go Plus и нулевую оплату с отдельного баланса. Синтетические оценки не являются
результатами бенчмарка. Одна train-задача не гарантирует расход остальных задач.

Бюджет пилота — максимум 20 allowance, 2000 запросов. По ролям:
оптимизатор 4/200, train+dev 8/1000, H0 test 4/400, H1 test 4/400.
Allowance — единица включённой квоты, не дополнительный платёж в долларах.
Перед каждым запросом проверяются все три окна квоты (остановка при 70%),
размер контекста и резерв стоимости. Нет автоматических повторов или смены
провайдера. Ответ должен подтверждать Go Plus и нулевое списание с баланса.
Настройка Extra Usage аккаунта не изменялась и отдельно не проверена;
превентивная защита — остановка задолго до исчерпания квоты. Новые платежи
и пополнения не выполняются. Лимиты одинаковы для H0/H1: максимум 200000
байт сериализованного запроса и 8192 выходных токена; штатные таймауты задач.
Ошибка API/инфраструктуры останавливает прогон и не превращается в score=0.

Перед запуском подготавливаются все 12 выбранных окружений без inference и
без проверки решений. Harbor install_only пропускает агента и verifier;
PrepareOnlyOpenCode дополнительно запрещает запуск агента, ключи удаляются.
Запуск эксперимента требует completed.json с теми же ID задач, ревизией H0
и неизменившимися Docker image IDs. Все ID, параметры и исходники фиксируются
до оптимизации; test-результаты не используются при выборе H1.

```powershell
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.terminal_preflight --install
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.go_experiment run --run-id go-terminal-pilot-20261103 --config experiments/autosaddler_reproduction/configs/terminal_go_plus_20261103.yaml --allowance-limit 20 --request-limit 2000 --n-test 5
```

Журналы запуска: `results/autosaddler_reproduction/go_experiments/<run-id>/`.
Трассы, события V2 и итоговая comparison.csv: `results/autosaddler_reproduction/runs/<run-id>/`.
Учёт токенов раздельный: optimizer, target_search, target_final_baseline,
target_final_optimized. Все вызовы дополнительно сохраняются в журнале Go API.
Незавершённый из-за квоты прогон не объявляется завершённым экспериментом.

Актуальность AutoSaddler проверена через git ls-remote: main/HEAD совпадает с
локальным e841543ce6cf6bdf690e65134b12a829ec9f66d1, версия V2 0.2.0.
Существующие Windows-патчи сохранены и зафиксированы в environment_lock.json.
Черновик terminal_go_plus.draft.yaml остаётся заблокированным.

## Общая постановка исследования

Один исходный харнес H0, один поиск AutoSaddler и один выбранный по dev харнес H1.
Общий план предусматривает оценку обоих харнесов на GAIA2, SWE-Bench Pro **V1**
и Terminal-Bench **2.0**; текущий первый пилот ограничен Terminal-Bench.
Внутри H0/H1 неизменны версия OpenCode (1.19.5), целевая модель и ограничения.
Оптимизируются `instructions.md`, `.opencode/tools/` и `.opencode/skills/`.
Исходный код ядра OpenCode и адаптеры/оценщики не входят в изменяемую область.

Использован движок **AutoSaddler V2, 0.2.0, e841543**, включая
diagnosis/patch, reflection, EvoDAG, проверку на том же батче, dev gate,
durable events и resume. V1 — реализация статьи. Поэтому это воспроизведение
метода на новом общем харнесе, а не буквальное воспроизведение таблиц статьи.
Для общей задачи добавлены батчи с одинаковым числом задач каждого бенчмарка
и выбор кандидата по macro-average трёх dev-результатов.
В исходных локальных checkouts уже были Windows-совместимые изменения:
они сохранены в `autosaddler_local.patch` и `meta_are_windows.patch`, их хэши
и исходные ревизии записаны в `environment_lock.json`. Чистым upstream-кодом
эти рабочие деревья не объявляются. Служебные Git worktree размещаются в
коротких путях `.test_tmp/ocw`, обходя ограничение MinGit `$GIT_DIR too big`.

## Данные

| Бенчмарк | Train | Dev | Test | Источник разбиения |
|---|---:|---:|---:|---|
| GAIA2 | 75 | 65 | 300 | Исходные манифесты AutoSaddler; U29 / U30 / U21,U22,U27 |
| SWE-Bench Pro V1 | 79 | 80 | 237 | Репозитории и размеры статьи; 40 dev ID на репозиторий с seed=42 |
| Terminal-Bench 2.0 | 30 | 19 | 40 | Фиксированное случайное разбиение с seed=42 |

**Точные ID dev SWE и разбиения TB2 из статьи в доступном upstream не найдены.**
Локальные ID закреплены в `manifests/`; они не выдаются за точные ID авторов.
Все 440 выбранных сценариев GAIA2 загружены по ревизии
`78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99`.
SWE загружен по неизменяемому тегу `v1.0`; на этом теге конфигурация называется
`default`. Текущий `default` на main уже соответствует другому бенчмарку V2.
Test не передаётся оптимизатору и используется только после фиксации H1.

## Среда и изоляция

- `.venv-experiment`: AutoSaddler, Harbor, datasets, Docker SDK и pytest.
- `third_party/Meta-ARE/.venv`: отдельные совместимые зависимости ARE/MCP.
- Оптимизатор запускается через OpenCode ACP в контейнере
  `autoharness-opencode:1.19.5`; видит только рабочее дерево кандидата и
  предоставленную AutoSaddler историю/train evidence.
- GAIA2 запускает тот же OpenCode в контейнере и предоставляет только
  агентные инструменты через авторизованный MCP. События и judge остаются в ARE.
- SWE генерирует патч в контейнере с базовым коммитом, затем применяет его в
  **свежем** контейнере штатного оценщика. Исправлена только передача LF-скриптов
  с Windows; логика fail-to-pass/pass-to-pass сохранена.
- TB2 использует Harbor и его штатный verifier. Наш адаптер устанавливает
  ту же версию OpenCode и тот же набор файлов харнеса.
- Остальные Docker-образы задач загружаются по мере необходимости до запуска
  агента. Полный набор из сотен образов заранее не скачан.

## Проверки без покупки подписки

Из корня проекта:

```powershell
.\experiments\autosaddler_reproduction\run.ps1 -Action doctor
.\experiments\autosaddler_reproduction\run.ps1 -Action smoke
.\.venv-experiment\Scripts\python.exe -m pytest experiments/autosaddler_reproduction/tests -q --basetemp=.test_tmp/pytest-experiment-check
```

Дополнительные проверки, которые запускаются отдельно и сохраняют артефакты:

```powershell
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.mock_model
.\third_party\Meta-ARE\.venv\Scripts\python.exe -m experiments.autosaddler_reproduction.gaia2_event_smoke
.\third_party\Meta-ARE\.venv\Scripts\python.exe -m experiments.autosaddler_reproduction.gaia2_oracle_smoke
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.swe_adapter
```

`smoke` использует синтетические оценки и проверяет весь цикл AutoSaddler и
идемпотентное возобновление. `mock_model` запускает настоящий OpenCode через
ACP с локальным сервером сценарных ответов, сверяет ответы и количество
вызовов с native usage. `gaia2_event_smoke` проверяет инструменты, уведомления
во время хода, AUI и остановку. Это **не результаты решения бенчмарка моделью**.

Дополнительно выполнен короткий проверочный запрос к `gpt-6-luna` через
`codex-acp` и существующий вход ChatGPT. Он расходует доступ Codex, а не баланс
OpenRouter/OpenCode или API-кредиты. Этот иной агент не заменяет OpenCode в
основном эксперименте. Его не нужно повторять для обычного doctor.

Для временных проверок через текущую подписку добавлен `acp_checks`:

```powershell
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.acp_checks protocol --run-id acp-check-001
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.acp_checks gaia --run-id acp-check-001
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.acp_checks swe --run-id acp-check-001
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.acp_checks terminal --run-id acp-check-001
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.acp_check_report --prefix acp-check
```

Это реальные вызовы `gpt-6-luna` через `codex-acp`, без API-ключа; они расходуют
обычный лимит Codex. Проверка GAIA2 использует календарь и AUI реального
train-сценария, но не оценивает его исходную цель. SWE и TB2 решают по одной
train-задаче с официальным оценщиком. Это иной агент, поэтому результаты
не включаются в сравнение OpenCode H0/H1. Каждый запуск ограничен временем и
20 командами внутри отдельного контейнера; клиент разрешает только разовые
действия известного MCP-сервера тестовой задачи. Разрешения на прочие действия
остаются отклонёнными. Токены учитываются по native ACP turn aggregate:
`inputTokens` у этого адаптера не включает отдельно переданные cache-токены.
Отчёт включает расходы неудачных попыток. Для новых попыток нужен новый run ID;
сбор отчёта повторяется без вызовов модели.

### Модель для OpenCode через ACP

`acp_gateway.py` реализует отдельный локальный адаптер:
OpenCode → OpenAI-compatible HTTP → Codex ACP → существующая авторизация ChatGPT.
Сам OpenCode исполняет инструменты и хранит историю. Backend Codex получает
переданную беседу и типизированные MCP-инструменты выбора действий. Эти
MCP-инструменты только записывают намерение; реальное действие выполняет OpenCode.
Native shell, плагины, web search и подагенты backend отключены.
Это экспериментальный транспорт с фиксированными дополнительными инструкциями,
а не обычный API-вызов модели. Его нужно одинаково закреплять для H0/H1.

```powershell
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.acp_gateway --smoke
```

Сквозная проверка в `acp_opencode_wire/run-cfcb6e95/result.json` подтвердила:
OpenCode сам прочитал файл, применил патч, повторно прочитал результат и выдал
финальный ответ. Все 4 completion и 192 420 токенов совпали между native usage
OpenCode и ACP transport ledger. Это проверка связки, не результат бенчмарка.

Gateway поддерживает предел вызовов/токенов и `minimum_weekly_remaining`:
перед запросом локальная команда ACP `/status` проверяет недельный остаток.
Если остаток достиг порога или недоступен, новый запрос модели не отправляется.
Один запрос может пересечь порог до следующей проверки; точный процент расхода
не вычисляется из сырых токенов. Автоматический переход на API или платное
пополнение не используется. Пользователь разрешил расход до остатка 50%.

Пилот `acp-pilot-v3-20261102` использует зафиксированный
`configs/acp_pilot_20261102.yaml`: 4 итерации (2 capability + 2 steering),
батч 3, по 2 train и 1 dev-задаче каждого бенчмарка. На test оцениваются
по 3 задачи каждого бенчмарка для H0/H1, один повтор: 18 финальных выполнений.
Выборка детерминирована seed=42 до просмотра test; ID находятся в
`acp_experiments/acp-pilot-v3-20261102/frozen_plan.json`.
Это ограниченный пилот, а не полный test или воспроизведение чисел статьи.
Judge также `gpt-6-luna` через ACP вместо Llama статьи; адаптеры и модель
одинаковы для H0/H1. `acp-pilot-20261102` и `acp-pilot-v2-20261102` остановлены
до test из-за транспортной ошибки и ошибки ограничения числа ходов GAIA;
их отладочные результаты не используются как качество H0.

Текущий запуск использует Codex через ACP как backend модели внутри OpenCode.
ACP получает только подписанную копию файла авторизации временного
изолированного профиля, без пользовательских MCP и плагинов. Каждый модельный
ход получает отдельный ACP-процесс; OpenCode остаётся владельцем реальных
файловых, терминальных и benchmark-инструментов. Backend и judge отличаются
от статьи, поэтому итог маркируется как ограниченный методологический пилот,
а не буквальная репликация чисел статьи.

```powershell
.\.venv-experiment\Scripts\python.exe -m experiments.autosaddler_reproduction.acp_experiment run --run-id acp-pilot-v3-20261102 --config experiments/autosaddler_reproduction/configs/acp_pilot_20261102.yaml --minimum-weekly-remaining 50 --n-test 3 --test-repetitions 1 --max-requests 2500 --max-tokens 120000000
```

Команда сохраняет native engine resume и завершённые test-попытки. Повторять её
можно только после завершения предыдущего процесса, с неизменными исходниками,
моделями и конфигурацией. При достижении порога данные остаются на диске.
Новая покупка, сброс лимитов или переключение провайдера не выполняются.
ACP 2.5.0 имеет локальную правку `codex_acp_scope.patch`, сохраняющую запрет
глобальных MCP при добавлении инструментов задачи; хэш адаптера закреплён.

## После подключения моделей

1. Скопировать `configs/experiment.template.yaml` в `configs/local.yaml`.
2. Задать `scenario.settings.target_model` и `provider.settings.model` явными
   ID вида `opencode-go/<model>`. Выбор моделей и параметры замораживаются
   перед baseline. Не включать автоматический fallback на другого провайдера.
3. Передать ключ Go через переменную окружения `OPENCODE_GO_API_KEY`.
   Ключи не записываются в YAML, git или команды запуска.
4. Задать `scenario.settings.judge` в формате ARE `LLMEngineConfig`:
   `model_name`, `provider`, `endpoint`. Judge статьи — Llama-3.3-70B-Instruct.
   Доступ к нему/другому явно объявленному judge нужно настроить отдельно;
   наличие подписки Go не означает, что именно эта модель в неё входит.
5. Выполнить train-only калибровку, по измеренным токенам и квотам зафиксировать
   число задач/итераций и запустить основной поиск с новым run ID.

```powershell
.\experiments\autosaddler_reproduction\run.ps1 -Action calibrate -RunId calibration-001
.\experiments\autosaddler_reproduction\run.ps1 -Action optimize -RunId experiment-001
.\experiments\autosaddler_reproduction\run.ps1 -Action evaluate -RunId experiment-001
```

Числа 8 итераций / 150 rollout в шаблоне — **технический пример**, не финальный
научный бюджет. Размер фиксируется после калибровки, до просмотра test.
Переход capability → steering по умолчанию вычисляется как один проход
обучающих задач при сбалансированных батчах; в шаблоне это 6 + 2 итерации.
Изменение исходников, моделей, манифестов или параметров требует нового run ID;
старый run ID используется только для настоящего resume.

## Учёт расходов

SQLite request ledger разделяет `optimizer`, `target_search`,
`target_final_baseline`, `target_final_optimized` и `judge`.
Optimizer usage также сохраняется штатными событиями AutoSaddler по фазам.
Входные, выходные, cache и reasoning счётчики хранятся раздельно;
В ACP/provider usage output уже включает reasoning. В native OpenCode 1.19.5
`tokens.output` хранит видимую часть отдельно от `tokens.reasoning`;
нормализация объединяет их один раз и сверяет сумму с `tokens.total`.
Неизвестные usage не заменяются
нулём. Вспомогательные title/summary-запросы и подагенты отключены.
В попытках до `go-terminal-gitfix-20261104` включительно compaction была отключена.
После повторного превышения лимита контекста пользователь 4 ноября разрешил
новую версию `native-compaction-v1`: штатные auto compaction/prune OpenCode,
одинаковые для H0/H1 и оптимизатора. Модель сжатия совпадает с моделью родительской
сессии; её запросы помечены `purpose=compaction` и входят в прежние лимиты роли
и всей цепочки. Отдельная разбивка не прибавляется к общему расходу повторно.
Effective input limit равен 32000, reserved — 8192, максимум ответа сжатия — 4096.
Вывод инструмента свыше 8000 UTF-8 байт целиком сохраняется в артефактах;
агент получает начало, конец и путь для чтения нужных фрагментов. API guard
200000 байт и основной output limit 8192 остаются прежними. Отсутствие
учётного plugin в реальном запуске приводит к отказу до обращения к Go API.
Новая реализация требует нового run-id, снимка исходников и переноса расходов
всех предыдущих попыток; старые оценки не импортируются.
Новые custom tools не должны самостоятельно обращаться к моделям.

`comparison.csv` содержит результаты одного и того же H1 на всех бенчмарках.
Расходы общего поиска учитываются один раз, отдельно от test-инференса H0/H1.
Включённые в подписку вызовы тоже расходуют токены и allowance; цена подписки
не подменяет фактическую стоимость/квоту работы.

Проверка software environment без авторизации не гарантирует доступность
конкретной модели, её лимиты, поддержку параметров или корректность ответов.
После подключения credentials выполняется небольшой train/dev smoke на
выбранных моделях, затем замораживается окончательная конфигурация.
