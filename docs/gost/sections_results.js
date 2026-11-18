const fs = require('fs');
const path = require('path');
const {P, H1, H2, item, cap, table, gap} = require('./gost_lib');

const S = JSON.parse(fs.readFileSync(path.join(__dirname, '../../results/harness_lab/summary.json'), 'utf8'));
const pc = x => (x == null ? '—' : (100 * x).toFixed(1).replace('.', ','));
const sg = x => (x >= 0 ? '+' : '−') + Math.abs(100 * x).toFixed(1).replace('.', ',');

module.exports = function () {
  const out = [];
  // ---------- Раздел 3
  out.push(H1('3 Среда оценки harness'));
  out.push(H2('3.1 Выбор модели и оборудования'));
  out.push(P('Эксперименты выполнялись на ноутбуке с процессором Apple M4 и 16 ГБ единой памяти. Целевой моделью и моделью-оптимизатором выбрана Qwen2.5-7B-Instruct (квантование Q4 в среде Ollama, около 4,7 ГБ) [7]: она помещается в память вместе с контекстом 8 192 токенов, работает полностью на графическом ускорителе и даёт 3–15 секунд на задачу. Более крупные модели (14 млрд параметров и выше) оставляют слишком мало памяти для одновременной работы Docker, а облачные модели нарушали бы цель малоресурсной постановки. Использование одной и той же модели в обеих ролях (исполнитель и оптимизатор) соответствует крайнему случаю нехватки ресурсов.'));
  out.push(H2('3.2 Состав среды'));
  out.push(P('Среда размещена в каталоге harness_lab репозитория проекта и состоит из следующих модулей (таблица 4).'));
  out.push(cap('Таблица 4 – Модули среды harness_lab'));
  out.push(table(['Модуль', 'Назначение'], [
    ['llm.py', 'Клиент Ollama с учётом токенов и времени каждого вызова'],
    ['code_harness.py', 'Harness из трёх текстовых компонентов (solver_prompt, checklist, repair_prompt); цикл «решение → видимый тест → один ремонт → оценка по всем тестам», изолированный запуск кода'],
    ['benchmarks.py', 'MBPP-sanitized (30/20/50 задач, зафиксированный манифест), HumanEval (164 задачи, закреплённое разбиение 40/24/100)'],
    ['evaluate.py, stats.py', 'Кэшируемая возобновляемая оценка, интервалы Уилсона, парный бутстрэп, точный тест Макнемара'],
    ['ollama_provider.py', 'Провайдер AutoSaddler для локальной модели: одна безинструментальная сессия с выводом JSON, валидируемая штатной схемой'],
    ['code_scenario.py', 'Плагин сценария для AutoSaddler V2: оценщик, построитель улик (только видимые тесты и класс ошибки), набор промптов'],
    ['policies.py', 'Малоресурсные политики: критерий net_wins_k и выбор батча failure_first'],
    ['run_autosaddler.py', 'Запуск AutoSaddler (политики default и lite) с последующей оценкой на закрытом тесте'],
    ['tb_runner.py', 'Запуск Terminal-Bench (агент terminus, модель через LiteLLM/Ollama) и разбор отказов'],
    ['pipeline.py, report.py, run.sh', 'Оркестратор стадий, генератор отчёта, единая точка входа'],
  ], [2700, 6400]));
  out.push(gap());
  out.push(H2('3.3 Запуск одной командой'));
  out.push(P('Единая точка входа — скрипт run.sh. Команда ./run.sh setup устанавливает AutoSaddler (uv sync, коммит закреплён), CLI Terminal-Bench и проверяет Ollama и Docker; ./run.sh smoke выполняет проверку всей цепочки на минимальных бюджетах; ./run.sh full выполняет все базовые замеры, шесть прогонов AutoSaddler, Terminal-Bench, гипотезы и формирует отчёт results/harness_lab/REPORT.md. Каждая стадия идемпотентна: при наличии результата она пропускается, флаги --only и --force выбирают стадии. Оценки модели кэшируются по хэшу (harness, модель, параметры, задача), поэтому повторные эксперименты не тратят время на уже посчитанное. Новый эксперимент добавляется как стадия в pipeline.py и вызывает общие функции evaluate и paired_compare.'));
  out.push(H2('3.4 Адаптация AutoSaddler к локальной модели'));
  out.push(P('Штатные провайдеры AutoSaddler требуют облачных агентных моделей. Реализован внешний провайдер, регистрируемый в реестре без изменения исходного кода AutoSaddler. Он помещает в один запрос системный контекст, схему результата и все файлы рабочей области (текущий candidate.json, улики обучающих задач), требует ответ в формате JSON и передаёт его штатному BaseAgentProvider, который проверяет схему; повторы и тайм-ауты остаются за движком. Ограничение: оптимизатор не умеет самостоятельно исследовать историю и трассы, поэтому оценка относится к упрощённому варианту методики, а не к полному воспроизведению статьи.'));

  // ---------- Раздел 4
  out.push(H1('4 Базовые показатели и результаты прогонов'));
  out.push(H2('4.1 Базовые показатели'));
  out.push(P('Таблица 5 содержит показатели исходного harness (seed). Доверительные интервалы Уилсона шириной 15–30 п.п. показывают, что на выборках такого размера различия меньше 10 п.п. неразличимы.'));
  out.push(cap('Таблица 5 – Базовые показатели Qwen2.5-7B'));
  const bl = [['base_mbpp_train_repair', 'MBPP train (30), repair'], ['base_mbpp_val_repair', 'MBPP val (20), repair'],
    ['base_mbpp_test_direct', 'MBPP test (50), без repair'], ['base_mbpp_test_repair', 'MBPP test (50), repair'],
    ['base_humaneval_train', 'HumanEval train (40)'], ['base_humaneval_val', 'HumanEval dev (24)'],
    ['base_humaneval_test', 'HumanEval test (100)'], ['base_humaneval_direct', 'HumanEval, все 164']];
  out.push(table(['Конфигурация', 'pass@1, %', 'Токенов на задачу', 'Секунд на задачу'],
    bl.filter(([k]) => S[k]).map(([k, l]) => [l, pc(S[k].pass_at_1), Math.round(S[k].tokens_per_task), S[k].latency_s_per_task.toFixed(1).replace('.', ',')]),
    [4000, 1700, 1800, 1600]));
  out.push(gap());
  out.push(P('Обучающая часть MBPP почти насыщена (96,7 %), поэтому оптимизация на ней невозможна: диагностика не запускается без провалов. HumanEval даёт 25 % провалов на обучающей части, поэтому AutoSaddler запускался на нём.'));
  out.push(H2('4.2 Прогоны AutoSaddler'));
  out.push(P('Выполнено шесть прогонов (по три сида разбиения для каждой политики): 40 обучающих и 24 dev-задачи HumanEval, 4 итерации, батч 4 задачи; результат оценивался на закрытых 100 задачах. Политика default — штатная (строгое улучшение, любые компоненты, подробные улики, фиксированный порядок батчей); lite — предложенная в разделе 5.'));
  out.push(cap('Таблица 6 – Прогоны AutoSaddler на HumanEval'));
  const rows = (S.runs || []).map(x => [x.run_id.replace('as_', ''), x.accepted + '/' + x.proposals,
    sg(x.mean_train_batch_gain_of_accepted), (x.dev_children.length ? x.dev_children.map(pc).join('; ') : '—'),
    pc(x.test_seed) + ' → ' + pc(x.test_selected), sg(x.test_delta) + ' (' + x.test_p.toFixed(2).replace('.', ',') + ')', x.proposer_tokens]);
  out.push(table(['Прогон', 'Принято / предложено', 'Прирост на батче, п.п.', 'dev принятых, % (seed 91,7)', 'test, %', 'Δtest, п.п. (p)', 'Токены оптим.'],
    rows, [1300, 1200, 1300, 1700, 1400, 1500, 1000]));
  out.push(gap());
  const ad = S.agg_default, al = S.agg_lite;
  if (ad && al) {
    out.push(P(`В среднем штатная политика принимала ${(100 * ad.accept_rate).toFixed(0)} % предложений с приростом ${sg(ad.mean_train_gain)} п.п. на батче, однако все принятые патчи снижали dev-балл ниже seed (83,3–87,5 % против 91,7 %), и итоговым кандидатом во всех трёх прогонах оставался seed (Δtest = 0). Политика lite принимала ${(100 * al.accept_rate).toFixed(0)} % предложений; средний прирост на тестовой выборке составил ${sg(al.mean_test_delta)} п.п. при p ≥ 0,69, то есть неотличим от нуля. Расход токенов оптимизатора практически одинаков (${Math.round(ad.mean_proposer_tokens)} и ${Math.round(al.mean_proposer_tokens)}).`));
  }
  out.push(H2('4.3 Terminal-Bench'));
  const tb = (S.terminal_bench || []).find(t => t.run_dir.endsWith('tb_base'));
  const names = {resolved: 'решена', agent_timeout: 'тайм-аут агента (зацикливание)', test_timeout: 'тайм-аут проверки', unknown_failure: 'неверный результат'};
  out.push(P('Запущены пять лёгких задач набора terminal-bench-core 0.1.1 (hello-world, fix-permissions, create-bucket, heterogeneous-dates, csv-to-parquet) с агентом terminus и моделью ollama/qwen2.5:7b, один сид, одна попытка. Базовый показатель приведён в таблице 7.'));
  out.push(cap('Таблица 7 – Terminal-Bench, агент terminus, Qwen2.5-7B'));
  if (tb) out.push(table(['Задача', 'Исход', 'Проверки'], tb.tasks.map(t => [t.task_id, names[t.failure_mode] || t.failure_mode,
    t.parser_results ? Object.entries(t.parser_results).map(([k, v]) => k.split(' - ')[0].replace('test_', '') + ': ' + (v === 'passed' ? 'ок' : 'нет')).join('; ') : '—']), [2300, 3000, 3800]));
  out.push(gap());
  if (tb) out.push(P(`Решено ${tb.n_resolved} из ${tb.n_tasks} задач (${(100 * tb.accuracy).toFixed(0)} %), на выполнение ушло ${Math.round(tb.wall_s / 60)} минут. Выборка из пяти лёгких задач даёт интервал Уилсона примерно [4; 62] %, поэтому показатель годится как ориентир порядка величины, а не как оценка. Характерный отказ — деградация цикла: в hello-world агент создал файл, но затем многократно повторял одну и ту же команду записи до тайм-аута (360 с), в результате содержимое файла оказалось неверным, хотя задача тривиальна. Это указывает, что при слабой модели основная доля отказов приходится на управление циклом (остановка, защита от повторов), то есть на компоненты harness, а не на знания модели; такие компоненты как раз являются целью AutoSaddler, но их оптимизация на Terminal-Bench требует около 9 минут на задачу и поэтому в данной работе не выполнялась.`));
  return out;
};
