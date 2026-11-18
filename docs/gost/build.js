const fs = require('fs');
const {d, P, H1, footer, run, mm, FONT} = require('./gost_lib');
const {Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, TableOfContents} = d;

const center = (t, o = {}) => new Paragraph({alignment: AlignmentType.CENTER, spacing: {line: 360, before: o.before || 0, after: o.after || 0}, children: [run(t, o)]});
const title = [
  center('МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ'),
  center('[Наименование образовательной организации]', {after: 1200}),
  center('ОТЧЁТ', {bold: true, before: 600}),
  center('о научно-исследовательской работе', {bold: true}),
  center('на тему', {before: 240}),
  center('«Автоматическая оптимизация harness LLM-агентов в условиях малых данных и ограниченных вычислительных ресурсов: анализ AutoSaddler и среда оценки»', {bold: true, after: 1800}),
  center('Исполнитель: [ФИО], [группа]'),
  center('Руководитель: [ФИО, должность]', {after: 1800}),
  center('2026', {before: 1200}),
];

const abstract = [
  H1('Реферат', {center: true, pageBreak: true, plain: true}),
  P('Отчёт содержит 6 разделов, таблицы и список из 9 источников. Ключевые слова: HARNESS, LLM-АГЕНТ, AUTOSADDLER, ОПТИМИЗАЦИЯ ПОДСКАЗОК, МАЛЫЕ ДАННЫЕ, MBPP, HUMANEVAL, TERMINAL-BENCH, OLLAMA.'),
  P('Объект исследования — алгоритм автоматической оптимизации harness AutoSaddler. Цель — выяснить механизмы и пределы его применимости при малом числе задач, получить базовые показатели на локальной модели, создать воспроизводимую среду и проверить гипотезы об улучшении. Выполнены анализ архитектуры и кода AutoSaddler V2, анализ предшествующих прогонов проекта, шесть прогонов AutoSaddler на HumanEval и пять проверок гипотез на Qwen2.5-7B. Показано, что на выборках в десятки задач штатный критерий принятия фиксирует шум, а шум перефразирования промпта (±6 п.п.) превышает полученные приросты.'),
];

const doc = new Document({
  creator: 'harness_lab', title: 'Отчёт о НИР: оптимизация harness',
  features: {updateFields: true},
  styles: {default: {document: {run: {font: FONT, size: 28}}},
    paragraphStyles: [
      {id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: {font: FONT, size: 28, bold: true, color: '000000'}, paragraph: {outlineLevel: 0}},
      {id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: {font: FONT, size: 28, bold: true, color: '000000'}, paragraph: {outlineLevel: 1}}]},
  numbering: {config: [{reference: 'dash', levels: [{level: 0, format: LevelFormat.BULLET, text: '–', alignment: AlignmentType.LEFT,
    style: {paragraph: {indent: {left: mm(12.5) + 360, hanging: 360}}, run: {font: FONT}}}]}]},
  sections: [{
    properties: {page: {size: {width: 11906, height: 16838}, margin: {top: mm(20), bottom: mm(20), left: mm(30), right: mm(15)}}, titlePage: true},
    footers: {default: footer(), first: new d.Footer({children: [new Paragraph({children: []})]})},
    children: [
      ...title,
      ...abstract,
      H1('Содержание', {center: true, plain: true}),
      new TableOfContents('Содержание', {hyperlink: true, headingStyleRange: '1-2'}),
      H1('Введение', {center: true}),
      P('Языковые модели в роли агентов работают внутри «обвязки» (harness) — набора подсказок, инструментов и правил цикла, качество которой сильно влияет на результат. Автоматизация её настройки по трассам выполнения предложена в AutoSaddler [1]. Опубликованные результаты получены на облачных агентных моделях и сотнях задач, тогда как практическая работа с локальными моделями располагает десятками задач и ограниченным временем.'),
      P('Цель работы — определить, как AutoSaddler ведёт себя в таких условиях. Задачи: (1) разобраться в алгоритмах, механизмах оптимизации и ограничениях при малом количестве данных; (2) проанализировать предшествующие прогоны, провести дополнительные запуски и определить базовые показатели; (3) подготовить среду оценки harness с запуском одной командой; (4) предложить улучшения AutoSaddler для малоресурсных вычислений, сформулировать пять гипотез и проверить их.'),
      ...require('./sections_static')(),
      ...require('./sections_results')(),
      ...require('./sections_final')(),
    ],
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(process.argv[2] || 'report.docx', b); console.log('written'); });
