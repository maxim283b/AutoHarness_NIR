// ГОСТ 7.32-2017 helpers for docx-js
const d = require('docx');
const {Paragraph, TextRun, AlignmentType, HeadingLevel, Table, TableRow, TableCell, WidthType, BorderStyle,
  ShadingType, Footer, PageNumber, LevelFormat, PageBreak, TableOfContents, TabStopType} = d;

const FONT = 'Times New Roman';
const mm = v => Math.round(v * 56.7); // mm -> twips
const SIZE = 28; // 14 pt

const run = (t, o = {}) => new TextRun({text: t, font: FONT, size: o.size || SIZE, bold: o.bold, italics: o.italics, color: '000000', ...o.extra});

function P(text, o = {}) {
  const parts = Array.isArray(text) ? text : [text];
  return new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED,
    spacing: {line: 360, before: 0, after: 0},
    indent: o.noIndent ? undefined : {firstLine: mm(12.5)},
    keepNext: o.keepNext,
    children: parts.map(x => typeof x === 'string' ? run(x, o) : x),
  });
}
const H1 = (t, o = {}) => new Paragraph({
  heading: o.plain ? undefined : HeadingLevel.HEADING_1, pageBreakBefore: o.pageBreak !== false,
  alignment: o.center ? AlignmentType.CENTER : AlignmentType.LEFT,
  spacing: {line: 360, before: 0, after: 360}, indent: o.center ? undefined : {firstLine: mm(12.5)}, keepNext: true,
  children: [run(o.center ? t.toUpperCase() : t, {bold: true})],
});
const H2 = t => new Paragraph({
  heading: HeadingLevel.HEADING_2, alignment: AlignmentType.LEFT,
  spacing: {line: 360, before: 240, after: 240}, indent: {firstLine: mm(12.5)}, keepNext: true,
  children: [run(t, {bold: true})],
});
const item = (t) => new Paragraph({
  numbering: {reference: 'dash', level: 0}, alignment: AlignmentType.JUSTIFIED, spacing: {line: 360},
  children: [run(t)],
});
const cap = t => new Paragraph({alignment: AlignmentType.LEFT, spacing: {line: 360, before: 120, after: 0}, keepNext: true,
  indent: {firstLine: 0}, children: [run(t)]});

function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const border = {style: BorderStyle.SINGLE, size: 4, color: '000000'};
  const borders = {top: border, bottom: border, left: border, right: border};
  const cell = (t, w, bold) => new TableCell({
    width: {size: w, type: WidthType.DXA}, borders, margins: {top: 40, bottom: 40, left: 80, right: 80},
    children: [new Paragraph({alignment: AlignmentType.LEFT, spacing: {line: 240},
      children: [run(String(t), {size: 24, bold})]})],
  });
  return new Table({
    width: {size: total, type: WidthType.DXA}, columnWidths: widths,
    rows: [new TableRow({tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true))}),
      ...rows.map(r => new TableRow({cantSplit: true, children: r.map((c, i) => cell(c, widths[i], false))}))],
  });
}
const gap = () => new Paragraph({spacing: {line: 360}, children: []});
const footer = () => new Footer({children: [new Paragraph({alignment: AlignmentType.CENTER,
  children: [new TextRun({children: [PageNumber.CURRENT], font: FONT, size: SIZE})]})]});

module.exports = {d, P, H1, H2, item, cap, table, gap, footer, run, mm, FONT, SIZE};
