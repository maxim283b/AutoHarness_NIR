"""Collect saved ACP checks and reconcile initially unparsed native counters.

This command never calls a model. Failed attempts remain in the report and their
tokens count toward the total; cached tokens are included exactly once.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .acp_checks import normalize_acp

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/autosaddler_reproduction/acp_checks"


def collect(prefix):
    directories = sorted(path for path in BASE.iterdir() if path.is_dir() and path.name.startswith(prefix))
    latest, attempts, sessions = {}, [], {}
    for directory in directories:
        for action in ("protocol", "gaia", "swe", "terminal"):
            path = directory / action / "result.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                attempts.append({"run_id": directory.name, "action": action, "passed": result.get("passed"), "error": result.get("error"), "path": str(path.relative_to(ROOT))})
                latest[action] = {**result, "path": str(path.relative_to(ROOT))}
        for path in directory.rglob("prompt_result.json"):
            native = json.loads(path.read_text(encoding="utf-8"))
            usage = normalize_acp(native)
            capabilities = json.loads(path.with_name("capabilities.json").read_text(encoding="utf-8"))
            session_id = capabilities["session"]["sessionId"]
            if session_id in sessions:
                raise ValueError("Duplicate ACP session usage source")
            sessions[session_id] = {"run_id": directory.name, "usage": usage, "source": str(path.relative_to(ROOT))}
            # Repair only missing counters, with a durable before/after audit.
            ledger = path.parent.parent / "usage.sqlite"
            if ledger.exists() and usage["usage_complete"]:
                connection = sqlite3.connect(ledger)
                try:
                    request_id = "acp:" + session_id + ":turn:1"
                    row = connection.execute("SELECT payload FROM calls WHERE request_id=?", (request_id,)).fetchone()
                    previous = json.loads(row[0]) if row else None
                    if previous is not None and not previous.get("usage_complete"):
                        replacement = {**previous, **usage}
                        audit = {"reason": "Reparse saved ACP turn: inputTokens excludes separately reported cache tokens", "source": str(path.relative_to(ROOT)), "previous": previous, "replacement": replacement}
                        path.with_name("usage_reconciliation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
                        with connection:
                            connection.execute("UPDATE calls SET payload=? WHERE request_id=? AND payload=?", (json.dumps(replacement, sort_keys=True, ensure_ascii=False), request_id, row[0]))
                finally:
                    connection.close()
    fields = ("input_tokens", "uncached_input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "cache_write_tokens", "reasoning_tokens")
    totals = {field: sum(s["usage"].get(field, 0) or 0 for s in sessions.values()) for field in fields}
    complete = all(s["usage"]["usage_complete"] for s in sessions.values())
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "agent": "codex-acp", "model": "gpt-6-luna",
        "scope": "Bounded infrastructure checks; GAIA2 tool subset and one training task each for SWE Pro V1 / TB2",
        "is_opencode_h0_h1_result": False, "optimizer_run_performed": False, "held_out_test_used": False,
        "all_latest_checks_passed": set(latest) == {"protocol", "gaia", "swe", "terminal"} and all(r.get("passed") for r in latest.values()),
        "latest_checks": latest, "attempts": attempts, "completed_acp_turns": len(sessions), "sessions": sessions,
        "usage_all_completed_attempts": {**totals, "usage_complete": complete, "includes_failed_attempts": True},
        "usage_granularity": "Native ACP turn aggregates, not request-level OpenCode accounting",
        "billing": "Existing signed-in Codex allowance; API credits not used; no USD price inferred",
        "limitations": ["One training task per scored benchmark cannot establish accuracy or improvement",
            "GAIA2 check covers read/write/AUI on train apps, not the full original objective, event lifecycle or judge",
            "Separate OpenCode fixture/event and AutoSaddler resume checks provide other infrastructure evidence"]}
    BASE.mkdir(parents=True, exist_ok=True)
    target = BASE / (prefix + "-report.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Проверки через Codex ACP", "", "Модель: `gpt-6-luna`, reasoning low. Использован обычный лимит Codex.", "",
        "Это технические проверки с агентом Codex, а не сравнение OpenCode H0/H1. Test не использован; оптимизация не запускалась.", "",
        "| Проверка | Результат | Входные токены, включая кеш | Выходные | Всего |", "|---|---|---:|---:|---:|"]
    labels = {"protocol": "ACP / структурированный ответ", "gaia": "GAIA2 / инструменты и AUI", "swe": "SWE-Bench Pro V1 / одна train-задача", "terminal": "Terminal-Bench 2.0 / одна train-задача"}
    for action, label in labels.items():
        result = latest.get(action, {})
        usage = result.get("usage", {})
        outcome = "пройдено" if result.get("passed") else "ошибка"
        if action == "swe" and result.get("grading"):
            grade = result["grading"]
            outcome += f"; {grade['passed_required_tests']}/{grade['required_tests']} обязательных тестов"
        if action == "terminal" and "score" in result:
            outcome += f"; reward {result['score']}"
        lines.append(f"| {label} | {outcome} | {usage.get('input_tokens', '—')} | {usage.get('output_tokens', '—')} | {usage.get('total_tokens', '—')} |")
    lines += ["", f"Все {len(sessions)} завершённых ACP-вызовов, включая неудачные попытки: **{totals['total_tokens']} токенов**.",
        f"Из них некешированный вход: {totals['uncached_input_tokens']}; кешированный вход: {totals['cached_input_tokens']}; выход: {totals['output_tokens']}.",
        "Reasoning входит в output и повторно не прибавляется. Отдельная ошибка выбора формата модели произошла до отправки промпта.", "",
        "Исправлены имена календарных инструментов, согласование разрешений MCP и обработка отдельных cache-счётчиков ACP. Неудачные попытки и исходные ответы сохранены.", "",
        "Данные не дают оценки качества на полном бенчмарке и не подтверждают эффект AutoSaddler.", ""]
    BASE.joinpath(prefix + "-report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(target), "all_latest_checks_passed": report["all_latest_checks_passed"], "completed_acp_turns": len(sessions), "usage": report["usage_all_completed_attempts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if not args.prefix.replace("-", "").replace("_", "").isalnum():
        parser.error("Invalid run prefix")
    print(json.dumps(collect(args.prefix), indent=2))


if __name__ == "__main__":
    main()
