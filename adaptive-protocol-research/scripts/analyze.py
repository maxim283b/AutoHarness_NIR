from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.storage.jsonl import read_jsonl


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, center-half), min(1, center+half)


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf8"))
    rows = read_jsonl(args.run / "traces.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["arm"])].append(row)
    summary = {}
    lines = [f"# {'Mock: проверка архитектуры' if config['mode'] == 'mock' else 'Реальный LLM-пилот'}", "", "Метод | Успехи / n (95% Wilson) | Токены | Вызовы | Ошибки | Проверки | Задержка, с | Переключения | Токены контроллера", "---|---:|---:|---:|---:|---:|---:|---:|---:"]
    for (split, arm), subset in sorted(groups.items()):
        n = len(subset)
        k = sum(bool(r["success"]) for r in subset)
        lo, hi = wilson(k, n)
        vals = {"split": split, "arm": arm, "n": n, "successes": k, "success_rate": k/n, "wilson_95": [lo, hi], "mean_tokens": mean(subset,"tokens"), "mean_calls": mean(subset,"calls"), "mean_errors": mean(subset,"errors"), "mean_tool_calls": mean(subset,"tool_calls"), "mean_wall_s": mean(subset,"wall_s"), "mean_switches": mean(subset,"switch_count"), "mean_controller_tokens": mean(subset,"controller_tokens")}
        summary[f"{split}/{arm}"] = vals
        lines.append(f"{split}/{arm} | {k}/{n} ({lo:.2f}–{hi:.2f}) | {vals['mean_tokens']:.0f} | {vals['mean_calls']:.1f} | {vals['mean_errors']:.1f} | {vals['mean_tool_calls']:.1f} | {vals['mean_wall_s']:.2f} | {vals['mean_switches']:.2f} | {vals['mean_controller_tokens']:.0f}")
    best = config["best_fixed_train"]
    lines += ["", f"Лучший фиксированный протокол по train: **{best}**.", ""]
    by_arm = {arm: {r["task_id"]: r for r in rows if r["split"] == "test" and r["arm"] == arm} for arm in {r["arm"] for r in rows if r["split"] == "test"}}
    baseline = by_arm[best]
    lines += ["## Парные расхождения с train-selected fixed", "", "Метод | Улучшено | Ухудшено | Без изменения", "---|---:|---:|---:"]
    for arm in ["rule", "llm_controller", "bandit_trained", "bandit_cold", "random", "rule_front_only"]:
        if arm not in by_arm:
            continue
        better = worse = same = 0
        for task_id, row in by_arm[arm].items():
            delta = int(row["success"]) - int(baseline[task_id]["success"])
            better += delta > 0
            worse += delta < 0
            same += delta == 0
        lines.append(f"{arm} | {better} | {worse} | {same}")
    if "rule" in by_arm and "rule_front_only" in by_arm:
        full, front = by_arm["rule"], by_arm["rule_front_only"]
        helped = sum(full[t]["success"] and not front[t]["success"] for t in full)
        harmed = sum(front[t]["success"] and not full[t]["success"] for t in full)
        lines += ["", f"Rule с переключениями против выбора только в начале: улучшено {helped}/{len(full)}, ухудшено {harmed}/{len(full)}. Это парное сравнение запусков, а не идентификация эффекта каждого отдельного switch."]
    switch_examples = [r for r in rows if r["split"] == "test" and r["switch_count"]]
    lines += ["", "## Примеры переключений", ""]
    for row in switch_examples[:5]:
        chain = " → ".join([row["initial_protocol"]] + [s["current"] for s in row["switches"]])
        lines.append(f"- Задача {row['task_id']}, {row['arm']}: {chain}; успех={row['success']}; причина первого переключения: {row['switches'][0]['reason']}.")
    if config["mode"] == "mock":
        lines += ["", "**Эти значения получены на scripted mock. Они проверяют трассировку и подсчёт метрик, но не измеряют качество LLM или полезность выбора протокола.**"]
    (args.run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    (args.run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [arm for (split, arm) in sorted(groups) if split == "test"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(names, [summary[f"test/{arm}"]["success_rate"] for arm in names])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Success rate")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(args.run / "success_rate.png", dpi=140)
        plt.close(fig)
    except ImportError:
        pass
    print(args.run / "summary.md")


if __name__ == "__main__":
    main()
