"""Hourly local campaign monitor and restart launcher."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/autosaddler_reproduction/improved_v2_campaign"
sys.path.insert(0, str(ROOT))
from improved_v2.account_selection import availability, choose


def docker_ready(env):
    command = ["/usr/local/bin/docker", "info", "--format", "{{.ServerVersion}}"]
    def healthy():
        try:
            return subprocess.run(command, env=env, capture_output=True,
                                  timeout=12).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
    if healthy():
        return True
    subprocess.run(["/usr/bin/open", "-a", "Docker"], capture_output=True, timeout=15)
    for _ in range(12):
        time.sleep(5)
        if healthy():
            return True
    return False


def active():
    rows = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True).stdout
    return any(("improved_v2/supervise.py" in row or "improved_v2/recovery_supervise.py" in row
                or "improved_v2/run.py run" in row or "improved_v2/resume_repaired.py" in row)
               and not row.strip().startswith(str(os.getpid()) + " ") for row in rows.splitlines())


def report(event, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    runs = {}
    for label in ("H1", "H2"):
        path = ROOT / "results/autosaddler_reproduction/improved_v2_runs" / (
            "swe-improved-v2-" + label.lower() + "-20261112") / "events.jsonl"
        iterations = set()
        if path.exists():
            for line in path.open():
                item = json.loads(line)
                if item.get("event_type") == "IterationCompleted":
                    iterations.add(item.get("payload", {}).get("iteration"))
        runs[label] = {"completed_iterations": len(iterations),
                       "optimization_complete": (OUT / label / "complete.json").exists()}
    final = OUT / "final/results"
    alternative = OUT / "final-alternative/results/H2_alternative"
    gateways = {}
    for label in ("H1", "H2", "final"):
        path = OUT / label / "gateway/status.json"
        if path.exists():
            value = json.loads(path.read_text())
            gateways[label] = {key: value.get(key) for key in ("requests", "allowance", "halted")}
    free_gib = round(shutil.disk_usage(ROOT).free / 1024**3, 2)
    row = {"at_utc": datetime.now(timezone.utc).isoformat(), "event": event,
           "status": status.get("status"), "stage": status.get("stage"),
           "error_type": status.get("error_type"), "message": status.get("message"),
           "runs": runs, "gateways": gateways, "disk_free_gib": free_gib,
           "final_results": len(list(final.rglob("result.json"))) if final.exists() else 0,
           "final_alternative_results": len(list(alternative.rglob("result.json"))) if alternative.exists() else 0,
           **fields}
    with (OUT / "hourly-monitor.jsonl").open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if (OUT / "operator-pause.json").exists():
        report("owner_paused")
        return 0
    status_path = OUT / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
        report("complete")
        if (OUT / "final-alternative/plan.json").exists():
            if not (OUT / "final-alternative/analysis.json").exists():
                from improved_v2.final_alternative import analyze
                analyze()
            return 0
        if not (OUT / "analysis-full40.json").exists():
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT) + ":" + str(ROOT.parent / "third_party/AutoSaddler/src")
            subprocess.run([sys.executable, str(HERE / "analyze.py")], cwd=ROOT,
                           env=env, check=True, stdout=(OUT / "analysis.log").open("w"))
        return 0
    if active():
        report("active")
        return 0
    log = OUT / "supervisor.jsonl"
    if log.exists():
        last = json.loads(log.read_text().splitlines()[-1])
        if last.get("event") == "needs_reconciliation" and not any(
                word in last.get("reason", "").lower() for word in ("quota", "balance")):
            report("needs_reconciliation")
            return 2
    try:
        accounts = availability()
    except Exception as error:
        report("account_check_failed", error_type=type(error).__name__)
        return 2
    selected = choose(accounts)
    if selected is None:
        report("waiting_for_go_quota", accounts=accounts)
        return 0
    env = dict(os.environ)
    env["PATH"] = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT) + ":" + str(ROOT.parent / "third_party/AutoSaddler/src")
    env["AUTOSADDLER_OPENCODE_ORG"] = selected
    if not docker_ready(env):
        report("waiting_for_docker", selected_workspace=selected, accounts=accounts)
        return 2
    with (OUT / "account-selection.jsonl").open("a") as stream:
        stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                 "workspace": selected, "reason": "first available Go quota"}) + "\n")
    with (OUT / "monitor-launch.log").open("a") as output:
        child = subprocess.Popen(["/usr/bin/caffeinate", "-dimsu", sys.executable,
                                  str(HERE / "recovery_supervise.py")], cwd=ROOT, env=env,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    report("launched", pid=child.pid, selected_workspace=selected, accounts=accounts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
