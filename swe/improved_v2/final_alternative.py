"""Operator amendment: finish H0/H1 and test the tied H2 candidate separately.

The frozen optimization, its selected H2 seed, and completed original H2
test artifacts remain intact. The alternative H2 test is exploratory because
the owner changed the final arm after seeing development results.
"""
from __future__ import annotations

import hashlib
import json
import csv
import math
import random
import shutil
import sqlite3
from pathlib import Path

from autosaddler.v2.core.domain import Case
from autosaddler.v2.core.run_state import RunState
from experiments.autosaddler_reproduction import runner
from experiments.autosaddler_reproduction.opencode_runtime import candidate_hash
from improved_v2 import run


ALT_ID = "sha256:fa214efe0698bf2ee2c953fe30b2aa5b15098c9f29912805f252f7d39370b64e"
ALT_RUN_ID = "swe-improved-v2-h2-tied-final-20261117"
ALT_ROOT = run.OUT / "final-alternative"


def _prepared():
    h1_runtime, h1 = run.candidate("H1")
    h2_runtime, selected_h2 = run.candidate("H2")
    state = RunState.replay(h2_runtime.store.events())
    if ALT_ID not in state.accepted_candidate_ids or ALT_ID == selected_h2.candidate_id:
        raise RuntimeError("Alternative H2 provenance drift")
    dev = [evaluation for evaluation in state.evaluations.values()
           if evaluation.candidate_id == ALT_ID and evaluation.split == "development"]
    if len(dev) != 1 or dev[0].aggregate_score != 0.25 or len(dev[0].observations) != 16:
        raise RuntimeError("Alternative H2 development evidence drift")
    baseline = h1_runtime.scenario.harness_space.seed()
    if baseline.candidate_id != selected_h2.candidate_id:
        raise RuntimeError("Original H2 selection was not the H0 seed")
    return h1_runtime, h2_runtime, {"H0": baseline, "H1": h1,
                                    "H2": state.candidates[ALT_ID]}, selected_h2


def _plans(candidates, selected_h2):
    original_path = run.OUT / "final/plan.json"
    original = json.loads(original_path.read_text())
    manifest = json.loads((run.SPEC / "manifest.json").read_text())
    order = sorted(manifest["test"], key=lambda task: hashlib.sha256(
        f"20261112:swe_bench_pro:test:{task}".encode()).hexdigest())
    expected_schedule = []
    for task in order:
        labels = sorted(("H0", "H1", "H2"), key=lambda label: hashlib.sha256(
            f"20261112:final:{task}:{label}".encode()).hexdigest())
        expected_schedule.extend([task, label, labels[0]] for label in labels)
    if (original.get("run_id") != run.FINAL_ID or original.get("test") != order
            or original.get("schedule") != expected_schedule
            or original.get("candidate_ids") != {"H0": candidates["H0"].candidate_id,
                                                     "H1": candidates["H1"].candidate_id,
                                                     "H2": selected_h2.candidate_id}
            or original.get("manifest_sha256") != run.sha(run.SPEC / "manifest.json")):
        raise RuntimeError("Original final plan provenance drift")
    plan = {"kind": "owner_requested_exploratory_H2_alternative",
            "original_plan_sha256": run.sha(original_path),
            "original_run_id": run.FINAL_ID, "alternative_run_id": ALT_RUN_ID,
            "test": order, "schedule": expected_schedule,
            "candidate_ids": {"H0": candidates["H0"].candidate_id,
                              "H1": candidates["H1"].candidate_id,
                              "H2_original": selected_h2.candidate_id,
                              "H2_alternative": ALT_ID},
            "development_score": 0.25,
            "provenance_note": "H2 alternative tied H0 on development (4/16). The owner changed the final arm after development scores and three original H2 test results were available. Original H2 selection and those results remain archived in final/results/H2. Alternative is exploratory, not the frozen selector's H2 winner."}
    path = ALT_ROOT / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError("Alternative final plan drift")
    run.save(path, plan)
    return expected_schedule


def _recoverable_grading_attempt(path, expected):
    attempt = json.loads((path / "attempt.json").read_text())
    if any(attempt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Interrupted final attempt provenance drift: {path}")
    generation = path / "native_swe/generation"
    status_path = generation / "generation_status.json"
    patch_path = generation / "patch.diff"
    if not status_path.exists() or not patch_path.exists():
        raise RuntimeError(f"Interrupted final attempt needs manual reconciliation: {path}")
    status = json.loads(status_path.read_text())
    if (status.get("status") != "complete"
            or status.get("agent_cleanup_verified") is not True
            or status.get("generation_container_removed_before_grading") is not True
            or hashlib.sha256(patch_path.read_bytes()).hexdigest() != status.get("patch_sha256")):
        raise RuntimeError(f"Incomplete generation cannot be reused: {path}")
    prior = path / "native_swe/grading/execution.json"
    if prior.exists():
        execution = json.loads(prior.read_text())
        if (execution.get("cleanup_verified") is not True
                or any(item.get("running") for item in execution.get("executions", []))):
            raise RuntimeError(f"Unclean prior grading cannot be retried: {path}")
    recovery_path = path / "grading-recovery.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {"retries": 0}
    if recovery["retries"] >= 2:
        raise RuntimeError(f"Grader remained technically invalid after two retries: {path}")
    run.save(recovery_path, {"retries": recovery["retries"] + 1,
                             "reason": "Reuse verified completed generation; rerun only official grading"})
    return status


def _recoverable_failed_generation(path, expected):
    import docker

    attempt = json.loads((path / "attempt.json").read_text())
    if any(attempt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Interrupted generation provenance drift: {path}")
    generation = path / "native_swe/generation"
    status_path = generation / "generation_status.json"
    if not status_path.exists():
        raise RuntimeError(f"Generation status absent: {path}")
    status = json.loads(status_path.read_text())
    if status.get("status") != "agent_running" or status.get("task_id") != expected["task_id"]:
        raise RuntimeError(f"Unexpected interrupted generation state: {path}")
    trace_path = generation / "opencode.jsonl"
    if not trace_path.exists():
        raise RuntimeError(f"Missing failed generation trace: {path}")
    errors = [json.loads(line).get("error") for line in trace_path.read_text().splitlines()
              if '"type":"error"' in line]
    if not errors:
        raise RuntimeError(f"No recorded model error in failed generation: {path}")
    client = docker.from_env()
    try:
        try:
            client.containers.get(status["container"])
        except docker.errors.NotFound:
            pass
        else:
            raise RuntimeError(f"Prior generation container still exists: {path}")
    finally:
        client.close()
    journal = run.OUT / "final/gateway/requests.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    rows = [row for row in rows if row.get("run_id") == expected["run_id"]
            and row.get("task_id") == expected["task_id"]]
    if not rows:
        raise RuntimeError(f"No recorded failed generation requests: {path}")
    last = rows[-1]
    unsettled = ({r["id"] for r in rows if r.get("status") == "reserved"}
                 - {r["id"] for r in rows if r.get("status") != "reserved"})
    error_text = str(errors[-1])
    if (last.get("status") == "http_error" and last.get("http_status") == 429
            and not unsettled and "429" in error_text):
        reason = "Upstream HTTP 429; all requests settled"
        ambiguous = None
    elif (last.get("status") == "reserved" and unsettled == {last["id"]}
          and last.get("error") == "RemoteProtocolError: Server disconnected without sending a response."
          and "Server disconnected without sending a response" in error_text):
        evidence = last.get("response_evidence") or {}
        body_path = Path(evidence.get("body", ""))
        metadata_path = Path(evidence.get("metadata", ""))
        if not body_path.is_file() or body_path.stat().st_size != 0 or not metadata_path.is_file():
            raise RuntimeError(f"Ambiguous request has response bytes: {path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("request_id") != last["id"] or metadata.get("http_status") is not None \
                or metadata.get("body_bytes") != 0:
            raise RuntimeError(f"Ambiguous request evidence mismatch: {path}")
        reason = "Transport disconnected before response; prior reservation retained as unknown usage"
        ambiguous = {"request_id": last["id"], "reservation": last["reservation"]}
    else:
        raise RuntimeError(f"Unrecognized or unsettled failed generation requests: {path}")
    recovery_path = path / "generation-recovery.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {"retries": 0}
    if recovery["retries"] >= 3:
        raise RuntimeError(f"Generation failed after three technical retries: {path}")
    backup = path / "technical-retries" / f"generation-{recovery['retries'] + 1}"
    if backup.exists():
        raise RuntimeError(f"Generation recovery archive already exists: {backup}")
    shutil.copytree(generation, backup)
    run.save(recovery_path, {"retries": recovery["retries"] + 1,
                             "reason": reason, "ambiguous_reservation": ambiguous})


def _recoverable_owner_pause(path, expected):
    import docker

    pause = json.loads((path / "paused-attempt.json").read_text())
    if any(pause.get(key) != expected[key] for key in ("task_id", "run_id", "candidate_id")):
        raise RuntimeError(f"Paused attempt provenance drift: {path}")
    generation = path / "native_swe/generation"
    status = json.loads((generation / "generation_status.json").read_text())
    if status.get("status") != "agent_running" or status.get("container") != pause.get("container_id"):
        raise RuntimeError(f"Paused generation status drift: {path}")
    rows = [json.loads(line) for line in (run.OUT / "final/gateway/requests.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("run_id") == expected["run_id"]
            and row.get("task_id") == expected["task_id"]]
    unsettled = ({row["id"] for row in rows if row.get("status") == "reserved"}
                 - {row["id"] for row in rows if row.get("status") != "reserved"})
    if unsettled != set(pause.get("unsettled_request_ids", [])):
        raise RuntimeError(f"Paused request accounting drift: {path}")
    client = docker.from_env()
    try:
        try:
            container = client.containers.get(pause["container_id"])
        except docker.errors.NotFound:
            container = None
        if container is not None:
            container.reload()
            if (container.status != "exited"
                    or container.labels.get("autoharness.swe-run") != expected["run_id"]):
                raise RuntimeError(f"Paused container is not safely stopped: {path}")
    finally:
        client.close()
    backup = path / "technical-retries/owner-paused-generation"
    if backup.exists():
        raise RuntimeError(f"Paused generation archive already exists: {backup}")
    shutil.copytree(generation, backup)
    run.save(path / "generation-recovery.json", {
        "retries": 1, "reason": "Owner paused during generation; restart this task after resume",
        "unsettled_request_ids": sorted(unsettled), "completed_request_count": pause["completed_request_count"],
    })
    run.save(path / "paused-attempt-recovered.json", {
        "paused_attempt": "paused-attempt.json", "generation_archive": str(backup.relative_to(path)),
    })
    if container is not None:
        container.remove(force=True)


def _restore_recorded_usage(result, expected, *, force=False):
    role = expected["role"]
    if not force and result.get("usage", {}).get(role, {}).get("calls", 0):
        return
    ledger = run.ROOT / "results/autosaddler_reproduction/usage" / expected["run_id"] / "requests.sqlite"
    with sqlite3.connect(ledger) as db:
        rows = db.execute("SELECT payload FROM calls WHERE run_id=? AND benchmark=? AND task_id=? AND role=?",
                          (expected["run_id"], expected["benchmark"], expected["task_id"], role)).fetchall()
    if not rows:
        raise RuntimeError("Completed generation lacks recorded model usage")
    usages = [json.loads(row[0]) for row in rows]
    if any(item.get("usage_complete") is not True for item in usages):
        raise RuntimeError("Recovered generation has incomplete model usage")
    result.setdefault("usage", {})[role] = {
        "calls": len(usages), "usage_complete": True,
        **{field: sum(item.get(field) or 0 for item in usages)
           for field in ("input_tokens", "output_tokens", "total_tokens")},
    }


def final():
    run.verify()
    h1_runtime, h2_runtime, candidates, selected_h2 = _prepared()
    schedule = _plans(candidates, selected_h2)
    with run.gateway(run.OUT / "final/gateway") as go:
        try:
            for index, (task, slot, first) in enumerate(schedule):
                label = "H2_alternative" if slot == "H2" else slot
                candidate = candidates[slot]
                runtime = h2_runtime if slot == "H2" else h1_runtime
                role = "target_final_baseline" if slot == "H0" else "target_final_optimized"
                case_root = ALT_ROOT / "results/H2_alternative" if slot == "H2" else run.OUT / "final/results" / slot
                path = case_root / hashlib.sha256(task.encode()).hexdigest()[:16]
                result_path = path / "result.json"
                run_id = ALT_RUN_ID if slot == "H2" else run.FINAL_ID
                expected = {"run_id": run_id, "label": label, "benchmark": "swe_bench_pro",
                            "task_id": task, "candidate_id": candidate.candidate_id,
                            "model": "opencode-go/qwen3.7-plus", "execution_index": index,
                            "planned_first_label": first, "role": role}
                preflight = run.ROOT / "results/autosaddler_reproduction/fullsource_image_preflight" / (
                    hashlib.sha256(task.encode()).hexdigest()[:20] + ".json")
                certificate = json.loads(preflight.read_text())
                if certificate.get("task_id") != task or certificate.get("passed") is not True:
                    raise RuntimeError("Task image certificate drift")
                expected["grading_image_id"] = certificate["image_id"]
                materialized = runtime.scenario.harness_space.materialize(candidate, "evaluate")
                try:
                    expected["candidate_hash"] = candidate_hash(materialized.root)
                    if result_path.exists():
                        if not run.result_valid(json.loads(result_path.read_text()), expected, role):
                            raise RuntimeError(f"Incomplete or mismatched final result: {result_path}")
                        continue
                    recovery_kind = None
                    if (path / "attempt.json").exists():
                        status_path = path / "native_swe/generation/generation_status.json"
                        status = json.loads(status_path.read_text()) if status_path.exists() else {}
                        if status.get("status") == "complete":
                            _recoverable_grading_attempt(path, expected)
                            recovery_kind = "grade"
                        elif status.get("status") == "agent_running":
                            if ((path / "paused-attempt.json").exists()
                                    and not (path / "paused-attempt-recovered.json").exists()):
                                _recoverable_owner_pause(path, expected)
                            elif ((path / "paused-attempt-recovered.json").exists()
                                  and status.get("container") == json.loads(
                                      (path / "paused-attempt.json").read_text()).get("container_id")
                                  and json.loads((path / "generation-recovery.json").read_text()).get(
                                      "reason") == "Owner paused during generation; restart this task after resume"):
                                # An offline image import can fail before the
                                # replacement generation starts. The archived
                                # paused attempt remains the only attempt.
                                pass
                            else:
                                _recoverable_failed_generation(path, expected)
                            recovery_kind = "generation_retry"
                        else:
                            raise RuntimeError(f"Interrupted final attempt needs reconciliation: {path}")
                    else:
                        run.save(path / "attempt.json", {**expected, "started_at_utc": run.now()})
                    case = Case("swe_bench_pro:" + task, "test",
                                {"benchmark": "swe_bench_pro", "task_id": task})
                    if recovery_kind == "grade":
                        from experiments.autosaddler_reproduction import swe_go_adapter
                        original_generate = swe_go_adapter.generate

                        def recorded_generate(row, candidate_root, model, artifacts, request_run_id,
                                              request_role, image, **kwargs):
                            saved = json.loads((Path(artifacts) / "generation_status.json").read_text())
                            patch = (Path(artifacts) / "patch.diff").read_text()
                            if (saved.get("status") != "complete" or row["instance_id"] != task
                                    or request_run_id != run_id or request_role != role
                                    or hashlib.sha256(patch.encode()).hexdigest() != saved.get("patch_sha256")):
                                raise RuntimeError("Recorded generation provenance drift")
                            return patch, {key: value for key, value in saved.items() if key != "status"}

                        swe_go_adapter.generate = recorded_generate
                        try:
                            result = runner.run_case(case, materialized.root, path,
                                                     runtime.config.scenario.settings, run_id, role)
                        finally:
                            swe_go_adapter.generate = original_generate
                        _restore_recorded_usage(result, expected)
                    else:
                        result = runner.run_case(case, materialized.root, path,
                                                 runtime.config.scenario.settings, run_id, role)
                        if recovery_kind == "generation_retry":
                            _restore_recorded_usage(result, expected, force=True)
                finally:
                    materialized.release()
                result.update(expected)
                result["completed_at_utc"] = run.now()
                if not run.result_valid(result, expected, role):
                    run.save(path / "invalid-result.json", result)
                    raise RuntimeError(f"Official evaluation incomplete: {task} {label}")
                run.save(result_path, result)
        finally:
            run.save(run.OUT / "final/accounting.json", go.accounting_summary())


def analyze():
    run.verify()
    plan = json.loads((ALT_ROOT / "plan.json").read_text())
    original = json.loads((run.OUT / "final/plan.json").read_text())
    if plan["original_plan_sha256"] != run.sha(run.OUT / "final/plan.json"):
        raise RuntimeError("Original final plan changed after amendment")
    manifest = json.loads((run.SPEC / "manifest.json").read_text())
    if set(plan["test"]) != set(manifest["test"]) or len(plan["test"]) != 40:
        raise RuntimeError("Alternative test manifest drift")
    if plan["schedule"] != original["schedule"]:
        raise RuntimeError("Alternative test schedule drift")
    candidates = plan["candidate_ids"]
    index = {(task, label): (i, first)
             for i, (task, label, first) in enumerate(plan["schedule"])}
    rows = {}
    tokens = {}
    for slot in ("H0", "H1", "H2_alternative"):
        root = ALT_ROOT / "results/H2_alternative" if slot == "H2_alternative" else run.OUT / "final/results" / slot
        role = "target_final_baseline" if slot == "H0" else "target_final_optimized"
        original_slot = "H2" if slot == "H2_alternative" else slot
        run_id = ALT_RUN_ID if slot == "H2_alternative" else run.FINAL_ID
        values = []
        hashes = set()
        for task in plan["test"]:
            path = root / hashlib.sha256(task.encode()).hexdigest()[:16]
            result = json.loads((path / "result.json").read_text())
            attempt = json.loads((path / "attempt.json").read_text())
            certificate_path = (run.ROOT / "results/autosaddler_reproduction/fullsource_image_preflight"
                                / (hashlib.sha256(task.encode()).hexdigest()[:20] + ".json"))
            certificate = json.loads(certificate_path.read_text())
            if certificate.get("task_id") != task or certificate.get("passed") is not True:
                raise RuntimeError(f"Invalid task image certificate: {task}")
            execution_index, first = index[(task, original_slot)]
            expected = {"run_id": run_id, "label": slot, "benchmark": "swe_bench_pro",
                        "task_id": task, "candidate_id": candidates[slot],
                        "model": "opencode-go/qwen3.7-plus", "execution_index": execution_index,
                        "planned_first_label": first, "role": role,
                        "candidate_hash": attempt.get("candidate_hash"),
                        "grading_image_id": certificate["image_id"]}
            if (any(attempt.get(key) != value for key, value in expected.items())
                    or not run.result_valid(result, expected, role)):
                raise RuntimeError(f"Invalid official final result: {path}")
            hashes.add(result["candidate_hash"])
            values.append(result)
        if len(list(root.rglob("result.json"))) != 40 or len(hashes) != 1:
            raise RuntimeError(f"Incomplete or mixed candidate final arm: {slot}")
        rows[slot] = {v["task_id"]: int(v["score"]) for v in values}
        buckets = [v["usage"][role] for v in values]
        tokens[slot] = {field: sum(bucket[field] for bucket in buckets)
                        for field in ("calls", "input_tokens", "output_tokens", "total_tokens")}
    def paired(a, b):
        left, right = rows[a], rows[b]
        better = sum(left[t] == 0 and right[t] == 1 for t in plan["test"])
        worse = sum(left[t] == 1 and right[t] == 0 for t in plan["test"])
        discordant = better + worse
        tail = (sum(math.comb(discordant, k) for k in range(min(better, worse) + 1))
                / 2 ** discordant) if discordant else 0.5
        rng = random.Random(20261112)
        deltas = [right[t] - left[t] for t in plan["test"]]
        bootstrap = sorted(100 * sum(deltas[rng.randrange(40)] for _ in range(40)) / 40
                           for _ in range(20000))
        return {"delta_pp": 100 * (better - worse) / 40,
                "better": better, "worse": worse,
                "mcnemar_exact_two_sided_p": min(1.0, 2 * tail),
                "bootstrap_95_pp": [bootstrap[500], bootstrap[19500]]}
    result = {"kind": "exploratory_H2_alternative", "tasks": 40,
              "candidate_ids": candidates,
              "scores": {label: sum(values.values()) for label, values in rows.items()},
              "pass_at_1": {label: sum(values.values()) / 40 for label, values in rows.items()},
              "usage": tokens,
              "paired": {"H1_minus_H0": paired("H0", "H1"),
                         "H2_alternative_minus_H0": paired("H0", "H2_alternative"),
                         "H2_alternative_minus_H1": paired("H1", "H2_alternative")},
              "note": plan["provenance_note"],
              "limits": "One optimization run per arm; H2 alternative chosen after seeing Dev and three original H2 test outcomes. Confidence intervals describe the sampled tasks and do not establish a general improvement."}
    run.save(ALT_ROOT / "analysis.json", result)
    with (ALT_ROOT / "paired-40.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task_id", "H0", "H1", "H2_alternative"))
        for task in plan["test"]:
            writer.writerow((task, rows["H0"][task], rows["H1"][task],
                             rows["H2_alternative"][task]))
    return result
