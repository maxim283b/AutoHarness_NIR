"""MBPPHarnessAdapter: GEPAAdapter for the two-component harness.

reflection_mode:
  - "standard"     (baseline arm): one record per minibatch example, as specified in pilot_selection.md §3.4.
  - "contrastive"  (the single modification): the same minibatch records, labelled SUCCEEDED / FAILED, and each
                   FAILED record is paired with a successful trace taken from the archive of traces that this run
                   has already produced on the train split (same task by another prompt variant if available,
                   otherwise another train task). No extra rollouts are made; only the teacher's input changes.

Leakage guard: feedback shows only the visible (first) assert and its error; hidden asserts are never shown.
"""

from gepa.core.adapter import EvaluationBatch

from harness import TrajectoryLog, repair_user_msg, run_task, sha, solve_user_msg

MAX_CODE_CHARS = 1500
MAX_INSTR_CHARS = 800
MAX_CONTRAST_PAIRS = 2


def _clip(s: str | None, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


class MBPPHarnessAdapter:
    propose_new_texts = None  # use GEPA's default instruction proposer (same in both arms)

    def __init__(self, client, log_path, ctx: dict, train_ids: set, reflection_mode: str = "standard"):
        assert reflection_mode in ("standard", "contrastive")
        self.client = client
        self.log = TrajectoryLog(log_path)
        self.ctx = ctx
        self.train_ids = set(train_ids)
        self.mode = reflection_mode
        self.archive: dict[int, list[dict]] = {}  # task_id -> trajectories already produced (train only)
        self.candidates: dict[str, dict] = {}  # candidate_hash -> candidate
        self.contrast_log: list[dict] = []

    # ---------------- evaluate ----------------
    def evaluate(self, batch, candidate, capture_traces=False):
        self.candidates[sha(candidate)] = dict(candidate)
        outputs, scores, trajs, objs = [], [], [], []
        for task in batch:
            try:
                t = run_task(task, candidate, self.client, self.log, self.ctx)
            except RuntimeError:
                raise  # systemic (model server down)
            except Exception as e:  # per-example failure -> 0.0
                t = {"task_id": task["task_id"], "task": task, "candidate_hash": sha(candidate), "steps": [],
                     "repair_used": False, "final_code": "", "final_pass": False, "n_asserts_passed": 0,
                     "tokens": 0, "error": repr(e)}
            if task["task_id"] in self.train_ids:
                self.archive.setdefault(task["task_id"], []).append(t)
            outputs.append({"task_id": t["task_id"], "final_code": t["final_code"], "final_pass": t["final_pass"]})
            scores.append(1.0 if t["final_pass"] else 0.0)
            trajs.append(t)
            objs.append({"pass": float(t["final_pass"]), "repair_used": float(t["repair_used"]),
                         "tokens": float(t["tokens"])})
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs if capture_traces else None,
                               objective_scores=objs)

    # ---------------- reflective dataset ----------------
    @staticmethod
    def _feedback(t: dict) -> str:
        n = len(t["task"]["test_list"])
        if t["final_pass"]:
            return f"PASS: the final code passed all {n} tests."
        s = t["steps"][-1] if t["steps"] else {}
        vis = "passed" if s.get("visible_test_ok") else f"failed with: {s.get('visible_error')}"
        return (f"FAIL: the final code passed {t['n_asserts_passed']} of {n} tests (the other tests are hidden). "
                f"Visible test {vis}.")

    def _record(self, t: dict, comp: str) -> dict | None:
        task, steps = t["task"], t["steps"]
        if not steps:
            return {"Inputs": solve_user_msg(task), "Generated code": "", "Visible test result": "n/a",
                    "Repair attempt": "n/a", "Feedback": f"FAIL: harness error {t.get('error')}"}
        s1 = steps[0]
        vis1 = "passed" if s1["visible_test_ok"] else f"failed: {s1['visible_error']}"
        if comp == "solver_prompt":
            if len(steps) > 1:
                s2 = steps[1]
                rep = (f"The repair step produced:\n{_clip(s2['extracted_code'], MAX_CODE_CHARS)}\n"
                       f"Visible test after repair: {'passed' if s2['visible_test_ok'] else 'failed: ' + str(s2['visible_error'])}")
            else:
                rep = "not needed (visible test passed)"
            return {"Inputs": solve_user_msg(task), "Generated code": _clip(s1["extracted_code"], MAX_CODE_CHARS),
                    "Visible test result": vis1, "Repair attempt": rep, "Feedback": self._feedback(t)}
        # repair_prompt
        if len(steps) < 2:
            return None
        s2 = steps[1]
        return {"Inputs": repair_user_msg(task, _clip(s1["extracted_code"], MAX_CODE_CHARS), s1["visible_error"] or ""),
                "Generated code": _clip(s2["extracted_code"], MAX_CODE_CHARS),
                "Visible test result": "passed" if s2["visible_test_ok"] else f"failed: {s2['visible_error']}",
                "Repair attempt": "this record IS the repair attempt", "Feedback": self._feedback(t)}

    def _records_standard(self, trajs, comp):
        recs = [r for t in trajs if (r := self._record(t, comp)) is not None]
        if not recs:  # repair never triggered in this minibatch
            recs = [{"Inputs": solve_user_msg(t["task"]), "Generated code": _clip(t["final_code"], MAX_CODE_CHARS),
                     "Visible test result": "passed", "Repair attempt": "not triggered",
                     "Feedback": "The repair step was not triggered for this example. " + self._feedback(t)}
                    for t in trajs]
        return recs

    def _find_success(self, t: dict, comp: str, exclude_hash: str, used: set):
        """Successful archived trace for contrast: same task first, then any other train task."""
        def ok(x):
            if not x["final_pass"] or x["candidate_hash"] == exclude_hash or not x["steps"]:
                return False
            return comp == "solver_prompt" or x["repair_used"]
        same = [x for x in reversed(self.archive.get(t["task_id"], [])) if ok(x)]
        if same:
            return same[0], "the SAME task, solved by an earlier variant of the instruction"
        for tid in sorted(self.archive, key=lambda k: -len(self.archive[k])):
            if tid == t["task_id"] or tid in used:
                continue
            cand = [x for x in reversed(self.archive[tid]) if ok(x)]
            if cand:
                return cand[0], "a DIFFERENT training task, solved successfully"
        return None, None

    def _records_contrastive(self, trajs, comp, parent_hash):
        recs, used, n_pairs = [], set(), 0
        for t in sorted(trajs, key=lambda x: x["final_pass"]):  # failures first
            r = self._record(t, comp)
            if r is None:
                continue
            r = {"Outcome": "SUCCEEDED" if t["final_pass"] else "FAILED", **r}
            if t["final_pass"]:
                r["Feedback"] += " Keep the instruction behaviour that made this attempt succeed."
            elif n_pairs < MAX_CONTRAST_PAIRS:
                s, src = self._find_success(t, comp, parent_hash, used)
                if s is not None:
                    used.add(s["task_id"])
                    n_pairs += 1
                    sc = self.candidates.get(s["candidate_hash"], {}).get(comp, "")
                    r["Contrasting successful attempt"] = {
                        "Source": src,
                        "Instruction used": _clip(sc, MAX_INSTR_CHARS),
                        "Inputs": solve_user_msg(s["task"]) if comp == "solver_prompt"
                        else repair_user_msg(s["task"], _clip(s["steps"][0]["extracted_code"], MAX_CODE_CHARS),
                                             s["steps"][0]["visible_error"] or ""),
                        "Generated code": _clip(s["steps"][-1]["extracted_code"] if comp == "repair_prompt"
                                                else s["steps"][0]["extracted_code"], MAX_CODE_CHARS),
                        "Feedback": self._feedback(s),
                    }
                    self.contrast_log.append({"failed_task": t["task_id"], "success_task": s["task_id"],
                                              "success_candidate": s["candidate_hash"], "component": comp,
                                              "same_task": s["task_id"] == t["task_id"]})
                r["Feedback"] += (" Contrast this failure with the successful attempt(s): state what the successful "
                                  "attempt did differently and turn it into a general rule in the instruction.")
            recs.append(r)
        if not recs:
            recs = self._records_standard(trajs, comp)
        return recs

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        trajs = eval_batch.trajectories or []
        parent_hash = sha(candidate)
        out = {}
        for comp in components_to_update:
            out[comp] = (self._records_contrastive(trajs, comp, parent_hash) if self.mode == "contrastive"
                         else self._records_standard(trajs, comp))
        return out
