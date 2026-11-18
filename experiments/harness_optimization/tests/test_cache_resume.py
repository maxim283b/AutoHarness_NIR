import json

from experiments.harness_optimization.engine import ExperimentRun, ModelSettings, create_manifest
from experiments.harness_optimization.genome import seed_genome


class MockClient:
    calls = 0

    def __init__(self, settings):
        self.settings = settings

    def chat(self, messages, max_tokens):
        MockClient.calls += 1
        return {"text": "SELECT product_id, name FROM products WHERE category='Books' AND price < 30 ORDER BY product_id",
                "prompt_tokens": 10, "completion_tokens": 10, "latency_s": 0.0, "finish_reason": "stop"}


def test_cache_key_and_resume_no_duplicate(tmp_path):
    protocol = json.loads((__import__("pathlib").Path(__file__).parents[1] / "preregistered_protocol.json").read_text(encoding="utf-8"))
    run = ExperimentRun(tmp_path, protocol, client_factory=MockClient)
    task = run.tasks["search"][0]
    genome = seed_genome()
    first = run.evaluate(genome, task, "search", "seed_harness", 0, "search")
    second = run.evaluate(genome, task, "search", "seed_harness", 0, "search")
    assert first["cache_key"] == second["cache_key"]
    assert len((tmp_path / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert MockClient.calls == 1
    events = [json.loads(line) for line in (tmp_path / "optimization_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([event for event in events if event["event"] == "evaluation_used"]) == 1

    resumed = ExperimentRun(tmp_path, protocol, client_factory=MockClient)
    resumed.evaluate(genome, task, "search", "seed_harness", 0, "search")
    events = [json.loads(line) for line in (tmp_path / "optimization_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([event for event in events if event["event"] == "evaluation_used"]) == 1
    assert MockClient.calls == 1
