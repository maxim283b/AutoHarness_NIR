import json
from types import SimpleNamespace

import httpx
import pytest

from experiments.autosaddler_reproduction import go_gateway as go


def test_large_billed_output_is_reserved_without_changing_request(tmp_path, monkeypatch):
    payload = {"model":"qwen3.7-plus", "max_tokens":8192, "messages":[{"role":"user", "content":"fixture"}]}
    original = json.loads(json.dumps(payload))
    seen = []
    usage = {"cost":0, "opencode":{"billing_source":"go-plus", "allowance":0.07946539,
        "tokens":{"input":6,"output":36890,"cache_read":14316}, "list_cost":0.05959904}}
    def upstream(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"usage":usage})
    gateway = go.GoGateway(tmp_path)
    gateway.client.close()
    gateway.client = httpx.Client(transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(go, "credential", lambda:{"access":"fixture","metadata":{"orgID":"fixture"}})
    monkeypatch.setattr(go, "quota", lambda *args:{})
    status, _, _ = gateway.forward('/anthropic/v1/messages', payload, {'x-autoharness-role':'target_final_baseline'})
    assert status == 200 and seen == [original] and payload == original
    assert gateway.rows[-1]['reservation'] > 0.07946539
    assert gateway.rows[-1]['usage']['total_tokens'] == 51212
    assert gateway.spent() == 0.07946539
    gateway.role_limits['target_final_baseline'] = gateway.spent() + 0.01
    with pytest.raises(RuntimeError, match='role budget'):
        gateway.forward('/anthropic/v1/messages', payload, {'x-autoharness-role':'target_final_baseline'})
    assert len(seen) == 1
    with pytest.raises(RuntimeError, match='8192'):
        go.reservation(dict(payload, max_tokens=8193), 'qwen3.7-plus')
    gateway.client.close()


def test_evaluation_only_uses_frozen_selection_and_fresh_equal_test_cases(tmp_path, monkeypatch):
    from experiments.autosaddler_reproduction import runner
    from autosaddler.v2.core.run_state import RunState
    package = tmp_path / 'package'
    (package / 'manifests').mkdir(parents=True)
    (package / 'manifests/terminal_bench_2.json').write_text(json.dumps({'test':['a','b','c','d','e']}))
    monkeypatch.setattr(runner, 'PACKAGE', package)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    seed = SimpleNamespace(candidate_id='seed')
    selected = SimpleNamespace(candidate_id='selected')
    state = SimpleNamespace(optimization_completed=True,selected_candidate_id='selected',candidates={'selected':selected})
    monkeypatch.setattr(RunState, 'replay', lambda events:state if events == ['source-events'] else pytest.fail('Wrong selection journal'))
    made = []
    def materialize(candidate, purpose):
        made.append(candidate.candidate_id)
        return SimpleNamespace(root=candidate.candidate_id, release=lambda:None)
    source = SimpleNamespace(store=SimpleNamespace(run_id='source',events=lambda:['source-events'],events_of_type=lambda kind:[]),
        scenario=SimpleNamespace(harness_space=SimpleNamespace(seed=lambda:seed,materialize=materialize)))
    dest = tmp_path / 'evaluation';dest.mkdir()
    runtime = SimpleNamespace(store=SimpleNamespace(run_id='new-evaluation',run_dir=dest),
        config=SimpleNamespace(scenario=SimpleNamespace(settings={'benchmarks':['terminal_bench_2'],'mode':'real','target_model':'opencode-go/qwen3.7-plus','case_selection_seed':42}),storage=SimpleNamespace(run_root=tmp_path)))
    class Ledger:
        def __init__(self,*args): pass
        def totals(self,*args): return {}
        def close(self): pass
    monkeypatch.setattr(runner,'UsageLedger',Ledger)
    calls=[]
    def run_case(case,candidate,path,settings,run_id,role):
        path.mkdir(parents=True,exist_ok=True)
        calls.append((candidate,case.case_id,run_id,role))
        return {'score':1,'benchmark':'terminal_bench_2','model':settings['target_model']}
    monkeypatch.setattr(runner,'run_case',run_case)
    table=runner.evaluate_final(runtime,repetitions=1,n_test=5,selection_runtime=source)
    assert len(calls)==10 and made==['seed','selected']
    assert {c[1] for c in calls[:5]}=={c[1] for c in calls[5:]}
    assert all(c[2]=='new-evaluation' for c in calls)
    assert table[0]['optimization_run_id']=='source'
    assert json.loads((dest/'final_evaluation_plan.json').read_text())['selection_run_id']=='source'
    runner.evaluate_final(runtime,repetitions=1,n_test=5,selection_runtime=source)
    assert len(calls)==10  # Only this evaluation run's completed results are cached.
