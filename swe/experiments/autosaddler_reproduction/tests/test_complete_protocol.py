import asyncio
import json
import time
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from autosaddler.v2.prompting.models import SessionResult

from experiments.autosaddler_reproduction.complete_engine import IncompleteOptimizationStage, require_complete_search
from experiments.autosaddler_reproduction.engine import build, SmokeProvider
from experiments.autosaddler_reproduction.optimizer_provider import Transport, structured_output_error


def runtime(tmp_path):
    package=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((package/'configs/smoke.yaml').read_text())
    config['storage']['run_root']=str(tmp_path/'runs')
    config['scenario']['settings'].update(benchmarks=['terminal_bench_2'],require_complete_optimization_stages=True,opencode_prompt_version=1)
    config['optimization']['task_selection']['batch_size']=1
    config['optimization']['budget']={'max_iterations':2,'max_rollouts':30}
    config['optimization']['session_retries']=0
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    return build(path,'complete-fixture')


class FaultProvider(SmokeProvider):
    def __init__(self,kind,once=False): self.kind=kind;self.once=once;self.failed=False
    async def run(self,request):
        if request.spec.kind==self.kind and (not self.once or not self.failed):
            self.failed=True
            return SessionResult(status='failed',error='Synthetic technical fault',raw_response='',structured_output=None,tool_calls=(),usage=())
        return await super().run(request)


@pytest.mark.parametrize('kind',['evolve','diagnose_patch','reflect'])
def test_technical_stage_cannot_be_skipped(tmp_path,kind):
    r=runtime(tmp_path);r.engine.provider=FaultProvider(kind)
    with pytest.raises(IncompleteOptimizationStage): r.engine.run()
    events=r.store.events()
    assert not any(e.event_type in {'OptimizationCompleted','RunCompleted','DeferredWorkAbandoned'} for e in events)
    if kind=='diagnose_patch':
        assert not any(e.event_type=='IterationCompleted' for e in events)


@pytest.mark.parametrize('kind',['evolve','diagnose_patch','reflect'])
def test_transient_fault_recovers_all_required_stages(tmp_path,kind):
    r=runtime(tmp_path);r.engine.provider=FaultProvider(kind,once=True);r.engine.session_retries=1
    result=r.engine.run()
    assert result.iterations==2
    assert require_complete_search(r.store,2)['passed']
    count=len(r.store.events());assert r.engine.run()==result;assert len(r.store.events())==count
    completed={e.payload['logical_operation_id'] for e in r.store.events() if e.event_type=='SessionCompleted'}
    assert any('diagnose-patch' in x for x in completed)
    assert any('reflect' in x for x in completed)
    assert any('evolve' in x for x in completed)


def test_shortened_search_cannot_pass_final_gate():
    store=SimpleNamespace(events=lambda:[SimpleNamespace(event_type='IterationCompleted',payload={'iteration':0})])
    with pytest.raises(IncompleteOptimizationStage,match='Expected 2'): require_complete_search(store,2)


def test_unfinished_session_blocks_final_gate():
    events=[SimpleNamespace(event_type='IterationCompleted',payload={'iteration':0}),
            SimpleNamespace(event_type='SessionStarted',payload={'logical_operation_id':'unfinished-reflection'})]
    with pytest.raises(IncompleteOptimizationStage,match='Unresolved'):
        require_complete_search(SimpleNamespace(events=lambda:events),1)


def test_structured_output_is_checked_against_the_actual_schema(tmp_path):
    output=tmp_path/'output.json';schema=tmp_path/'schema.json'
    schema.write_text(json.dumps({'type':'object','required':['actual_result'],'properties':{'actual_result':{'type':'string'}}}))
    evidence=tmp_path/'evidence.txt';evidence.write_text('verified training evidence')
    files={'evidence.txt':hashlib.sha256(evidence.read_bytes()).hexdigest()}
    (tmp_path/'.autosaddler').mkdir();(tmp_path/'.autosaddler/evidence_index.json').write_text(json.dumps({'files':files}))
    session=SimpleNamespace(output_path=output,output_schema_path=schema,workspace=tmp_path)
    assert structured_output_error(session)
    output.write_text('{"wrong":true}');assert structured_output_error(session)
    output.write_text(json.dumps({'actual_result':'verified','evidence_receipt':files}));assert structured_output_error(session) is None
    evidence.write_text('changed after receipt');assert structured_output_error(session)


def test_cancel_waits_for_native_worker_cleanup(monkeypatch):
    done=[];transport=Transport('opencode-go/kimi-k2.7-code')
    def worker(*args):
        try: time.sleep(.05)
        finally: done.append('cleaned')
    from experiments.autosaddler_reproduction import gaia_optimizer_supervisor
    monkeypatch.setattr(gaia_optimizer_supervisor,'run_session',worker)
    async def check():
        with pytest.raises(TimeoutError): await asyncio.wait_for(transport.run(None,1),timeout=.005)
        assert done==['cleaned']
    asyncio.run(check())
