"""Complete real optimizer ACP/CLI lifecycle using only a local scripted API."""
import hashlib
import json
from pathlib import Path
import uuid

import httpx
import yaml

from . import go_gateway as go
from .engine import build
from .scoped_gateway import ScopedGoGateway
from .optimizer_provider import create_provider
from .complete_engine import require_complete_search


SCRIPT=r'''python3 - <<'PY'
import json
from pathlib import Path
schema=json.loads(Path('.autosaddler/session_output_schema.json').read_text())
context=json.loads(Path('.autosaddler/session_context.json').read_text())
version=schema['properties']['schema_version']['const']
if 'evolution' in version:
 result={'schema_version':version,'parent_ids':[context['candidate_ids'][0]],'component_sources':{},'rationale':'Synthetic local lifecycle check'}
elif 'diagnosis' in version:
 path=Path('instructions.md');path.write_text(path.read_text()+'\nSMOKE_VERIFIED_INSTRUCTION\n')
 result={'schema_version':version,'intent':'Synthetic lifecycle patch','diagnosis':'Synthetic evaluator only','expected_effect':'Synthetic pass','changed_paths':['instructions.md']}
else:
 result={'schema_version':version,'lessons':[]}
result['evidence_receipt']=json.loads(Path('.autosaddler/evidence_index.json').read_text())['files']
Path('.autosaddler/session_output.json').write_text(json.dumps(result))
print('STRUCTURED_OUTPUT_WRITTEN')
PY'''


def main():
    package=Path(__file__).parent
    sources=sorted(package.glob('*.py'))
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    root=go.ROOT/'results/autosaddler_reproduction/full_protocol_preparation/fixture'/uuid.uuid4().hex[:10]
    root.mkdir(parents=True)
    config=yaml.safe_load((package/'configs/smoke.yaml').read_text())
    config['storage']['run_root']=str(root/'runs')
    config['scenario']['settings'].update(benchmarks=['terminal_bench_2'],require_complete_optimization_stages=True,opencode_prompt_version=1)
    config['optimization'].update(task_selection={'type':'balanced_benchmarks','batch_size':1,'seed':42},budget={'max_iterations':2,'max_rollouts':30},diagnosis_patch_timeout_seconds=240,selection_timeout_seconds=240,reflection_timeout_seconds=240,session_retries=1)
    path=root/'config.yaml';path.write_text(yaml.safe_dump(config))
    go.credential=lambda:{'access':'fixture','metadata':{'orgID':'fixture'}}
    go.quota=lambda *args:{k:{'status':'ok','percent':0} for k in ('rolling','weekly','monthly')}
    gateway=ScopedGoGateway(root/'gateway',20,40,require_context_policy=True)
    wire=[]; omitted=False; repair_seen=False
    def respond(request):
        nonlocal omitted,repair_seen
        payload=json.loads(request.content);row=gateway.rows[-1]
        wire.append(dict(purpose=row['purpose'],bytes=len(request.content)))
        assert payload['model']=='kimi-k2.7-code'
        messages=payload.get('messages',[])
        repair_seen = repair_seen or 'The current stage has no valid final JSON artifact.' in json.dumps(messages)
        tool=row['purpose']=='primary' and messages[-1].get('role')=='user'
        # First response intentionally omits final JSON, exercising a same-stage
        # contract-repair turn through the actual ACP transport.
        command=SCRIPT
        first=tool and not omitted
        if tool and not repair_seen: command='printf SYNTHETIC_MISSING_OUTPUT'
        if first: omitted=True
        delta={'tool_calls':[{'index':0,'id':'call_'+str(len(wire)),'type':'function','function':{'name':'bash','arguments':json.dumps({'command':command,'description':'Execute synthetic lifecycle contract'})}}]} if tool else {'content':'FIXTURE_STAGE_DONE'}
        if first: delta['content']='LONG_NATIVE_RESPONSE '*14000
        base={'id':'fixture-'+str(len(wire)),'created':1,'model':payload['model'],'object':'chat.completion.chunk'}
        output=68883 if first else 20;tokens=40000 if first else 100
        meter={'cost':0,'opencode':{'billing_source':'go-plus','allowance':.00001,'tokens':{'input':tokens,'output':output}}}
        events=[{**base,'choices':[{'index':0,'delta':{'role':'assistant',**delta},'finish_reason':None}]},
                {**base,'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls' if tool else 'stop'}]},
                {**base,'choices':[],'usage':{'prompt_tokens':tokens,'completion_tokens':output,'total_tokens':tokens+output,**meter}}]
        if len(wire)>35: raise RuntimeError('Unexpected local fixture loop')
        data=b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)+b'data: [DONE]\n\n'
        return httpx.Response(200,headers={'content-type':'text/event-stream'},content=data)
    gateway.client.close();gateway.client=httpx.Client(transport=httpx.MockTransport(respond))
    with gateway:
        runtime=build(path,'native-complete-fixture')
        runtime.engine.provider=create_provider(ledger=None,settings={'model':'opencode-go/kimi-k2.7-code','output_repair_attempts':1},run_id='native-complete-fixture')
        result=runtime.engine.run();proof=require_complete_search(runtime.store,2)
        events=runtime.store.events()
        requests=[json.loads((runtime.store.run_dir/e.payload['request']['uri']).read_text()) for e in events if e.event_type=='SessionStarted']
        assert {r['spec']['kind'] for r in requests}=={'evolve','diagnose_patch','reflect'}
        assert result.development_score==1
        assert list(runtime.store.run_dir.rglob('structured_output_check_0.json'))
        assert any(r['purpose']=='compaction' and r['bytes']>210000 for r in wire)
        count=len(events);runtime.engine.run();assert len(runtime.store.events())==count
        assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}==frozen
        proof.update(real_native_optimizer_transport=True,session_kinds=['evolve','diagnose_patch','reflect'],structured_output_repair_verified=True,
            native_compaction_verified=True,resume_does_not_repeat_completed_work=True,external_model_calls=0,
            synthetic_scores_not_benchmark_results=True,fixture_requests=len(wire),disconnect_watch_version=go.DISCONNECT_WATCH_VERSION,source_sha256=frozen,artifacts=str(root))
        go.save(root/'proof.json',proof);go.save(root.parent/'latest.json',proof)
    print(json.dumps(proof,indent=2))


if __name__=='__main__': main()
