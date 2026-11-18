"""Actual SWE image/native CLI against a local synthetic API. No external models."""
import hashlib
import json
import shutil
import time
import tarfile
import uuid

import httpx

from . import go_gateway as go
from .swe_adapter import load_row
from .swe_go_adapter import generate
from .scoped_gateway import ScopedGoGateway, target_scope
from .swe_cached_images import ensure_image, image_lease
from .opencode_runtime import candidate_hash
from autosaddler.v2.core.engine import _copy_session_file


def main():
    sources=sorted((go.ROOT/'experiments/autosaddler_reproduction').glob('*.py'))
    frozen_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    root=go.ROOT/'results/autosaddler_reproduction/swe_go_fixture'/uuid.uuid4().hex[:10]
    root.mkdir(parents=True)
    selected=json.loads((go.ROOT/'fullsource/selected-manifest.json').read_text())
    row=load_row(selected['train'][0])
    with image_lease(): image=ensure_image(row['instance_id'],row)
    candidate=root/'candidate'
    shutil.copytree(go.ROOT.parent/'tbv7/experiments/autosaddler_reproduction/harness_seed',candidate,copy_function=_copy_session_file)
    (candidate/'instructions.md').write_text('SWE_GO_INSTRUCTION_FIXTURE. Follow the synthetic task only.\n')
    go.credential=lambda:{'access':'local-fixture','metadata':{'orgID':'fixture'}}
    go.quota=lambda *args:{k:{'status':'ok','percent':0} for k in ('rolling','weekly','monthly')}
    gateway=ScopedGoGateway(root/'gateway',20,20,require_context_policy=True)
    wire=[]; primary=0
    def respond(request):
        nonlocal primary
        payload=json.loads(request.content); entry=gateway.rows[-1]
        wire.append(dict(purpose=entry['purpose'],payload=payload,bytes=len(request.content),role=entry['role']))
        if entry['purpose']=='primary': primary+=1
        first=entry['purpose']=='primary' and primary==1
        meter={'cost':0,'opencode':{'billing_source':'go-plus','allowance':.00001,'tokens':{'input':40000 if first else 100,'output':68883 if first else 10}}}
        content={'type':'tool_use','id':'call_fixture','name':'bash','input':{'command':"printf SWE_PATCH_FIXTURE > swe_fixture.txt; head -c 40000 /dev/zero | tr '\\000' x",'description':'Write synthetic fixture file and print long output'}} if first else {'type':'text','text':''}
        text='SWE_FIXTURE_DONE' if entry['purpose']=='primary' else 'Continue the synthetic fixture and finish.'
        events=[{'type':'message_start','message':{'id':'msg_fixture','type':'message','role':'assistant','model':payload['model'],'content':[],'stop_reason':None,'usage':{'input_tokens':40000 if first else 100,'output_tokens':0}}},
                {'type':'content_block_start','index':0,'content_block':content},
                *([] if first else [{'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':text}}]),
                {'type':'content_block_stop','index':0},
                *([{'type':'content_block_start','index':1,'content_block':{'type':'text','text':''}}, {'type':'content_block_delta','index':1,'delta':{'type':'text_delta','text':'LONG_NATIVE_RESPONSE '*14000}}, {'type':'content_block_stop','index':1}] if first else []),
                {'type':'message_delta','delta':{'stop_reason':'tool_use' if first else 'end_turn','stop_sequence':None},'usage':{'output_tokens':68883 if first else 10,**meter}}, {'type':'message_stop'}]
        if len(wire)>5: raise RuntimeError('Synthetic fixture loop')
        data=b''.join(('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n').encode() for e in events)
        return httpx.Response(200,headers={'content-type':'text/event-stream'},content=data)
    gateway.client.close(); gateway.client=httpx.Client(transport=httpx.MockTransport(respond))
    with gateway:
        with target_scope('offline-swe-fixture','target_final_baseline',row['instance_id'],candidate_hash(candidate)) as token:
            patch,meta=generate(row,candidate,'opencode-go/qwen3.7-plus',root/'normal','offline-swe-fixture','target_final_baseline',image,timeout=120,prompt_override='Write the synthetic fixture file using bash, then finish.',scoped_token=token)
        assert 'swe_fixture.txt' in patch and 'SWE_PATCH_FIXTURE' in patch
        assert meta['agent_cleanup_verified'] and meta['generation_container_removed_before_grading']
        assert not meta['agent_timed_out'] and meta['native_completion']['completion_reason'] in {'native_root_idle','process_exit'}
        assert [x['purpose'] for x in wire]==['primary','compaction','primary']
        assert 'SWE_GO_INSTRUCTION_FIXTURE' in json.dumps(wire[0]['payload'])
        assert 'LONG_NATIVE_RESPONSE '*14000 in json.dumps(wire[1]['payload'])
        assert wire[1]['bytes']>210000
        trace=(root/'normal/opencode.jsonl').read_text(encoding='utf-8')
        assert 'full 40000 bytes saved' in trace
        with tarfile.open(root/'normal/agent_logs.tar.gz','r:gz') as archive:
            full=[m for m in archive.getmembers() if '/context-outputs/' in m.name and m.isfile()]
            assert len(full)==1 and archive.extractfile(full[0]).read()==b'x'*40000
        summary=gateway.accounting_summary()
    # Separate synthetic deadline check, with a live streaming connection.
    stalled=ScopedGoGateway(root/'timeout_gateway',20,20,require_context_policy=True)
    class Slow(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(120):
                yield b'event: ping\ndata: {"type":"ping"}\n\n'
                time.sleep(.2)
    stalled.client.close();stalled.client=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Slow())))
    with stalled:
        with target_scope('offline-swe-fixture','target_final_optimized',row['instance_id'],candidate_hash(candidate)) as token:
            _,timeout_meta=generate(row,candidate,'opencode-go/qwen3.7-plus',root/'timeout','offline-swe-fixture','target_final_optimized',image,timeout=6,prompt_override='Wait for the synthetic local fixture.',scoped_token=token)
        assert timeout_meta['agent_timed_out'] and timeout_meta['agent_cleanup_verified']
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}==frozen_sources, 'Fixture source changed during execution'
    proof=dict(passed=True,paid_model_requests=0,synthetic_usage_not_campaign_spend=True,
               fixture_requests=len(wire),native_compactions=1,long_response_compaction_verified=True,
               native_instruction_loaded=True,patch_extracted=True,cleanup_before_grading_verified=True,
               timeout_cleanup_verified=True,image_id=image['image_id'],normal_accounting=summary,artifacts=str(root),installed_binary_sha256=meta['installed_candidate_binary_sha256'],
               timeout_unknown_reservation_preserved=any(r.get('usage') is None for r in stalled.rows),
               source_sha256=frozen_sources)
    go.save(root/'wire.json',wire);go.save(root/'proof.json',proof);go.save(root.parent/'latest.json',proof)
    print(json.dumps(proof,indent=2))


if __name__=='__main__': main()
