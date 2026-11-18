"""Evaluation only: same TB-selected H0/H1 on five frozen SWE V1 test IDs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
import zipfile
from datetime import datetime, timezone

from .go_gateway import GoGateway, ROOT, save
from .opencode_runtime import candidate_hash
from .swe_adapter import load_row
from .swe_go_adapter import evaluate
from .terminal_final_audit import summarize

BASE=ROOT/'results/autosaddler_reproduction'
PACKAGE=ROOT/'experiments/autosaddler_reproduction'
TB='go-terminal-contextbound-eval-20261105'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(artifacts):
    if (artifacts/'gateway/requests.jsonl').exists():
        raise RuntimeError('Cannot refreeze a dispatched experiment; use a new run ID')
    prepared=read(BASE/'swe_go_preflight_20261105/completed.json')
    fixture=read(BASE/'swe_go_fixture/latest.json')
    assert prepared['passed'] and fixture['passed'] and fixture['paid_model_requests']==0
    assert sha(PACKAGE/'manifests/swe_bench_pro.json')==prepared['plan']['manifest_sha256']
    assert sha(ROOT/'third_party/experiment_data/swe_bench_pro_v1.jsonl')==prepared['plan']['raw_data_sha256']
    assert subprocess.check_output(['git','-C',str(ROOT/'third_party/SWE-bench_Pro-os'),'rev-parse','HEAD'],text=True).strip()==prepared['plan']['evaluator_revision']
    for name,digest in fixture['source_sha256'].items():
        assert sha(PACKAGE/name)==digest, 'Stale SWE fixture'
    tb=BASE/'go_experiments'/TB
    audit=read(tb/'final_audit.json');assert audit['passed']
    old=read(tb/'frozen_plan.json')
    plan=dict(prepared['plan'],selection_source=old['selection_source'],context_policy=old['context_policy'],
              reservation_policy=old['reservation_policy'],stream_transport=old['stream_transport'],request_timeout_policy=old['request_timeout_policy'],
              budget_prior_runs=old['budget_prior_runs']+[TB],allowance_limit=200,request_limit=20000,
              role_allowance_limits=old['role_allowance_limits'],role_request_limits=old['role_request_limits'],images=prepared['images'],
              fixture_proof=str(Path(fixture['artifacts'])/'proof.json'),fixture_proof_sha256=sha(Path(fixture['artifacts'])/'proof.json'),
              evaluator='Pinned official SWE-bench Pro V1 fresh Docker grading',judge='official executable tests; no LLM judge')
    hashes={}
    for label in ('H0','H1'):
        src=next((BASE/'runs'/TB/'final_test'/label).glob('*/*/*/target_workspace'))
        target=artifacts/'harnesses'/label
        shutil.copytree(src,target,dirs_exist_ok=True)
        hashes[label]=candidate_hash(target)
        assert {r['candidate_hash'] for r in audit['results'] if r['label']==label}=={hashes[label]}
    plan['materialized_candidate_hashes']=hashes
    if (artifacts/'frozen_plan.json').exists() and read(artifacts/'frozen_plan.json')!=plan:
        raise RuntimeError('Frozen SWE plan drift')
    save(artifacts/'frozen_plan.json',plan)
    files=[p for p in PACKAGE.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.yaml','.md','.patch'} and '__pycache__' not in p.parts]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}
    save(artifacts/'source_manifest.json',manifest)
    with zipfile.ZipFile(artifacts/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in manifest: z.write(ROOT/path,path)
    save(artifacts/'status.json',dict(status='frozen',model_calls=0))
    print(json.dumps(dict(status='frozen',test_ids=plan['test_ids'],files=len(manifest))))


def campaign(gateway,run_id):
    rows=list({r['id']:r for r in gateway.rows}.values())
    active=set(gateway.active_requests)
    def snapshot(items):
        pending=[r for r in items if r['id'] in active and not r.get('usage') and not r.get('error')]
        ids={r['id'] for r in pending}
        result=summarize([r for r in items if r['id'] not in ids])
        result.update(requests=len(items),pending_requests=len(pending),pending_reserve=round(sum(r['reservation'] for r in pending),8))
        if pending: result['total_tokens']=None
        return result
    return dict(campaign_all_attempts=snapshot(rows),current_evaluation={label:snapshot([r for r in rows if r['run_id']==run_id and r['role']==role]) for label,role in [('H0','target_final_baseline'),('H1','target_final_optimized')]},
                details=gateway.accounting_summary(),compaction_is_subset=True)


def run(artifacts):
    import docker
    plan=read(artifacts/'frozen_plan.json');manifest=read(artifacts/'source_manifest.json')
    if any(sha(ROOT/p)!=digest for p,digest in manifest.items()): raise RuntimeError('Frozen SWE source drift')
    if sha(plan['fixture_proof'])!=plan['fixture_proof_sha256']: raise RuntimeError('Fixture proof drift')
    client=docker.from_env()
    try:
        for image in plan['images']:
            if client.images.get(image['tag']).id!=image['image_id']: raise RuntimeError('SWE image drift')
    finally: client.close()
    journals=[]
    for prior in plan['budget_prior_runs']:
        location=BASE/'go_experiments'/prior
        if read(location/'status.json')['status'] not in {'stopped','complete'}: raise RuntimeError('Prior run still active')
        journals.append(location/'gateway/requests.jsonl')
    status=dict(status='running',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),phase='final_evaluation')
    save(artifacts/'status.json',status)
    gateway=None
    try:
        with GoGateway(artifacts/'gateway',plan['allowance_limit'],plan['request_limit'],role_limits=plan['role_allowance_limits'],role_request_limits=plan['role_request_limits'],prior_journals=journals,require_context_policy=True) as gateway:
            results=[]
            for label,role in [('H0','target_final_baseline'),('H1','target_final_optimized')]:
                candidate=artifacts/'harnesses'/label
                if candidate_hash(candidate)!=plan['materialized_candidate_hashes'][label]: raise RuntimeError('Candidate drift')
                for task_id in plan['test_ids']:
                    path=artifacts/'final_test'/label/hashlib.sha256(task_id.encode()).hexdigest()[:16]
                    result_path=path/'result.json'
                    status.update(label=label,task_id=task_id,completed=len(results));save(artifacts/'status.json',status)
                    if result_path.exists():
                        result=read(result_path)
                        if result['candidate_id']!=plan['candidate_ids'][label] or result['task_id']!=task_id: raise RuntimeError('Cached result drift')
                    else:
                        # Failed attempts need explicit diagnosis/archive before a bounded recovery.
                        if path.exists() and any(path.iterdir()): raise RuntimeError('Unarchived previous attempt; diagnose before recovery')
                        image=next(i for i in plan['images'] if i['task_id']==task_id)
                        result=evaluate(load_row(task_id),candidate,plan['target_model'],path,artifacts.name,role,image,plan['agent_timeout_seconds'])
                        result.update(label=label,task_id=task_id,benchmark='swe_bench_pro',model=plan['target_model'],candidate_id=plan['candidate_ids'][label],candidate_hash=candidate_hash(candidate),repetition=0)
                        save(result_path,result)
                    results.append(result)
                    if gateway.halted: raise RuntimeError(gateway.halted)
                    save(artifacts/'campaign_accounting.json',campaign(gateway,artifacts.name))
            comparison=dict(benchmark='swe_bench_pro',version='V1',N=5,H0_pass_at_1=sum(r['score'] for r in results if r['label']=='H0')/5,H1_pass_at_1=sum(r['score'] for r in results if r['label']=='H1')/5,completeness='10/10',judge=plan['judge'])
            save(artifacts/'comparison.json',comparison)
            status.update(status='complete',completed=10)
    except Exception as exc:
        status.update(status='stopped',error=str(exc));save(artifacts/'failure.json',dict(traceback=traceback.format_exc()))
    finally:
        if gateway is not None: save(artifacts/'campaign_accounting.json',campaign(gateway,artifacts.name))
        status['updated']=datetime.now(timezone.utc).isoformat();save(artifacts/'status.json',status)
    return 0 if status['status']=='complete' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run']);parser.add_argument('--run-id',required=True);args=parser.parse_args()
    if not args.run_id.replace('-','').isalnum(): parser.error('Invalid run ID')
    artifacts=BASE/'go_experiments'/args.run_id
    if args.phase=='freeze': freeze(artifacts)
    else: raise SystemExit(run(artifacts))
