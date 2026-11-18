"""Prepare the frozen SWE V1 test environments without any model invocation."""
import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .case_selection import select_ids
from .go_gateway import save
from .swe_adapter import ROOT, load_row

OUT = ROOT / 'results/autosaddler_reproduction/swe_go_preflight_20261105'


def prepare():
    import docker
    manifest = ROOT / 'experiments/autosaddler_reproduction/manifests/swe_bench_pro.json'
    ids = select_ids(json.loads(manifest.read_text())['test'], 5, 42, 'swe_bench_pro', 'test')
    plan = dict(benchmark='swe_bench_pro', version='V1', test_ids=ids, seed=42, repetitions=1,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                evaluator_revision=subprocess.check_output(['git','-C',str(ROOT/'third_party/SWE-bench_Pro-os'),'rev-parse','HEAD'], text=True).strip(),
                raw_data_sha256=hashlib.sha256((ROOT/'third_party/experiment_data/swe_bench_pro_v1.jsonl').read_bytes()).hexdigest(),
                candidate_ids={'H0':'sha256:8b13738861628a4603a40a31be7afc63650d920a8e42495ee4c953bb818dfd87',
                               'H1':'sha256:51464eef1a97de6d9387cae4522742c204f96dbbf6822975aa667ab0ff770410'},
                optimization_repeated=False, model_calls=0,
                target_model='opencode-go/qwen3.7-plus', agent_timeout_seconds=1800,
                timeout_origin='Predeclared bounded pilot generation limit, identical for H0/H1; not claimed as an official SWE task deadline')
    if (OUT/'frozen_plan.json').exists() and json.loads((OUT/'frozen_plan.json').read_text()) != plan:
        raise RuntimeError('SWE preflight plan drift')
    save(OUT/'frozen_plan.json', plan)
    client = docker.from_env(timeout=2400)
    images=[]
    try:
        for index, task_id in enumerate(ids):
            row=load_row(task_id);tag='jefzda/sweap-images:'+row['dockerhub_tag']
            save(OUT/'status.json', dict(status='running',task_id=task_id,index=index,model_calls=0))
            try:
                image=client.images.get(tag)
            except docker.errors.ImageNotFound:
                image=client.images.pull(tag)
            container=client.containers.run(image.id,entrypoint='/bin/sh',command=['-c','sleep infinity'],detach=True,
                name='autoharness-swe-preflight-'+hashlib.sha256(task_id.encode()).hexdigest()[:10],mem_limit='8g',pids_limit=512)
            try:
                result=container.exec_run(['/bin/sh','-c','git rev-parse --show-toplevel'])
                if result.exit_code: raise RuntimeError('Task repository unavailable')
                repo=result.output.decode().strip()
                commit=container.exec_run(['git','cat-file','-t',row['base_commit']],workdir=repo)
                if commit.exit_code or commit.output.strip()!=b'commit': raise RuntimeError('Base commit unavailable')
                container.exec_run(['mkdir','-p','/opt/autoharness'])
                from .swe_adapter import _archive_files
                binary=ROOT/'third_party/opencode-1.19.5-linux/opencode'
                container.put_archive('/opt/autoharness',_archive_files([('opencode',binary.read_bytes(),True)]))
                version=container.exec_run(['/opt/autoharness/opencode','--version'])
                if version.exit_code or b'1.19.5' not in version.output: raise RuntimeError('Pinned CLI incompatible')
                images.append(dict(task_id=task_id,tag=tag,image_id=image.id,repo_dir=repo,base_commit=row['base_commit'],opencode_version=version.output.decode().strip()))
                save(OUT/'progress.json',dict(images=images,model_calls=0))
            finally:
                container.remove(force=True)
        save(OUT/'completed.json',dict(plan=plan,images=images,model_calls=0,passed=True))
        save(OUT/'status.json',dict(status='complete',model_calls=0,updated=datetime.now(timezone.utc).isoformat()))
    finally:
        client.close()


if __name__=='__main__':
    try:
        prepare()
    except Exception:
        save(OUT/'status.json',dict(status='stopped',model_calls=0,error=traceback.format_exc()))
        raise
