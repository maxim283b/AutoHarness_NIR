"""Serial environment-only installation of all frozen Terminal-Bench tasks."""
import json
import os
import subprocess
import tomllib
import traceback
from datetime import datetime, timezone

from .go_gateway import ROOT, save
from .engine import prepare_seed
from .terminal_adapter import run_task

OUT=ROOT/'results/autosaddler_reproduction/full_protocol_preparation/terminal_environments'


def main():
    manifest=json.loads((ROOT/'experiments/autosaddler_reproduction/manifests/terminal_bench_2.json').read_text())
    selected={k:manifest[k] for k in ('train','development','test')}
    tasks=[(split,task) for split,ids in selected.items() for task in ids]
    memory=int(subprocess.check_output(['docker','info','--format','{{.MemTotal}}'],text=True).strip())
    repo,revision=prepare_seed();images={}
    save(OUT/'frozen_selection.json',dict(selected=selected,seed_revision=revision,model_calls=0,workers=1))
    for index,(split,task_id) in enumerate(tasks):
        task=ROOT/'third_party/terminal-bench-2'/task_id
        config=tomllib.loads((task/'task.toml').read_text(encoding='utf-8'))
        environment=config['environment']
        if environment.get('gpus',0): raise RuntimeError('GPU must be provisioned for '+task_id)
        if environment.get('memory_mb',0)*1024**2>memory-2*1024**3:
            raise RuntimeError('Insufficient Docker memory for unchanged task '+task_id)
        if not (task/'tests/test.sh').is_file(): raise RuntimeError('Missing official verifier: '+task_id)
        save(OUT/'status.json',dict(status='running',pid=os.getpid(),index=index,total=len(tasks),task_id=task_id,split=split,model_calls=0))
        target=OUT/task_id
        installed=target/'installed.json'
        if installed.exists():
            result=json.loads(installed.read_text())
            if result.get('seed_revision')!=revision or result.get('inference_calls')!=0: raise RuntimeError('Stale setup proof')
        else:
            number=len(list(target.glob('attempt-*')))+1
            result=run_task(task_id,repo,'opencode-go/qwen3.7-plus',target/f'attempt-{number}',None,'environment-only','target_search',install_only=True)
            if not result.get('installed'): raise RuntimeError('Environment install did not complete')
            result.update(seed_revision=revision,inference_calls=0);save(installed,result)
        tag=environment['docker_image']
        image=json.loads(subprocess.check_output(['docker','image','inspect',tag],text=True,timeout=30))[0]
        images[task_id]=dict(tag=tag,id=image['Id'],digests=image.get('RepoDigests',[]))
        save(OUT/'progress.json',dict(completed=index+1,total=len(tasks),images=images,model_calls=0))
    save(OUT/'completed.json',dict(selected=selected,seed_revision=revision,images=images,inference_calls=0))
    save(OUT/'status.json',dict(status='complete',model_calls=0,total=len(tasks),updated=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    try: main()
    except Exception:
        save(OUT/'status.json',dict(status='stopped',pid=os.getpid(),model_calls=0,error=traceback.format_exc()))
        raise
