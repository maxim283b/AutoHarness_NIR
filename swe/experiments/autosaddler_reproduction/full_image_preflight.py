"""Download official images before Harbor's unchanged environment-start timer."""
import argparse
import json
import os
import shutil
import time
import tomllib
import traceback
from datetime import datetime, timezone

from .go_gateway import ROOT, save

OUT=ROOT/'results/autosaddler_reproduction/full_protocol_preparation/image_downloads'


def inventory():
    manifest=json.loads((ROOT/'experiments/autosaddler_reproduction/manifests/terminal_bench_2.json').read_text())
    tasks=[]
    for split in ('train','development','test'):
        for task_id in manifest[split]:
            config=tomllib.loads((ROOT/'third_party/terminal-bench-2'/task_id/'task.toml').read_text(encoding='utf-8'))
            tasks.append(dict(split=split,task_id=task_id,tag=config['environment']['docker_image']))
    assert len(tasks)==89 and len({r['task_id'] for r in tasks})==89
    path=OUT/'frozen_inventory.json'
    plan=dict(tasks=tasks,model_calls=0,official_timeouts_unchanged=True)
    if path.exists() and json.loads(path.read_text())!=plan: raise RuntimeError('Image inventory drift')
    save(path,plan)
    return tasks


def main(pull):
    import docker
    tasks=inventory();client=docker.from_env(timeout=10800)
    images={};last_update=0
    try:
        for index,task in enumerate(tasks):
            status=dict(status='running',phase='image_download',pid=os.getpid(),index=index,total=len(tasks),**task,model_calls=0)
            save(OUT/'status.json',status)
            try: image=client.images.get(task['tag'])
            except docker.errors.ImageNotFound:
                if not pull: continue
                if shutil.disk_usage(ROOT).free<30*1024**3: raise RuntimeError('Host free disk below 30GiB reserve')
                for update in client.api.pull(task['tag'],stream=True,decode=True):
                    if update.get('error'): raise RuntimeError('Docker image pull: '+update['error'])
                    now=time.monotonic()
                    if now-last_update>2:
                        save(OUT/'download_progress.json',dict(task_id=task['task_id'],tag=task['tag'],layer=update.get('id'),status=update.get('status'),progress=update.get('progressDetail'),updated=datetime.now(timezone.utc).isoformat(),model_calls=0))
                        last_update=now
                image=client.images.get(task['tag'])
            images[task['task_id']]=dict(tag=task['tag'],id=image.id,digests=image.attrs.get('RepoDigests',[]))
            save(OUT/'progress.json',dict(available=len(images),total=len(tasks),images=images,model_calls=0))
        if pull:
            assert len(images)==89
            save(OUT/'completed.json',dict(images=images,model_calls=0,passed=True))
        save(OUT/'status.json',dict(status='complete' if pull else 'inventory_complete',available=len(images),total=89,model_calls=0,updated=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(available=len(images),total=89,pulled=pull,model_calls=0)))
    except Exception:
        status.update(status='stopped',error=traceback.format_exc())
        save(OUT/'status.json',status)
        raise
    finally: client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pull',action='store_true');args=parser.parse_args();main(args.pull)
