"""Load digest-pinned task images from a verified local OCI cache, without pulls."""
import hashlib
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import uuid
from contextlib import contextmanager
import sys

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'fullsource/image-digests.json'
CACHE = ROOT / 'third_party/experiment_image_cache'
PROOFS = ROOT / 'results/autosaddler_reproduction/fullsource_image_preflight'
FREE_FLOOR = 12 * 1024**3
STAGING_RESERVE = 6 * 1024**3
RECLAIM_IMAGE = 'docker/desktop-reclaim-space@sha256:8b26e45c51632fd78131f3bddca76122bebc1dd8376f16dd26f769c7f33121fd'


def execution_entry(task_id, original):
    path = ROOT/'fullsource/prepared-images.json'
    prepared = json.loads(path.read_text()).get(task_id) if path.exists() else None
    if not prepared:
        return original
    if prepared['source_registry_digest'] != original['registry_digest'] or prepared['task_source_unchanged'] is not True:
        raise RuntimeError('Prepared dependency image does not match the official source image')
    return {**original, **prepared['execution']}


@contextmanager
def image_lease():
    """Serialize our staging/eviction with each complete task attempt."""
    import fcntl
    PROOFS.mkdir(parents=True, exist_ok=True)
    with (PROOFS / 'management.lock').open('a+b') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def reclaim_deleted_space():
    if sys.platform != 'darwin':
        return
    before = shutil.disk_usage(ROOT).free
    result = subprocess.run(['docker','run','--rm','--pull','never','--platform','linux/amd64',
        '--privileged','--pid=host','--label','autoharness.kind=disk-space-reclamation', RECLAIM_IMAGE],
        capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError('Docker Desktop could not reclaim deleted space')
    with (PROOFS/'space-reclamation.jsonl').open('a') as file:
        file.write(json.dumps({'timestamp':time.time(),'utility_image':RECLAIM_IMAGE,
            'free_before':before,'free_after':shutil.disk_usage(ROOT).free,'no_objects_pruned':True})+'\n')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_offline(entry):
    """Stream one OCI image into Docker; no registry or image-tag resolution."""
    manifest_digest = entry['registry_digest']
    digests = [manifest_digest, entry['config']['digest']] + [item['digest'] for item in entry['layers']]
    index = {'schemaVersion': 2, 'manifests': [{
        'mediaType': entry['manifest']['mediaType'], 'digest': manifest_digest,
        'size': (CACHE / 'blobs/sha256' / manifest_digest.split(':')[1]).stat().st_size,
        'platform': {'os': 'linux', 'architecture': 'amd64'},
        'annotations': {'org.opencontainers.image.ref.name': entry['tag']}}]}
    log_path = PROOFS / ('load-' + manifest_digest.split(':')[1] + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab') as log:
        process = subprocess.Popen(['docker', 'load'], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
                for name, value in [('oci-layout', {'imageLayoutVersion': '1.0.0'}), ('index.json', index)]:
                    raw = json.dumps(value).encode()
                    info = tarfile.TarInfo(name); info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
                for digest in dict.fromkeys(digests):
                    algorithm, value = digest.split(':')
                    assert algorithm == 'sha256'
                    path = CACHE / 'blobs/sha256' / value
                    if not path.is_file() or path.is_symlink():
                        raise RuntimeError('Pinned local image blob is missing')
                    archive.add(path, arcname='blobs/sha256/' + value, recursive=False)
            process.stdin.close()
            if process.wait(timeout=900):
                raise RuntimeError('Offline Docker image import failed')
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=30)


def checked(container, command, **kwargs):
    result = container.exec_run(command, **kwargs)
    if result.exit_code:
        raise RuntimeError('Official task image failed setup verification')
    return result.output.decode().strip()


def unpacked_image_size(entry):
    """Count exact gzip output without allocating extracted staging files."""
    path=CACHE/'unpacked-layer-sizes.json'
    sizes=json.loads(path.read_text()) if path.exists() else {}
    for layer in entry['layers']:
        digest=layer['digest']
        if digest in sizes:continue
        if not layer['mediaType'].endswith(('+gzip','.gzip')):
            raise RuntimeError('Unrecognized compressed layer format for disk admission')
        blob=CACHE/'blobs/sha256'/digest.split(':')[1]
        total=0
        # Reading through EOF verifies gzip CRC/size too; a 32-bit ISIZE footer
        # alone would underestimate a layer larger than four GiB.
        with gzip.open(blob,'rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):total+=len(chunk)
        sizes[digest]=total
        save(path,sizes)
    return sum(sizes[layer['digest']] for layer in entry['layers'])


def ensure_image(task_id, row):
    import docker
    raw = PLAN.read_bytes()
    plan = json.loads(raw)
    original = plan['resolved'].get(task_id)
    if not original or plan.get('model_calls') != 0 or len(plan['resolved']) != plan['expected_tasks']:
        raise RuntimeError('Incomplete frozen image inventory')
    entry = execution_entry(task_id,original)
    if row['instance_id'] != task_id or row['base_commit'] != entry['base_commit']:
        raise RuntimeError('Task image plan drift')
    proof_path = PROOFS / (hashlib.sha256(task_id.encode()).hexdigest()[:20] + '.json')
    previous = json.loads(proof_path.read_text()) if proof_path.exists() else None
    plan_sha = hashlib.sha256(raw).hexdigest()
    if previous and (previous['registry_digest'] != original['registry_digest'] or previous['image_plan_sha256'] != plan_sha):
        raise RuntimeError('Task image provenance drift')
    if previous and previous.get('execution_digest',previous['registry_digest']) != entry['registry_digest']:
        if (ROOT/'fullsource/freeze.json').exists(): raise RuntimeError('Execution image changed after freeze')
        history = PROOFS/'history'; history.mkdir(exist_ok=True)
        save(history/(hashlib.sha256(task_id.encode()).hexdigest()[:20]+'-'+str(time.time_ns())+'.json'),previous)
        previous = None
    client = docker.from_env(timeout=900)
    container = None
    try:
        image = None
        for reference in (previous['image_id'],) if previous else (entry['registry_digest'], entry['config']['digest']):
            try:
                image = client.images.get(reference)
                break
            except docker.errors.ImageNotFound:
                pass
        if image is None:
            # Containerd's Docker Size can equal compressed content size.
            # Admission must use measured gzip output, never that API field.
            unpacked=unpacked_image_size(entry)
            reserve=unpacked+sum(layer['size'] for layer in entry['layers'])+1024**3
            evict_for_staging(client, protect_task=task_id, reserve_bytes=reserve)
            save(PROOFS / 'current-import.json', {'task_id': task_id, 'registry_digest': entry['registry_digest'], 'model_calls': 0})
            load_offline(entry)
            for attempt in range(50):
                try:
                    image = client.images.get(entry['registry_digest'])
                    break
                except docker.errors.ImageNotFound:
                    if attempt == 49: raise
                    time.sleep(0.2)
        if previous and image.id != previous['image_id']:
            raise RuntimeError('Reimported task image identity changed')
        config = json.loads((CACHE / 'blobs/sha256' / entry['config']['digest'].split(':')[1]).read_bytes())
        if image.attrs.get('Architecture') != 'amd64' or image.attrs['RootFS']['Layers'] != config['rootfs']['diff_ids']:
            raise RuntimeError('Local task image does not match the pinned filesystem')
        if image.id not in {entry['registry_digest'], entry['config']['digest']}:
            raise RuntimeError('Unexpected local image/config digest')
        if shutil.disk_usage(ROOT).free < FREE_FLOOR:
            raise RuntimeError('Image import crossed the free-space floor')
        container = client.containers.run(image.id, entrypoint='/bin/sh', command=['-c', 'sleep infinity'],
            detach=True, network_mode='none', platform='linux/amd64', mem_limit='6g', pids_limit=512,
            name='autoharness-swe-preflight-' + uuid.uuid4().hex[:12], labels={'autoharness.kind': 'no-model-image-preflight'})
        repo_dir = checked(container, ['git', 'rev-parse', '--show-toplevel'])
        checked(container, ['git', 'cat-file', '-e', row['base_commit'] + '^{commit}'], workdir=repo_dir)
        checked(container, ['git', 'reset', '--hard', row['base_commit']], workdir=repo_dir)
        assert checked(container, ['git', 'rev-parse', 'HEAD'], workdir=repo_dir) == row['base_commit']
        from .fullsource_host import checked_file, load as host_lock
        h0_sha = host_lock()['original_h0_binary']['sha256']
        if not previous or previous.get('native_h0_binary_sha256') != h0_sha:
            from .swe_adapter import _archive_files
            binary = checked_file('original_h0_binary')
            container.put_archive('/tmp',_archive_files([('autoharness-h0',binary.read_bytes(),True)]))
            version = checked(container,['/tmp/autoharness-h0','--version'])
            if version != '1.19.5': raise RuntimeError('Native H0 does not start in this task image')
        proof = {'passed': True, 'model_calls': 0, 'task_id': task_id, 'registry_digest': original['registry_digest'],
            'execution_digest':entry['registry_digest'],
            'tag': entry['tag'], 'image_id': image.id, 'repo_dir': repo_dir, 'base_commit': row['base_commit'],
            'image_plan_sha256': plan_sha, 'last_prepared_at': time.time(), 'network': 'none',
            'native_h0_binary_sha256':h0_sha,'native_h0_version':'1.19.5'}
        proof['docker_reported_size_bytes']=image.attrs['Size']
        proof['uncompressed_size_bytes']=unpacked_image_size(entry)
        container.remove(force=True); container = None
        proof['container_removed_before_inference'] = True
        save(proof_path, proof)
        return proof
    finally:
        if container is not None: container.remove(force=True)
        client.close()


def evict_for_staging(client, protect_task, reserve_bytes=STAGING_RESERVE):
    """Only this campaign's verified, unused task images may leave Docker."""
    import docker
    if shutil.disk_usage(ROOT).free >= FREE_FLOOR + reserve_bytes:
        return
    baseline = json.loads((ROOT / 'fullsource/docker-baseline.json').read_text())['existing_image_ids']
    records = sorted((json.loads(p.read_text()) for p in list(PROOFS.glob('*.json'))+list((PROOFS/'history').glob('*.json')) if p.name != 'current-import.json'),
                     key=lambda r: r.get('last_prepared_at', 0))
    prepared_path=ROOT/'fullsource/prepared-images.json'
    selected=json.loads(PLAN.read_bytes())['resolved']
    if prepared_path.exists():
        records += [{'task_id':task,'image_id':value['prepared_image_id']} for task,value in json.loads(prepared_path.read_text()).items()]
    for record in records:
        if shutil.disk_usage(ROOT).free >= FREE_FLOOR + reserve_bytes: break
        image_id = record['image_id']
        # Historical certificates survive pre-inference feasibility screening.
        # They cannot authorize deletion through the current sample inventory.
        if record['task_id'] not in selected:continue
        if record['task_id'] == protect_task or image_id in baseline: continue
        try: image = client.images.get(image_id)
        except docker.errors.ImageNotFound: continue
        if client.containers.list(all=True, filters={'ancestor': image_id}): continue
        original = selected[record['task_id']]
        prepared = execution_entry(record['task_id'],original)
        plan = original if image.id in {original['registry_digest'],original['config']['digest']} else prepared
        if image.id not in {plan['registry_digest'], plan['config']['digest']}:
            raise RuntimeError('Cache eviction image identity mismatch')
        for blob in plan['layers'] + [plan['config']]:
            if not (CACHE / 'blobs/sha256' / blob['digest'].split(':')[1]).is_file():
                raise RuntimeError('Cannot evict an image without its local recovery blob')
        client.images.remove(image_id, force=False)
        with (PROOFS / 'evictions.jsonl').open('a') as file:
            file.write(json.dumps({'task_id': record['task_id'], 'image_id': image_id,
                'registry_digest': plan['registry_digest'], 'local_recovery': True, 'timestamp': time.time()}) + '\n')
        reclaim_deleted_space()
    if shutil.disk_usage(ROOT).free < FREE_FLOOR + reserve_bytes:
        raise RuntimeError('Insufficient disk for offline image staging; no eligible owned image remains')
