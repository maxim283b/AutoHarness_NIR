"""Compile every distinct candidate in an isolated offline, owned container."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import uuid
import stat

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).parent
from .fullsource_host import load as host_lock
IMAGE = host_lock()['compiler_image_id']
IMAGE_CONFIG = host_lock()['compiler_config_id']
SOURCE_COMMIT = '810c0eba1a7348036f8bdf61e6632d0c36c6876c'
METADATA_SHA = 'ddf3427a1a8839c1703bae823008690f0ba76d12d60af8cc12e3ce73d6c6b5f6'
BUILD_TIMEOUT = 600
_lock = threading.Lock()

class CandidateCompilationError(RuntimeError):
    """Invalid candidate build with confirmed compiler resource cleanup."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_files(workspace):
    workspace = Path(workspace).resolve()
    result = {}
    for name in ('instructions.md', 'opencode.json', '.opencode', 'opencode-src'):
        root = workspace / name
        paths = [root] if root.is_file() else root.rglob('*') if root.is_dir() else []
        for path in paths:
            relative = path.relative_to(workspace)
            if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                raise RuntimeError('Candidate symlink/escape forbidden')
            if 'node_modules' in relative.parts:
                continue
            if path.is_file():
                result[relative.as_posix()] = {'sha256': sha(path),
                    'executable': bool(path.stat().st_mode & stat.S_IXUSR)}
    if 'instructions.md' not in result or 'opencode-src/package.json' not in result:
        raise RuntimeError('Incomplete full-source candidate')
    return dict(sorted(result.items()))


def source_digest(workspace):
    files = {p: d for p, d in candidate_files(workspace).items() if p.startswith('opencode-src/')}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def docker(*args, **kwargs):
    return subprocess.run(['docker', *map(str, args)], check=True, capture_output=True,
                          timeout=kwargs.pop('timeout', 30), **kwargs)


def inspect_owned(name, owner):
    result = subprocess.run(['docker', 'inspect', name, '--format',
        '{{json .Id}} {{json .Image}} {{json (index .Config.Labels "autoharness.source-owner")}}'],
        capture_output=True, timeout=30)
    if result.returncode:
        if b'no such' in result.stderr.lower() or b'not found' in result.stderr.lower():
            return None
        raise RuntimeError('Cannot independently inspect source-build resource')
    values = result.stdout.decode().strip().split(' ')
    if len(values) != 3 or json.loads(values[1]) != IMAGE or json.loads(values[2]) != owner:
        raise RuntimeError('Build container ownership/image mismatch')
    return json.loads(values[0])


@contextmanager
def owned_builder(source, directory):
    from .gaia_owned_resources import register_container, container_labels
    owner = uuid.uuid4().hex
    name = 'autoharness-source-build-' + owner[:12]
    record = {'name': name, 'owner': owner, 'image': IMAGE, 'network': 'none',
              'source_read_only': True, 'model_calls': 0, 'created': False}
    path = directory / ('owned-build-' + owner + '.json')
    path.write_text(json.dumps(record, indent=2))
    # The optimizer parent can discover this exact container after worker death.
    # Existing owner-aware cleanup never relies on the worker running finally.
    worker_resource = register_container(name, directory, kind='source-build')
    try:
        docker('create', '--name', name, '--network', 'none', '--pull', 'never', *container_labels(worker_resource),
               '--label', 'autoharness.source-owner=' + owner,
               '--label', 'autoharness.experiment=go-terminal-verifier-ready-v7-20261108',
               '--mount', f'type=bind,src={source.resolve()},dst=/candidate,readonly',
               '--mount', f'type=bind,src={PACKAGE / "source_build_linux.py"},dst=/autoharness-build.py,readonly',
               IMAGE, 'python3', '/autoharness-build.py', '/candidate')
        record['created'] = True
        path.write_text(json.dumps(record, indent=2))
        inspect_owned(name, owner)
        yield name
    finally:
        if inspect_owned(name, owner) is not None:
            docker('rm', '-f', name)
        if inspect_owned(name, owner) is not None:
            raise RuntimeError('Source compiler cleanup unverified')
        record['cleanup_verified'] = True
        path.write_text(json.dumps(record, indent=2))


def build_candidate(workspace):
    workspace = Path(workspace).resolve()
    digest = source_digest(workspace)
    wrapper_digest = sha(PACKAGE / 'source_build_linux.py')
    driver_digest = sha(Path(__file__))
    build_key = hashlib.sha256((digest + IMAGE + wrapper_digest + driver_digest).encode()).hexdigest()
    directory = ROOT / 'results/autosaddler_reproduction/source_builds' / build_key
    binary = directory / 'opencode'
    proof = directory / 'build.json'
    with _lock:
        if proof.exists():
            value = json.loads(proof.read_text())
            if (value.get('source_sha256') != digest or value.get('builder_image') != IMAGE
                    or value.get('build_wrapper_sha256') != wrapper_digest
                    or value.get('build_driver_sha256') != driver_digest
                    or value.get('binary_sha256') != sha(binary)
                    or value.get('cleanup_verified') is not True):
                raise RuntimeError('Cached candidate compiler provenance changed')
            return binary, value
        directory.mkdir(parents=True, exist_ok=True)
        # A failed compile is never automatically retried or turned into score0.
        marker = directory / 'started.json'
        if marker.exists():
            log = directory / 'compile.log'
            details = log.read_text(encoding='utf-8', errors='replace')[-12000:] if log.exists() else 'No complete compiler log'
            raise CandidateCompilationError('This exact source/build attempt already failed; change the source rather than restarting it.\n' + details)
        with marker.open('x') as file:
            json.dump({'source_sha256': digest, 'builder_image': IMAGE}, file)
        with owned_builder(workspace / 'opencode-src', directory) as name:
            with (directory / 'compile.log').open('xb') as output:
                completed = subprocess.run(['docker', 'start', '-a', name], stdout=output,
                    stderr=subprocess.STDOUT, timeout=BUILD_TIMEOUT)
            state = json.loads(docker('inspect', name, '--format', '{{json .State}}').stdout)
            if completed.returncode or state.get('ExitCode') != 0:
                details = (directory / 'compile.log').read_text(encoding='utf-8', errors='replace')[-12000:]
                raise CandidateCompilationError('Candidate compilation failed: technical NOSCORE, never task score0.\n' + details)
            docker('cp', name + ':/opt/opencode-source/packages/opencode/dist/opencode-linux-x64/bin/opencode', binary)
        if source_digest(workspace) != digest:
            raise RuntimeError('Candidate source changed while compiling')
        value = {'source_sha256': digest, 'builder_image': IMAGE, 'binary_sha256': sha(binary),
                 'compile_exit': 0, 'cleanup_verified': True, 'network': 'none',
                 'build_wrapper_sha256': wrapper_digest, 'build_driver_sha256': driver_digest, 'model_calls': 0}
        proof.write_text(json.dumps(value, indent=2))
        return binary, value
