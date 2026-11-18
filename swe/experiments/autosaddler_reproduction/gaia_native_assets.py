"""Pinned offline plugin SDK for the new host; no GAIA runtime is required."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from .fullsource_host import load

ROOT = Path(__file__).resolve().parents[2]
SDK_ROOT = ROOT / 'third_party/opencode-plugin-sdk-1.19.5-linux'
NATIVE_IMAGE_ID = load()['compiler_image_id']


def validate_sdk(root=None):
    root = Path(root or SDK_ROOT).resolve()
    manifest_path = root / 'file_manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != load()['sdk_manifest_sha256']:
        raise RuntimeError('Offline SDK manifest changed')
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest.items():
        path = root / relative
        if PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts or not path.resolve().is_relative_to(root):
            raise RuntimeError('Offline SDK asset path escaped its root')
        if isinstance(digest, dict):
            if not path.is_symlink() or path.readlink().as_posix() != digest['symlink']:
                raise RuntimeError('Pinned SDK executable link changed')
            continue
        if path.is_symlink():
            raise RuntimeError('Unexpected SDK link')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Offline SDK asset changed: ' + relative)
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    if actual != set(manifest) | {'file_manifest.json'}:
        raise RuntimeError('Unexpected offline SDK files')
    return root


def prepare_sdk_mounts(workspace, artifacts):
    sdk = validate_sdk()
    candidate_config = Path(workspace) / '.opencode'
    global_config = Path(artifacts) / 'native-global-config'
    for config in (candidate_config, global_config):
        config.mkdir(parents=True, exist_ok=True)
        for filename in ('package.json', 'package-lock.json'):
            destination = config / filename
            if destination.exists() and destination.read_bytes() != (sdk / filename).read_bytes():
                raise RuntimeError('Conflicting offline SDK metadata')
            if not destination.exists(): shutil.copy2(sdk / filename, destination)
        ignore = config / '.gitignore'
        if not ignore.exists(): ignore.write_bytes(b'node_modules\npackage.json\npackage-lock.json\n.gitignore\n')
    (Path(artifacts) / 'native-sdk-assets.json').write_text(json.dumps({
        'sdk_manifest_sha256': load()['sdk_manifest_sha256'], 'verified_files': len(json.loads((sdk/'file_manifest.json').read_text())),
        'native_image_id': NATIVE_IMAGE_ID, 'node_modules_read_only': True, 'package_scripts_executed': 0}, indent=2))
    return ['--mount', f'type=bind,src={global_config.resolve()},dst=/root/.config/opencode',
            '--mount', f'type=bind,src={sdk / "node_modules"},dst=/root/.config/opencode/node_modules,readonly',
            '--mount', f'type=bind,src={sdk / "node_modules"},dst=/workspace/.opencode/node_modules,readonly']
