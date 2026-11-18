"""Host assets fixed before the new full-source SWE campaign."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'fullsource/host-lock.json'


def load():
    value = json.loads(LOCK.read_text(encoding='utf-8'))
    if value.get('schema_version') != 'swe-fullsource-host/v1':
        raise RuntimeError('Fresh full-source host lock is required')
    return value


def checked_file(name):
    value = load()[name]
    path = Path(value['path'])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != value['sha256']:
        raise RuntimeError('Frozen host asset changed: ' + name)
    return path
