"""Predeclared deterministic sampling independent of model outcomes."""
import hashlib
from pathlib import Path


def manifest_path(settings, benchmark):
    root = Path(__file__).resolve().parents[2]
    selected = settings.get('fullsource_manifest')
    if selected:
        path = (root / selected).resolve()
        if benchmark != 'swe_bench_pro' or not path.is_relative_to(root):
            raise RuntimeError('Invalid full-source benchmark manifest')
        return path
    return Path(__file__).parent / 'manifests' / f'{benchmark}.json'


def select_ids(ids, count=None, seed=None, benchmark="", split=""):
    values = list(ids)
    if seed is not None:
        values.sort(key=lambda task_id: hashlib.sha256(f"{seed}:{benchmark}:{split}:{task_id}".encode()).hexdigest())
    return values if count is None else values[:count]
