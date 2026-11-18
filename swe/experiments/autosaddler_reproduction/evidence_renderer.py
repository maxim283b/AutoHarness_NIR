"""Validate actual evidence bytes before a native optimizer can consume them."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from autosaddler.v2.providers.workspace_renderer import WorkspaceRenderer


class EvidenceRenderer(WorkspaceRenderer):
    def render(self, spec, workspace, **kwargs):
        rendered = super().render(spec, workspace, **kwargs)
        root = rendered.workspace.resolve()
        manifest = json.loads((root / '.autosaddler/evidence_index.json').read_text())
        checked = []
        for relative, digest in manifest['files'].items():
            path = root / relative
            if not path.resolve().is_relative_to(root) or not path.is_file() or path.is_symlink():
                raise RuntimeError('Required current optimizer evidence is missing/unsafe: ' + relative)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Current optimizer evidence bytes differ: ' + relative)
            checked.append({'path': relative, 'sha256': digest, 'bytes': path.stat().st_size})
        if spec.kind == 'diagnose_patch' and '.autosaddler/training_evidence.json' not in manifest['files']:
            raise RuntimeError('Diagnosis cannot start without complete current train evidence')
        if rendered.trace_dir:
            (rendered.trace_dir / 'evidence_delivery.json').write_text(json.dumps({
                'kind': spec.kind, 'checked_files': checked, 'all_present_sha_verified': True}, indent=2))
        return replace(rendered, task_prompt=rendered.task_prompt +
            '\nCurrent evidence files exist and were hash-verified before this session. '
            'Read .autosaddler/evidence_index.json, then read the listed evidence and relevant history. '
            'Return its exact files map as evidence_receipt. Report unavailable evidence as an error; '
            'never invent a missing file or fabricate an outcome.')


def validate_receipt(session, value):
    expected = json.loads((session.workspace / '.autosaddler/evidence_index.json').read_text())['files']
    if value.get('evidence_receipt') != expected:
        raise ValueError('evidence_receipt must match every delivered current evidence file SHA256')
    # Validate hashes again after tool use; candidate edits cannot rewrite evidence.
    for relative, digest in expected.items():
        if hashlib.sha256((session.workspace / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Delivered evidence changed during optimization')
