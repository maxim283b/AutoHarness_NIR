"""Preserve effective working H0 defaults, with an explicitly mutable opt-out."""
import json
from pathlib import Path

SCHEMA = 'opencode-full-source-runtime-policy/v1'
RELATIVE_PATH = '.opencode/runtime-policy.json'
DEFAULTS = {'schema_version': SCHEMA, 'use_baseline_context_policy': True}
DESCRIPTION = {
    'policy_path': RELATIVE_PATH,
    'absent_policy_means': 'Preserve original effective Go H0 runtime defaults: native compaction auto/prune, reserved8192, input32000, complete tool output saved with excerpts',
    'mutable_opt_out': {'schema_version': SCHEMA, 'use_baseline_context_policy': False},
    'when_opted_out': 'Native opencode.json, target source and candidate plugins fully control target context/loop. Accounting-only hook and external model/quota/budget/verifier limits remain trusted.',
    'reason': 'The historical raw seed config disables compaction, but the working H0 runtime always enabled it through a Go overlay. Full-source mutation must not accidentally remove that working default.',
}

def apply(config, model, candidate):
    path = Path(candidate) / RELATIVE_PATH
    policy = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(DEFAULTS)
    if (not isinstance(policy,dict) or set(policy) != set(DEFAULTS)
            or policy.get('schema_version') != SCHEMA
            or type(policy.get('use_baseline_context_policy')) is not bool):
        raise ValueError('Invalid mutable target runtime-policy contract')
    if policy['use_baseline_context_policy']:
        from .context_policy import configure
        configure(config, model, 'file:///opt/autoharness/context-policy.mjs', '/logs/agent/context-outputs')
    return policy
