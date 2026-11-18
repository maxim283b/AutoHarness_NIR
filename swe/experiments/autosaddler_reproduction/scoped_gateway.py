"""Trusted per-trial accounting identity, inaccessible to mutable target code."""
from contextlib import contextmanager
import secrets
from .go_gateway import GoGateway

_active = None
ROLES = {'target_search', 'target_final_baseline', 'target_final_optimized'}


class ScopedGoGateway(GoGateway):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_tokens = {}

    def authenticated(self, given):
        return super().authenticated(given) or given in self.target_tokens

    def bound_headers(self, given, headers):
        lower = {k.lower(): v for k, v in headers.items()}
        if secrets.compare_digest(given, self.token):
            if lower.get('x-autoharness-role') != 'optimizer':
                raise ValueError('Campaign key is restricted to the frozen optimizer')
            return lower
        bound = self.target_tokens.get(given)
        if bound is None:
            raise ValueError('Target trial scope has ended')
        # Supplied identity headers cannot select a role/task/run/candidate.
        # Purpose stays explicit, validated by the existing parent-role gateway.
        clean = {k: v for k, v in lower.items() if not k.startswith('x-autoharness-')}
        clean.update(bound)
        clean['x-autoharness-purpose'] = lower.get('x-autoharness-purpose', 'primary')
        clean['x-autoharness-context-policy'] = 'target-metering-v6'
        return clean

    def __enter__(self):
        global _active
        if _active is not None:
            raise RuntimeError('Only one campaign gateway is permitted')
        result = super().__enter__()
        from .go_network import GoNetwork
        self.private_network=GoNetwork(self.artifacts/'private-network',self.server.server_port)
        try:
            self.private_network.__enter__()
        except BaseException:
            super().__exit__();raise
        _active = self
        return result

    def __exit__(self, *args):
        global _active
        try:
            try:
                if hasattr(self,'private_network'):self.private_network.__exit__(*args)
            finally:
                super().__exit__(*args)
        finally:
            self.target_tokens.clear()
            _active = None


@contextmanager
def target_scope(run_id, role, task_id, candidate, benchmark='swe_bench_pro'):
    if _active is None or role not in ROLES:
        raise RuntimeError('An admitted gateway and exact target role are required')
    owner = _active
    token = secrets.token_urlsafe(32)
    bound = {'x-autoharness-run': run_id, 'x-autoharness-role': role,
             'x-autoharness-benchmark': benchmark, 'x-autoharness-task': task_id,
             'x-autoharness-candidate': candidate}
    with owner.lock:
        owner.target_tokens[token] = bound
    try:
        yield token
    finally:
        with owner.lock:
            owner.target_tokens.pop(token, None)
