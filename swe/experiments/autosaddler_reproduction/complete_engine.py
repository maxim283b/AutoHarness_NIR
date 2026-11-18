"""Completion policy: technical session failures must not consume an iteration."""
from autosaddler.v2.core.engine import AutoSaddlerEngine, SessionRetriesExhausted


class IncompleteOptimizationStage(RuntimeError):
    """A technical stage is unresolved; final test evaluation is forbidden."""


def require_complete_search(store, expected_iterations):
    events=store.events()
    completed={e.payload['iteration'] for e in events if e.event_type=='IterationCompleted'}
    if completed != set(range(expected_iterations)):
        raise IncompleteOptimizationStage(f'Expected {expected_iterations} completed iterations, observed {sorted(completed)}')
    if any(e.event_type=='DeferredWorkAbandoned' for e in events):
        raise IncompleteOptimizationStage('A required reflection was abandoned')
    terminal={}
    for event in events:
        if event.event_type in {'SessionStarted','SessionCompleted','SessionFailed'}:
            logical=event.payload.get('logical_operation_id')
            if logical: terminal[logical]=event.event_type
    unresolved=[op for op,kind in terminal.items() if kind!='SessionCompleted']
    if unresolved:
        raise IncompleteOptimizationStage('Unresolved optimizer sessions: '+', '.join(unresolved))
    return dict(passed=True,completed_iterations=len(completed),unresolved_sessions=0,abandoned_reflections=0)


class CompleteOptimizationEngine(AutoSaddlerEngine):
    async def _ensure_session(self, **kwargs):
        try:
            return await super()._ensure_session(**kwargs)
        except SessionRetriesExhausted as error:
            # Use a different exception type so upstream cannot convert it into
            # MutationRejected/IterationCompleted or DeferredWorkAbandoned.
            raise IncompleteOptimizationStage(str(error)) from error

    def _select_best(self):
        require_complete_search(self.store,self.policies.budget.max_iterations)
        return super()._select_best()


def replace_engine(runtime):
    from dataclasses import replace
    old=runtime.engine
    engine=CompleteOptimizationEngine(store=old.store,scenario=old.scenario,provider=old.provider,policies=old.policies,
        diagnosis_patch_timeout_seconds=old.diagnosis_patch_timeout_seconds,selection_timeout_seconds=old.selection_timeout_seconds,
        reflection_timeout_seconds=old.reflection_timeout_seconds,session_retries=old.session_retries,
        session_retry_backoff_seconds=old.session_retry_backoff_seconds)
    return replace(runtime,engine=engine)
