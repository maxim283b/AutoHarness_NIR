import json
from types import SimpleNamespace

import pytest

from experiments.autosaddler_reproduction.swe_go_adapter import runtime_config
from experiments.autosaddler_reproduction.swe_go_eval import campaign, freeze


def test_go_route_and_native_context_policy(tmp_path, monkeypatch):
    monkeypatch.setenv('GO_GATEWAY_URL','http://fixture')
    (tmp_path/'opencode.json').write_text(json.dumps({'provider':{'unrelated':{}},'permission':{'task':'deny'}}))
    config=runtime_config(tmp_path,'opencode-go/qwen3.7-plus','swe-run','target_final_baseline','synthetic')
    assert list(config['provider'])==['opencode-go']
    provider=config['provider']['opencode-go']
    assert provider['options']['baseURL']=='http://fixture/anthropic/v1'
    assert provider['options']['headers']['X-Autoharness-Benchmark']=='swe_bench_pro'
    assert provider['models']['qwen3.7-plus']['limit']['input']==32000
    assert provider['models']['qwen3.7-plus']['limit']['output']==8192
    assert config['compaction']=={'auto':True,'prune':True,'reserved':8192}
    assert config['instructions']==['/opt/autoharness/harness/instructions.md']
    assert config['permission']['task']=='deny'
    monkeypatch.delenv('GO_GATEWAY_URL')
    with pytest.raises(RuntimeError,match='guarded native Go'): runtime_config(tmp_path,'opencode-go/qwen3.7-plus','r','target_final_baseline','t')


def test_pending_and_historical_unknown_are_separate():
    rows=[dict(id='past',run_id='old',role='target_final_baseline',reservation=.2),dict(id='live',run_id='new',role='target_final_baseline',reservation=.3)]
    gateway=SimpleNamespace(rows=rows,active_requests={'live'},accounting_summary=lambda:{})
    summary=campaign(gateway,'new')['campaign_all_attempts']
    assert summary['requests']==2
    assert summary['unknown_requests']==1 and summary['unknown_reserve']==.2
    assert summary['pending_requests']==1 and summary['pending_reserve']==.3
    assert summary['total_tokens'] is None


def test_dispatched_run_cannot_be_refrozen(tmp_path):
    (tmp_path/'gateway').mkdir();(tmp_path/'gateway/requests.jsonl').touch()
    with pytest.raises(RuntimeError,match='Cannot refreeze'): freeze(tmp_path)
