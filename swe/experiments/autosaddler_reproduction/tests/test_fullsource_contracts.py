import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from experiments.autosaddler_reproduction.source_build import source_digest
from experiments.autosaddler_reproduction.opencode_runtime import candidate_hash
from experiments.autosaddler_reproduction.swe_adapter import score_official_output, score_official_artifacts
from experiments.autosaddler_reproduction.gaia_optimizer_supervisor import invoke
from experiments.autosaddler_reproduction.native_completion import run_native


def test_source_bytes_and_executable_modes_both_identify_candidate(tmp_path):
    (tmp_path/'opencode-src').mkdir()
    (tmp_path/'instructions.md').write_text('original')
    (tmp_path/'opencode-src/package.json').write_text('{}')
    file=tmp_path/'opencode-src/tool.ts';file.write_text('export const value=1')
    file.chmod(0o644);original=source_digest(tmp_path);harness=candidate_hash(tmp_path)
    file.chmod(0o755)
    assert source_digest(tmp_path)!=original and candidate_hash(tmp_path)!=harness
    file.chmod(0o644);assert source_digest(tmp_path)==original
    (tmp_path/'instructions.md').write_text('different instructions')
    assert source_digest(tmp_path)==original and candidate_hash(tmp_path)!=harness


def test_official_score_handles_absent_names_and_task_compilation_failure():
    row={'fail_to_pass':"['required']",'pass_to_pass':'[]'}
    assert score_official_output(row,{'tests':[{'name':'required','status':'PASSED'}]})['score']==1
    assert score_official_output(row,{'tests':[{'name':'another dynamic test','status':'PASSED'}]})['score']==0
    with pytest.raises(RuntimeError,match='no tests'):
        score_official_output(row,{'tests':[]})
    assert score_official_output(row,{'tests':[]},task_build_failed=True)['score']==0
    with pytest.raises(RuntimeError,match='conflicting'):
        score_official_output(row,{'tests':[{'name':'required','status':'PASSED'},{'name':'required','status':'FAILED'}]})


def test_actual_optimizer_child_is_stopped_and_reaped_on_cancellation(tmp_path):
    cancelled=threading.Event()
    timer=threading.Timer(.2,cancelled.set);timer.start()
    try:
        with pytest.raises(InterruptedError):
            invoke([sys.executable,'-c','import time; time.sleep(60)'],os.environ.copy(),
                tmp_path/'never-produced-result.json',tmp_path,10,cancelled)
    finally:timer.cancel()
    status=json.loads((tmp_path/'optimizer_worker_status.json').read_text())
    assert status['forced_worker_stop'] and status['resource_cleanup']['confirmed_absent']
    assert not status['complete'] and not status['valid_transport']
    with pytest.raises(ProcessLookupError):os.kill(status['worker_pid'],0)


def test_collection_failure_of_task_module_differs_from_missing_external_dependency(tmp_path):
    row={'repo':'ansible/ansible','instance_id':'task','fail_to_pass':"['required']",'pass_to_pass':'[]'}
    logs=tmp_path/'task';logs.mkdir();stdout=logs/'candidate_stdout.log'
    stdout.write_text("E ImportError: cannot import name 'icx_ping' from 'ansible.modules.network.icx' (/app/lib/ansible/modules/network/icx/__init__.py)")
    assert score_official_artifacts(row,{'tests':[]},tmp_path)['score']==0
    stdout.write_text("E ModuleNotFoundError: No module named 'numpy'")
    with pytest.raises(RuntimeError,match='no tests'):
        score_official_artifacts(row,{'tests':[]},tmp_path)
    row['repo']='flipt-io/flipt'
    stdout.write_text('--- FAIL: TestLoad (0.07s)\nFAIL\n')
    assert score_official_artifacts(row,{'tests':[]},tmp_path)['score']==0
    stdout.write_text('go: missing-module: network is unreachable\n')
    with pytest.raises(RuntimeError,match='no tests'):
        score_official_artifacts(row,{'tests':[]},tmp_path)
    row['patch']='diff --git a/go.sum b/go.sum\n'
    stdout.write_text('missing go.sum entry for module providing package example/module\n')
    assert score_official_artifacts(row,{'tests':[]},tmp_path)['score']==0
    stdout.write_text('go: module: Get "https://proxy.golang.org/module": dial tcp: network is unreachable\n')
    with pytest.raises(RuntimeError,match='dependency'):
        score_official_artifacts(row,{'tests':[{'name':'another','status':'PASSED'}]},tmp_path)


def test_native_root_completion_ignores_subagent_and_initial_idle(tmp_path):
    def event(session,status):
        return json.dumps({'type':'autoharness_session_status','sessionID':session,'status':status})
    script='\n'.join([
        'import time',
        f'print({event("root","busy")!r},flush=True)',
        f'print({json.dumps({"type":"step_start","sessionID":"root"})!r},flush=True)',
        f'print({event("child","busy")!r},flush=True)',
        f'print({event("child","idle")!r},flush=True)',
        'time.sleep(.2)',
        f'print({event("root","idle")!r},flush=True)',
        'time.sleep(10)',
    ])
    result=run_native([sys.executable,'-c',script],os.environ.copy(),tmp_path/'completed.jsonl',2,idle_grace=.05)
    assert result['completion_reason']=='native_root_idle' and not result['timed_out']
    assert .2<=result['elapsed_seconds']<2 and result['native_root_session_id']=='root'
    assert result['native_root_busy_observed'] and result['native_root_idle_observed']
    script=f'import time;print({event("root","idle")!r},flush=True);time.sleep(10)'
    result=run_native([sys.executable,'-c',script],os.environ.copy(),tmp_path/'initial-idle.jsonl',.2,idle_grace=.05)
    assert result['completion_reason']=='deadline' and result['timed_out']
    script='import sys\nwhile True: sys.stdout.write("x"*512+"\\n");sys.stdout.flush()'
    result=run_native([sys.executable,'-c',script],os.environ.copy(),tmp_path/'continuous-output.jsonl',.2,idle_grace=.05)
    assert result['completion_reason']=='deadline' and result['elapsed_seconds']<2
