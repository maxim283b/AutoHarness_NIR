"""One isolated optimizer child per logical attempt, with parent-owned cleanup."""
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import hashlib
import sys

from .gaia_owned_resources import OWNER_ENV, JOURNAL_ENV, cleanup_owner

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
MODELS = {'opencode-go/kimi-k2.7-code', 'opencode-go/glm-5.3'}


def worker_environment():
    from .fullsource_host import checked_file
    if checked_file('python').resolve() != PYTHON.resolve():
        raise RuntimeError('Optimizer interpreter changed before child launch')
    allowed = {'PATH','SYSTEMROOT','WINDIR','COMSPEC','PATHEXT','TEMP','TMP','USERPROFILE','HOME',
        'APPDATA','LOCALAPPDATA','PROGRAMDATA','HOMEDRIVE','HOMEPATH','NUMBER_OF_PROCESSORS',
        'PROCESSOR_ARCHITECTURE','HTTP_PROXY','HTTPS_PROXY','NO_PROXY','SSL_CERT_FILE',
        'GO_GATEWAY_URL','OPENCODE_GO_API_KEY','ACP_GATEWAY_TOKEN','AUTOHARNESS_DOCKER_NETWORK'}
    env = {k:v for k,v in os.environ.items() if k.upper() in allowed}
    env.update(PYTHONPATH=str(ROOT),PYTHONUTF8='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    return env


def invoke(command, env, result_path, artifacts, timeout, cancellation=None, expected_model=None):
    """No retries. A score/output is usable only after verified owner cleanup."""
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not 0 < timeout < float('inf'):
        raise ValueError('Optimizer timeout must be finite and positive')
    artifacts, result_path = Path(artifacts).resolve(), Path(result_path).resolve()
    if result_path.exists():
        raise RuntimeError('Optimizer worker result already exists; no redispatch')
    owner=uuid.uuid4().hex
    journal=artifacts/('optimizer_owned_resources-'+owner+'.jsonl')
    env={**env,OWNER_ENV:owner,JOURNAL_ENV:str(journal)}
    metadata=dict(worker_id=owner,valid_transport=False,complete=False,automatic_worker_retries=0,
                  resource_owner=owner,resource_journal=str(journal))
    process=None
    failure=None
    result=None
    deadline=time.monotonic()+timeout
    try:
        with (artifacts/'optimizer_worker.stdout.log').open('xb') as stdout, (artifacts/'optimizer_worker.stderr.log').open('xb') as stderr:
            if cancellation is not None and cancellation.is_set():
                raise InterruptedError('Optimizer attempt cancelled before launch')
            options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,**options)
            metadata['worker_pid']=process.pid
            while process.poll() is None:
                if cancellation is not None and cancellation.is_set():
                    raise InterruptedError('Optimizer attempt cancelled')
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise subprocess.TimeoutExpired(command,timeout)
                try:
                    process.wait(timeout=min(remaining,0.1))
                except subprocess.TimeoutExpired:
                    pass
            metadata['returncode']=process.returncode
        if process.returncode or not result_path.is_file():
            raise RuntimeError('Optimizer child failed or result missing')
        result=json.loads(result_path.read_bytes())
        if not isinstance(result,dict) or result.get('worker_id')!=owner or result.get('complete') is not True:
            raise ValueError('Optimizer child result identity/completion mismatch')
        if not isinstance(result.get('raw_response'),str) or not isinstance(result.get('usage'),list) or not result['usage']:
            raise ValueError('Optimizer child result schema mismatch')
        from autosaddler.v2.prompting.models import Usage
        usage=tuple(Usage(**row) for row in result['usage'])
        if expected_model not in MODELS or any(u.usage_incomplete or u.role!='optimizer' or u.model not in {expected_model,expected_model.split('/',1)[1]} for u in usage):
            raise ValueError('Optimizer child usage is incomplete or routed incorrectly')
    except BaseException as exc:
        failure=exc
        metadata.update(error_type=type(exc).__name__,usage_reconciliation_required=process is not None)
        raise
    finally:
        cleanup_error=None
        try:
            if process is not None and process.poll() is None:
                if os.name=='nt':
                    stopped=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=20)
                    metadata['process_tree_stop_returncode']=stopped.returncode
                    process.wait(timeout=10)
                    if process.poll() is None:
                        raise RuntimeError('Optimizer child process remains alive')
                else:
                    import signal
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=10)
                metadata['forced_worker_stop']=True
        except Exception as exc:
            cleanup_error=exc
            metadata['process_cleanup_error_type']=type(exc).__name__
        try:
            metadata['resource_cleanup']=cleanup_owner(owner,journal)
            if metadata['resource_cleanup'].get('confirmed_absent') is not True:
                raise RuntimeError('Optimizer owner cleanup not verified')
        except Exception as exc:
            cleanup_error=exc
            metadata['resource_cleanup_error_type']=type(exc).__name__
        if failure is None and cleanup_error is None and result is not None:
            metadata.update(complete=True,valid_transport=True)
        if cleanup_error is not None:
            metadata['usage_reconciliation_required']=True
        (artifacts/'optimizer_worker_status.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        if cleanup_error is not None and failure is None:
            raise RuntimeError('Optimizer child cleanup failed; transport unusable') from cleanup_error
    from autosaddler.v2.providers.base import TransportOutcome
    return TransportOutcome(raw_response=result['raw_response'],usage=usage)


def run_session(transport,session,timeout,cancellation):
    artifacts=Path(session.trace_dir or session.workspace/'.autosaddler/acp-trace')
    artifacts.mkdir(parents=True,exist_ok=True)
    payload={}
    for field in dataclasses.fields(session):
        value=getattr(session,field.name)
        payload[field.name]=str(value) if isinstance(value,Path) else value
    input_path=artifacts/'optimizer_worker_input.json'
    with input_path.open('x',encoding='utf-8') as file:
        json.dump(dict(schema_version='gaia-optimizer-worker/v1',model=transport.model,run_id=transport.run_id,
            output_repair_attempts=transport.output_repair_attempts,timeout_seconds=timeout,session=payload),file)
    result_path=artifacts/'optimizer_worker_result.json'
    return invoke([str(PYTHON),'-m','experiments.autosaddler_reproduction.gaia_optimizer_supervisor',
        '--input',str(input_path.resolve()),'--result',str(result_path.resolve())],
        worker_environment(),result_path,artifacts,timeout,cancellation,expected_model=transport.model)


def child_main():
    import argparse
    from autosaddler.v2.core.serde import record
    from autosaddler.v2.providers.workspace_renderer import RenderedSession
    from .optimizer_provider import Transport
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    if args.result.exists():
        raise RuntimeError('Optimizer result exists')
    value=json.loads(args.input.read_bytes())
    if value.get('schema_version')!='gaia-optimizer-worker/v1' or value.get('model') not in MODELS:
        raise ValueError('Optimizer worker requires exact declared Go model')
    row=value['session']
    for name in ('workspace','instruction_path','skill_directory','output_schema_path','output_path','trace_dir'):
        if row.get(name) is not None:
            row[name]=Path(row[name])
    session=RenderedSession(**row)
    outcome=Transport(value['model'],value['run_id'],value['output_repair_attempts'])._run(session,value['timeout_seconds'])
    with args.result.open('x',encoding='utf-8') as file:
        json.dump(dict(worker_id=os.environ[OWNER_ENV],complete=True,raw_response=outcome.raw_response,
                       usage=[record(u) for u in outcome.usage]),file)


if __name__=='__main__':
    child_main()
