"""Frozen native Go generation in official SWE V1 images, then fresh grading."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import uuid
import gzip

from .acp_gateway import context_headers
from .context_policy import configure, PLUGIN
from .go_gateway import provider_config, save
from .swe_adapter import ROOT, _archive_files, grade_patch, record_cli_usage
from .terminal_adapter import cleanup_command, git_stdin_compat_script, check_agent_errors
from .target_metering import PLUGIN as METERING_PLUGIN
from .native_completion import PLUGIN as LIFECYCLE_PLUGIN, PLUGIN_URL as LIFECYCLE_PLUGIN_URL, run_native


def runtime_config(candidate, model, run_id, role, task_id):
    if model != 'opencode-go/qwen3.7-plus' or not os.environ.get('GO_GATEWAY_URL'):
        raise RuntimeError('SWE generation requires the guarded native Go route')
    config = json.loads((Path(candidate) / 'opencode.json').read_text(encoding='utf-8'))
    config.update(model=model, instructions=['/opt/autoharness/harness/instructions.md'],
                  enabled_providers=['opencode-go'], share='disabled', autoupdate=False)
    config['provider'] = provider_config(model, context_headers(run_id, role, 'swe_bench_pro', task_id))
    from .target_policy import apply
    apply(config, model, candidate)
    from .target_metering import PLUGIN_URL
    config.setdefault('plugin', []).append(PLUGIN_URL)
    config['plugin'].append(LIFECYCLE_PLUGIN_URL)
    return config


def checked(container, command, **kwargs):
    result = container.exec_run(command, **kwargs)
    if result.exit_code:
        raise RuntimeError('SWE container command failed: '+str(command[:2])+' '+result.output.decode(errors='replace')[-1500:])
    return result.output


def generate(row, candidate, model, artifacts, run_id, role, image, *, timeout=1800, ledger=None, prompt_override=None, scoped_token=None):
    import docker
    artifacts=Path(artifacts); artifacts.mkdir(parents=True,exist_ok=True)
    config=runtime_config(candidate,model,run_id,role,row['instance_id'])
    client=docker.from_env(timeout=2400)
    container=None; scope=uuid.uuid4().hex; timed_out=False
    try:
        from .gaia_native_assets import prepare_sdk_mounts, SDK_ROOT
        sdk_workspace = artifacts / 'sdk-metadata-only'
        prepare_sdk_mounts(sdk_workspace, artifacts)
        volumes = {str((artifacts/'native-global-config').resolve()): {'bind':'/root/.config/opencode','mode':'rw'},
            str((SDK_ROOT/'node_modules').resolve()): {'bind':'/opt/autoharness/harness/.opencode/node_modules','mode':'ro'}}
        # Docker SDK's volumes map permits one destination per source. Mount
        # the global copy through mounts so both native resolution paths work.
        mounts = [docker.types.Mount('/root/.config/opencode/node_modules', str((SDK_ROOT/'node_modules').resolve()),
                                    type='bind', read_only=True)]
        # The image ID was verified and frozen before evaluation, never picked by score.
        cpu_limits={'cpuset_cpus':'0-1'} if row.get('repo')=='element-hq/element-web' else {}
        container=client.containers.run(image['image_id'],entrypoint='/bin/sh',command=['-c','sleep infinity'],
            detach=True,name='autoharness-swe-go-'+uuid.uuid4().hex[:12],mem_limit='6g',pids_limit=512,
            volumes=volumes,mounts=mounts,platform='linux/amd64',labels={'autoharness.swe-run':run_id},
            network=os.environ['AUTOHARNESS_DOCKER_NETWORK'],cap_drop=['NET_RAW'],**cpu_limits)
        container.reload()
        resources=container.attrs['HostConfig']
        if resources['CpusetCpus']!=cpu_limits.get('cpuset_cpus','') or resources['Memory']!=6*1024**3 or resources['PidsLimit']!=512:
            raise RuntimeError('Native target resource profile was not applied')
        save(artifacts/'generation_status.json',dict(status='setup',container=container.id,task_id=row['instance_id']))
        repo=checked(container,['git','rev-parse','--show-toplevel']).decode().strip()
        if repo != image['repo_dir']: raise RuntimeError('SWE repository location drift')
        checked(container,['git','reset','--hard',row['base_commit']],workdir=repo)
        checked(container,['mkdir','-p','/opt/autoharness/harness','/opt/autoharness/git-compat','/logs/agent'])
        from .source_build import build_candidate
        binary, build = build_candidate(candidate)
        save(artifacts/'candidate-build.json', build)
        if not scoped_token:
            raise RuntimeError('Full-source target requires an external per-trial Go identity')
        git=checked(container,['/bin/sh','-c','command -v git']).decode().strip()
        container.put_archive('/opt/autoharness',_archive_files([
            ('opencode',binary.read_bytes(),True),('context-policy.mjs',PLUGIN.encode(),False),
            ('target-metering.mjs',METERING_PLUGIN.encode(),False),
            ('target-lifecycle.mjs',LIFECYCLE_PLUGIN.encode(),False),
            ('git-compat/git',git_stdin_compat_script(git).encode(),True)]))
        installed_binary = checked(container,['sha256sum','/opt/autoharness/opencode']).decode().split()[0]
        if installed_binary != build['binary_sha256']:
            raise RuntimeError('Task container received a different candidate executable')
        files=[(p.relative_to(candidate).as_posix(),p.read_bytes(),False) for p in Path(candidate).rglob('*')
               if p.is_file() and '.git' not in p.parts and p.name!='opencode.json'
               and 'opencode-src' not in p.relative_to(candidate).parts and 'node_modules' not in p.relative_to(candidate).parts]
        files.append(('opencode.json',json.dumps(config).encode(),False))
        present = {name for name, _, _ in files}
        for name in ('package.json', 'package-lock.json'):
            if '.opencode/'+name not in present:
                files.append(('.opencode/'+name,(SDK_ROOT/name).read_bytes(),False))
        container.put_archive('/opt/autoharness/harness',_archive_files(files))
        prompt=prompt_override or (row['problem_statement']+'\n\nRequirements:\n'+str(row.get('requirements',''))+'\n\nInterface:\n'+str(row.get('interface','')))
        env={'OPENCODE_CONFIG':'/opt/autoharness/harness/opencode.json',
             'OPENCODE_CONFIG_DIR':'/opt/autoharness/harness/.opencode',
             'OPENCODE_DISABLE_AUTOUPDATE':'true','OPENCODE_DISABLE_MODELS_FETCH':'1',
             'OPENCODE_DISABLE_PROJECT_CONFIG':'1','OPENCODE_FAKE_VCS':'git',
             'XDG_DATA_HOME':'/logs/agent/opencode/xdg-data','XDG_STATE_HOME':'/logs/agent/opencode/xdg-state'}
        prefix=['docker','exec','-w',repo,'-e','OPENCODE_GO_API_KEY']
        for key,value in env.items(): prefix += ['-e',key+'='+value]
        # Same CLI permission mode as the frozen TB integration. Candidate tool rules remain in config.
        command='export PATH="/opt/autoharness/git-compat:$PATH"; echo $$ > /logs/agent/autoharness_agent.pid; exec /opt/autoharness/opencode --model='+shlex.quote(model)+' run --format=json --thinking --dangerously-skip-permissions -- '+shlex.quote(prompt)
        save(artifacts/'generation_status.json',dict(status='agent_running',container=container.id,task_id=row['instance_id'],timeout_seconds=timeout))
        completion=run_native(prefix+['-e','AUTOHARNESS_AGENT_SCOPE='+scope,container.name,'/bin/sh','-c',command],
            {**os.environ,'OPENCODE_GO_API_KEY':scoped_token},artifacts/'opencode.jsonl',timeout)
        timed_out=completion['timed_out'];returncode=completion['returncode']
        # Kill the command tree before reading the patch or invoking any verifier.
        checked(container,['/bin/sh','-c',cleanup_command(scope)])
        container.reload()
        if container.attrs['State']['OOMKilled']:
            raise RuntimeError('Native target suffered OOM; technical NOSCORE')
        cleanup=json.loads(checked(container,['cat','/logs/agent/autoharness_cleanup.json']))
        if not cleanup.get('cleanup_ok'): raise RuntimeError('Unverified agent cleanup')
        trace=(artifacts/'opencode.jsonl').read_text(encoding='utf-8',errors='replace')
        check_agent_errors(trace)
        if returncode!=0 and completion['completion_reason']=='process_exit':
            raise RuntimeError('Native OpenCode failed; this attempt has no score')
        session_ids=[]
        for line in trace.splitlines():
            try: event=json.loads(line)
            except ValueError: continue
            if event.get('sessionID'): session_ids.append(event['sessionID'])
        if session_ids:
            exported=container.exec_run(['/opt/autoharness/opencode','export',session_ids[-1]],environment={**env,'OPENCODE_GO_API_KEY':scoped_token})
            (artifacts/'native_export.json').write_bytes(exported.output)
        if ledger is not None:
            record_cli_usage(trace,ledger,run_id,role,benchmark='swe_bench_pro',task_id=row['instance_id'])
        checked(container,['git','add','--intent-to-add','.'],workdir=repo)
        patch=checked(container,['git','diff','--binary',row['base_commit']],workdir=repo).decode('utf-8')
        (artifacts/'patch.diff').write_bytes(patch.encode())
        try:
            stream,_=container.get_archive('/logs/agent')
            with gzip.open(artifacts/'agent_logs.tar.gz','wb') as output:
                for chunk in stream: output.write(chunk)
        finally:
            # Even reparented background processes cannot affect fresh grading.
            container.remove(force=True); container=None
        meta=dict(agent_timed_out=timed_out,agent_cleanup_verified=True,generation_container_removed_before_grading=True,
                  patch_sha256=__import__('hashlib').sha256(patch.encode()).hexdigest(),
                  candidate_binary_sha256=build['binary_sha256'],candidate_source_sha256=build['source_sha256'])
        meta['installed_candidate_binary_sha256'] = installed_binary
        meta.update(generation_cpuset_cpus=resources['CpusetCpus'],generation_memory_limit_bytes=resources['Memory'],generation_pids_limit=resources['PidsLimit'])
        meta['native_completion']=completion
        save(artifacts/'generation_status.json',dict(status='complete',**meta))
        return patch,meta
    finally:
        if container is not None: container.remove(force=True)
        client.close()


def evaluate(row,candidate,model,artifacts,run_id,role,image,timeout=1800,*,ledger=None,scoped_token=None):
    patch,meta=generate(row,candidate,model,Path(artifacts)/'generation',run_id,role,image,timeout=timeout,ledger=ledger,scoped_token=scoped_token)
    result=grade_patch(row,patch,Path(artifacts)/'grading',image=image)
    result.update(meta)
    result['grading_image_id']=image['image_id']
    return result
