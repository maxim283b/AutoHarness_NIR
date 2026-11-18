"""Observe OpenCode's own root-session completion without changing its loop."""
import json
import subprocess
import time

PLUGIN_URL='file:///opt/autoharness/target-lifecycle.mjs'
PLUGIN='''export default async () => ({
  event: async ({ event }) => {
    if (event.type !== "session.status") return;
    const p = event.properties;
    if (typeof p?.sessionID !== "string" || typeof p?.status?.type !== "string") return;
    console.log(JSON.stringify({type:"autoharness_session_status", sessionID:p.sessionID, status:p.status.type}));
  },
});
'''


def run_native(command, env, output_path, timeout, *, idle_grace=2.0):
    """Wait for exit, deadline, or busy→idle of the root actually producing output.

    Subagent idle and initial idle are insufficient. After a native idle event,
    give the CLI time to exit itself, then close its host exec client; the caller
    still performs the same verified in-container process-tree cleanup.
    """
    root=None;busy=set();idle=set();idle_at=None;reason=None
    started=time.monotonic()
    with output_path.open('wb') as output, output_path.open('r',encoding='utf-8',errors='replace') as trace:
        process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                position=trace.tell();line=trace.readline(2*1024**2);scanned=0
                while line.endswith('\n') and scanned<2*1024**2:
                    scanned+=len(line)
                    try:event=json.loads(line)
                    except ValueError:event={}
                    session=event.get('sessionID')
                    if session and event.get('type')=='autoharness_session_status':
                        if event.get('status') in {'busy','retry'}:busy.add(session);idle.discard(session)
                        elif event.get('status')=='idle' and session in busy:idle.add(session)
                    elif session and event.get('type') in {'step_start','text','tool_use'} and root is None:
                        root=session
                    position=trace.tell();line=trace.readline(2*1024**2)
                trace.seek(position)  # Keep an incomplete JSON line for the next poll.
                now=time.monotonic()
                idle_at=(idle_at or now) if root in idle else None
                if idle_at is not None and now-idle_at>=idle_grace:
                    reason='native_root_idle';break
                if now-started>=timeout:
                    reason='deadline';break
                time.sleep(.05)
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            return {'returncode':process.returncode,'timed_out':reason=='deadline',
                'completion_reason':reason or 'process_exit','native_root_session_id':root,
                'native_root_busy_observed':root in busy,'native_root_idle_observed':root in idle,
                'elapsed_seconds':time.monotonic()-started}
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=5)
