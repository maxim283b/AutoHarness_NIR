"""Real pinned OpenCode CLI and ACP -> guarded gateway -> local scripted API.

Never exports credentials or contacts Go. Forces native compaction and a long
tool result; verifies headers, models, parent budgets and continuation.
"""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import struct
import zlib
from types import SimpleNamespace
import uuid

import httpx

from . import go_gateway as go
from .context_policy import configure, write_plugin, PLUGIN
from .optimizer_provider import Transport, IMAGE


def main():
    root = go.ROOT / "results/autosaddler_reproduction/context_policy_fixture" / uuid.uuid4().hex[:10]
    root.mkdir(parents=True)
    wire = []
    primaries = {}
    go.credential = lambda: {"access":"local-fixture", "metadata":{"orgID":"fixture"}}
    go.quota = lambda *args: {k:{"status":"ok", "percent":0} for k in ("rolling","weekly","monthly")}
    gateway = go.GoGateway(root / "gateway", allowance_limit=20, request_limit=30, require_context_policy=True)

    def respond(request):
        payload = json.loads(request.content)
        row = gateway.rows[-1]
        purpose, role = row["purpose"], row["role"]
        wire.append({"model":payload["model"],"purpose":purpose,"role":role,"bytes":len(request.content),
                     "payload":payload})
        if len(wire) > 12:
            raise RuntimeError("Unexpected fixture loop")
        if purpose == "primary":
            primaries[role] = primaries.get(role, 0) + 1
        read_saved = purpose == "primary" and role == "optimizer" and primaries[role] == 2
        read_scratch = purpose == "primary" and role == "optimizer" and primaries[role] == 3
        read_opencode_scratch = purpose == "primary" and role == "optimizer" and primaries[role] == 4
        read_image = purpose == "primary" and role == "target_search" and primaries[role] == 2
        tool = purpose == "primary" and (primaries[role] == 1 or read_saved or read_scratch or read_opencode_scratch or read_image)
        text = "The long output was saved. Continue the fixture and reply FIXTURE_DONE." if purpose == "compaction" else "FIXTURE_DONE"
        args = {"command":"head -c 40000 /dev/zero | tr '\\000' x", "description":"Generate long local fixture output"}
        if role == "optimizer":
            args["command"] = "mkdir -p /tmp/opencode; printf SCRATCH_FIXTURE > /tmp/opencode/training_evidence_pretty.json; printf DIRECT_SCRATCH_FIXTURE > /tmp/training_rationale.txt; " + args["command"]
        tool_name = "bash"
        if read_saved:
            tool_name = "read"
            args = {"filePath":"/autoharness-context-output/"+hashlib.sha256(b"x"*40000).hexdigest()+".txt", "offset":1, "limit":1}
        if read_scratch:
            tool_name = "read"
            args = {"filePath":"/tmp/training_rationale.txt"}
        if read_opencode_scratch:
            tool_name = "read"
            args = {"filePath":"/tmp/opencode/training_evidence_pretty.json"}
        if read_image:
            tool_name = "read"
            args = {"filePath":"/fixture/media.png"}
        long_response = tool and primaries[role] == 1
        long_text = "LONG_NATIVE_RESPONSE " * 14000 if long_response else ""
        tokens = 40000 if long_response else 100
        output_tokens = 68883 if long_response else 10
        metered = {"cost":0,"opencode":{"billing_source":"go-plus","allowance":0.00001,
                   "tokens":{"input":tokens,"output":output_tokens}}}
        if payload["model"] == "qwen3.7-plus":
            block = {"type":"tool_use","id":"call_fixture_"+str(primaries[role]),"name":tool_name,"input":args} if tool else {"type":"text","text":""}
            events = [
                {"type":"message_start","message":{"id":"msg_fixture","type":"message","role":"assistant","model":payload["model"],"content":[],"stop_reason":None,"usage":{"input_tokens":tokens,"output_tokens":0}}},
                {"type":"content_block_start","index":0,"content_block":block},
                *([] if tool else [{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}}]),
                {"type":"content_block_stop","index":0},
                *([{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}},
                   {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":long_text}},
                   {"type":"content_block_stop","index":1}] if long_response else []),
                {"type":"message_delta","delta":{"stop_reason":"tool_use" if tool else "end_turn","stop_sequence":None},"usage":{"output_tokens":output_tokens,**metered}},
                {"type":"message_stop"},
            ]
        else:
            message = {"tool_calls":[{"index":0,"id":"call_fixture_"+str(primaries[role]),"type":"function","function":{"name":tool_name,"arguments":json.dumps(args)}}]} if tool else {"content":text}
            if long_response:
                message["content"] = long_text
            base = {"id":"fixture","created":1,"model":payload["model"],"object":"chat.completion.chunk"}
            events = [{**base,"choices":[{"index":0,"delta":{"role":"assistant",**message},"finish_reason":None}]},
                      {**base,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls" if tool else "stop"}]},
                      {**base,"choices":[],"usage":{"prompt_tokens":tokens,"completion_tokens":output_tokens,"total_tokens":tokens+output_tokens,**metered}}]
        data = b"".join(("event: "+e["type"]+"\n" if "type" in e else "").encode()+b"data: "+json.dumps(e).encode()+b"\n\n" for e in events)
        if payload["model"] != "qwen3.7-plus": data += b"data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type":"text/event-stream"}, content=data)

    gateway.client.close()
    gateway.client = httpx.Client(transport=httpx.MockTransport(respond))
    # Synthetic noise PNG exercises native attachment serialization, not task data.
    def chunk(kind, data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    pixels = b''.join(b'\x00'+os.urandom(512*3) for _ in range(512))
    png = b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',512,512,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')
    (root / 'media.png').write_bytes(png)
    with gateway:
        write_plugin(root / "context-policy.mjs")
        model = "opencode-go/qwen3.7-plus"
        config = {"provider":go.provider_config(model, {"x-autoharness-role":"target_search", "x-autoharness-run":"offline-fixture", "x-autoharness-benchmark":"fixture"}),
                  "enabled_providers":["opencode-go"], "share":"disabled", "autoupdate":False, "permission":{"task":"deny"}}
        configure(config, model, "file:///fixture/context-policy.mjs")
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        name = "context-cli-" + uuid.uuid4().hex[:8]
        try:
            run = subprocess.run(["docker","run","--rm","--name",name,"-e","OPENCODE_GO_API_KEY", "-v",f"{root}:/fixture",IMAGE,
                "sh","-c","mkdir -p /fixture/cli-workspace; cd /fixture/cli-workspace; OPENCODE_CONFIG=/fixture/config.json opencode --model=opencode-go/qwen3.7-plus run --format=json --dangerously-skip-permissions 'Run the local fixture.'"],
                capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120)
            (root / "cli.log").write_text(run.stdout+run.stderr,encoding="utf-8")
            assert run.returncode == 0 and '"type":"error"' not in run.stdout, str(root / "cli.log")
            assert "FIXTURE_DONE" in run.stdout, str(root / "cli.log")
        finally:
            subprocess.run(["docker","rm","-f",name],capture_output=True,timeout=20)
        workspace = root / "optimizer"
        workspace.mkdir()
        (workspace / "AGENTS.md").write_text("Local offline fixture.")
        session = SimpleNamespace(workspace=workspace,trace_dir=root / "acp",task_prompt="Run the local fixture.")
        outcome = Transport("opencode-go/kimi-k2.7-code", "offline-fixture")._run(session,120)
        assert "FIXTURE_DONE" in outcome.raw_response
        optimizer_first = next(x for x in wire if x["role"] == "optimizer" and x["purpose"] == "primary")
        assert "Local offline fixture." in json.dumps(optimizer_first["payload"]), "Optimizer instruction file was not loaded"
        native = json.loads((root / "acp/native_export.json").read_bytes())
        reads = [p for m in native["messages"] for p in m["parts"] if p.get("tool") == "read"]
        assert len(reads) == 3 and all(p["state"]["status"] == "completed" for p in reads), reads
        assert "x"*100 in reads[0]["state"]["output"]
        assert "DIRECT_SCRATCH_FIXTURE" in reads[1]["state"]["output"]
        assert "SCRATCH_FIXTURE" in reads[2]["state"]["output"]
        outputs = list(root.rglob("context-outputs/*.txt"))
        assert len(outputs) == 2 and all(p.read_bytes() == b"x"*40000 for p in outputs), outputs
        cli_events = [json.loads(l) for l in run.stdout.splitlines() if l.startswith('{')]
        tool_outputs = [e["part"]["state"]["output"] for e in cli_events if e.get("type") == "tool_use" and e["part"].get("tool") == "bash"]
        assert tool_outputs and all(len(x.encode()) < 8000 and "full 40000 bytes saved" in x for x in tool_outputs)
        for role in ("target_search","optimizer"):
            entries = [x for x in wire if x["role"] == role]
            expected = ["primary","compaction","primary"] + (["primary","primary","primary"] if role == "optimizer" else ["primary"])
            assert [x["purpose"] for x in entries] == expected, entries
            last = json.dumps(entries[-1]["payload"])
            assert "Continue the fixture" in last, last[:300]
            compaction = next(x for x in entries if x["purpose"] == "compaction")
            assert "LONG_NATIVE_RESPONSE " * 14000 in json.dumps(compaction["payload"])
            assert 210000 < go.request_sizes(compaction["payload"])[1] < go.TEXT_REQUEST_BYTES
        assert max(x["bytes"] for x in wire if x["role"] == "target_search") > 210000
        summary = gateway.accounting_summary()
    (root / "wire.json").write_text(json.dumps(wire,indent=2),encoding="utf-8")
    result = {"passed":True,"paid_model_requests":0,"synthetic_usage_not_campaign_spend":True,"policy_plugin_sha256":hashlib.sha256(PLUGIN.encode()).hexdigest(),"real_opencode_version":"1.19.5","models":["qwen3.7-plus","kimi-k2.7-code"],
              "native_transports":["CLI/Anthropic","ACP/OpenAI"],"native_compactions":2,"long_native_response_compaction_verified":True,"text_request_bytes":go.TEXT_REQUEST_BYTES,"saved_output_native_read_verified":True,"scratch_native_read_verified":True,"direct_tmp_native_read_verified":True,"native_media_over_210kb_verified":True,"request_guard_version":go.REQUEST_GUARD_VERSION,"stream_transport_version":go.STREAM_TRANSPORT_VERSION,"request_timeout_version":go.REQUEST_TIMEOUT_VERSION,"request_timeout_seconds":go.REQUEST_TIMEOUT_SECONDS,"optimizer_instruction_file_loaded":True,"fixture_requests":len(wire),"accounting":summary,"artifacts":str(root)}
    go.save(root / "proof.json", result)
    go.save(root.parent / "latest.json", result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
