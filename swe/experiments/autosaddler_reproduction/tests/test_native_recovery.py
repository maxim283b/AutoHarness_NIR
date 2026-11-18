import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from experiments.autosaddler_reproduction import go_gateway as go
from experiments.autosaddler_reproduction import optimizer_provider as optimizer


def media_payload():
    return {"model":"qwen3.7-plus", "max_tokens":100, "messages":[{"role":"user", "content":[
        {"type":"image", "source":{"type":"base64", "media_type":"image/png",
         "data":base64.b64encode(b'fixture'*40000).decode()}},
        {"type":"text", "text":"Inspect the image"}]}]}


def test_media_wire_kept_intact_and_full_bytes_reserved(tmp_path, monkeypatch):
    payload = media_payload()
    wire_size, text_size = go.request_sizes(payload)
    assert wire_size > 210000 and text_size < 200000
    reserved = go.reservation(payload, 'qwen3.7-plus')
    assert reserved > go.reservation({"max_tokens":100}, 'qwen3.7-plus')
    monkeypatch.setattr(go, 'credential', lambda:{'access':'fixture','metadata':{'orgID':'fixture'}})
    monkeypatch.setattr(go, 'quota', lambda *args:{'fixture':True})
    seen=[]
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'usage':{'cost':0,'opencode':{'billing_source':'go-plus','allowance':0.001,'tokens':{'input':100,'output':10}}}})
    gateway=go.GoGateway(tmp_path)
    gateway.client.close()
    gateway.client=httpx.Client(transport=httpx.MockTransport(respond))
    with gateway:
        response=httpx.post(gateway.base_url+'/anthropic/v1/messages',json=payload,
            headers={'x-api-key':gateway.token,'x-autoharness-role':'target_search'},timeout=10)
        assert response.status_code==200
    assert seen==[payload]
    assert gateway.rows[-1]['reservation']==reserved


def test_media_does_not_exempt_text_or_tool_arguments():
    payload=media_payload()
    payload['messages'][0]['content'][1]['text']='x'*(go.TEXT_REQUEST_BYTES+1)
    with pytest.raises(RuntimeError,match='context'):go.reservation(payload,'qwen3.7-plus')
    payload=media_payload()
    payload['messages'][0]['content'].append({'type':'tool_use','input':{'data':'x'*(go.TEXT_REQUEST_BYTES+1)}})
    with pytest.raises(RuntimeError,match='context'):go.reservation(payload,'qwen3.7-plus')


def test_invalid_media_and_oversized_wire_rejected():
    payload=media_payload()
    payload['messages'][0]['content'][0]['source']['data']='not base64!'
    with pytest.raises(ValueError,match='encoding'):go.request_sizes(payload)
    payload['messages'][0]['content'][0]['source']['data']='A'*(go.MEDIA_REQUEST_BYTES+1)
    with pytest.raises(RuntimeError,match='transport'):go.request_sizes(payload)


def test_optimizer_timeout_exports_before_closing_container(tmp_path,monkeypatch):
    monkeypatch.setenv('GO_GATEWAY_URL','http://fixture')
    actions=[]
    class Client:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):actions.append('close')
        def initialize(self):pass
        def new_session(self,**kwargs):return {'sessionId':'fixture'}
        def request(self,*args):pass
        def prompt(self,*args,**kwargs):raise TimeoutError('fixture timeout')
    monkeypatch.setattr(optimizer,'ACPClient',Client)
    monkeypatch.setattr(optimizer,'export_session',lambda *args:actions.append('export'))
    monkeypatch.setattr(optimizer.subprocess,'run',lambda *args,**kwargs:actions.append('remove'))
    session=SimpleNamespace(workspace=tmp_path,trace_dir=tmp_path/'trace',task_prompt='fixture')
    with pytest.raises(TimeoutError,match='fixture timeout'):
        optimizer.Transport('opencode-go/kimi-k2.7-code')._run(session,1)
    assert actions==['export','close','remove']
