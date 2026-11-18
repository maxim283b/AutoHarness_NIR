import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from experiments.autosaddler_reproduction import go_gateway as go


def billing():
    return {"cost":0,"opencode":{"billing_source":"go-plus","allowance":0.01,"tokens":{"input":20,"output":10}}}


def frame(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


class Stream(httpx.SyncByteStream):
    def __init__(self, chunks): self.chunks = chunks
    def __iter__(self): yield from self.chunks()


def setup_gateway(tmp_path, monkeypatch, respond):
    monkeypatch.setattr(go,"credential",lambda:{"access":"fixture","metadata":{"orgID":"fixture"}})
    monkeypatch.setattr(go,"quota",lambda *a:{"fixture":True})
    g=go.GoGateway(tmp_path)
    g.client.close()
    g.client=httpx.Client(transport=httpx.MockTransport(respond))
    return g


def test_http_sse_delivered_before_upstream_finishes(tmp_path, monkeypatch):
    observed=threading.Event();seen=[]
    first=frame({"type":"message_start","message":{"usage":{"input_tokens":20}}})
    last=frame({"type":"message_delta","usage":billing()})+frame({"type":"message_stop"})
    def chunks():
        yield first
        assert observed.wait(3), "Gateway buffered instead of streaming"
        yield last
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,headers={"content-type":"text/event-stream"},stream=Stream(chunks))
    g=setup_gateway(tmp_path,monkeypatch,respond)
    payload={"model":"qwen3.7-plus","max_tokens":8192,"stream":True,"messages":[]}
    with g:
        with httpx.stream("POST",g.base_url+"/anthropic/v1/messages",json=payload,headers={"x-api-key":g.token,"x-autoharness-role":"target_search"},timeout=5) as response:
            it=response.iter_bytes();data=next(it);assert data==first;observed.set();data+=b"".join(it)
        assert data==first+last
        assert g.rows[-1]["usage"]["total_tokens"]==30
    assert seen==[payload]


def test_disconnected_queued_request_never_dispatches(tmp_path, monkeypatch):
    seen=[]
    g=setup_gateway(tmp_path,monkeypatch,lambda r:seen.append(r))
    alive=threading.Event();alive.set()
    g.lock.acquire()
    with ThreadPoolExecutor() as pool:
        f=pool.submit(g.forward,"/anthropic/v1/messages",{"model":"qwen3.7-plus","max_tokens":8192},{"x-autoharness-role":"target_search"},client_alive=alive.is_set)
        alive.clear();g.lock.release()
        with pytest.raises(go.ClientDisconnected):f.result(timeout=3)
    assert not seen and not g.rows
    g.client.close()


def test_slow_reply_does_not_block_next_task_and_reservations_are_atomic(tmp_path,monkeypatch):
    slow=threading.Event();release=threading.Event();seen=[]
    payload={"model":"qwen3.7-plus","max_tokens":8192,"stream":True}
    def chunks():
        slow.set();assert release.wait(4)
        yield frame({"usage":billing()})
    def respond(request):
        seen.append(json.loads(request.content))
        if len(seen)==1:return httpx.Response(200,stream=Stream(chunks))
        return httpx.Response(200,content=frame({"usage":billing()}))
    g=setup_gateway(tmp_path,monkeypatch,respond)
    reserve=go.reservation(payload,'qwen3.7-plus');g.role_limits['target_search']=reserve+0.1
    with ThreadPoolExecutor() as pool:
        f=pool.submit(g.forward,'/anthropic/v1/messages',payload,{'x-autoharness-role':'target_search'})
        assert slow.wait(3)
        with pytest.raises(RuntimeError,match='role budget'):
            g.forward('/anthropic/v1/messages',payload,{'x-autoharness-role':'target_search'})
        g.forward('/anthropic/v1/messages',payload,{'x-autoharness-role':'target_final_baseline'})
        release.set();f.result(timeout=3)
    assert len(seen)==2 and g.spent()==0.02
    g.client.close()


def test_disconnect_retains_unknown_reserve_without_halting_next_task(tmp_path,monkeypatch):
    def chunks():yield frame({"type":"message_start","message":{"usage":{"input_tokens":20}}})
    g=setup_gateway(tmp_path,monkeypatch,lambda r:httpx.Response(200,stream=Stream(chunks)))
    def disconnected(chunk):raise go.ClientDisconnected('fixture disconnect')
    with pytest.raises(go.ClientDisconnected):
        g.forward('/anthropic/v1/messages',{'model':'qwen3.7-plus','stream':True,'max_tokens':8192},{'x-autoharness-role':'target_search'},stream_start=lambda *a:None,stream_chunk=disconnected)
    row=g.rows[-1];assert row['status']=='client_disconnected' and 'usage' not in row
    assert g.spent()==row['reservation'] and g.halted is None
    g.client.close()


def test_invalid_billing_withholds_terminal_frame_and_halts(tmp_path,monkeypatch):
    bad=billing();bad['cost']=1
    end=frame({'type':'message_stop'})
    g=setup_gateway(tmp_path,monkeypatch,lambda r:httpx.Response(200,content=frame({'usage':bad})+end))
    delivered=[]
    with pytest.raises(RuntimeError,match='balance charge'):
        g.forward('/anthropic/v1/messages',{'model':'qwen3.7-plus','stream':True,'max_tokens':8192},{'x-autoharness-role':'target_search'},stream_start=lambda *a:None,stream_chunk=delivered.append)
    assert end not in delivered and g.halted
    g.client.close()


def test_idle_upstream_is_cancelled_without_waiting_for_another_chunk(tmp_path,monkeypatch):
    entered=threading.Event();closed=threading.Event();alive=threading.Event();alive.set()
    class Idle(httpx.SyncByteStream):
        def __iter__(self):
            entered.set()
            assert closed.wait(3), 'Idle read was not cancelled'
            raise httpx.ReadError('Socket closed by watchdog')
            yield b''
        def close(self): closed.set()
    g=setup_gateway(tmp_path,monkeypatch,lambda r:httpx.Response(200,stream=Idle()))
    with ThreadPoolExecutor() as pool:
        future=pool.submit(g.forward,'/anthropic/v1/messages',{'model':'qwen3.7-plus','stream':True,'max_tokens':8192},{'x-autoharness-role':'target_search'},client_alive=alive.is_set)
        assert entered.wait(2);alive.clear()
        with pytest.raises(go.ClientDisconnected): future.result(timeout=1)
    assert closed.is_set() and not g.active_requests
    assert g.rows[-1]['status']=='client_disconnected' and not g.rows[-1].get('usage')
    assert g.halted is None and g.spent()==g.rows[-1]['reservation']
    g.client.close()


def test_real_idle_tcp_read_is_interrupted_on_client_disconnect(tmp_path,monkeypatch):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    entered=threading.Event();release=threading.Event();alive=threading.Event();alive.set()
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Transfer-Encoding','chunked');self.end_headers()
            data=frame({'type':'message_start'})
            self.wfile.write(('%x\r\n'%len(data)).encode()+data+b'\r\n');self.wfile.flush()
            entered.set();release.wait(4)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    class LocalTransport(httpx.BaseTransport):
        def __init__(self): self.inner=httpx.HTTPTransport()
        def handle_request(self,request):
            request.url=httpx.URL(f'http://127.0.0.1:{server.server_port}/fixture')
            return self.inner.handle_request(request)
        def close(self): self.inner.close()
    g=setup_gateway(tmp_path,monkeypatch,lambda r:None);g.client.close();g.client=httpx.Client(transport=LocalTransport(),timeout=10)
    try:
        with ThreadPoolExecutor() as pool:
            future=pool.submit(g.forward,'/anthropic/v1/messages',{'model':'qwen3.7-plus','stream':True,'max_tokens':8192},{'x-autoharness-role':'target_search'},client_alive=alive.is_set)
            assert entered.wait(2);alive.clear()
            try:
                with pytest.raises(go.ClientDisconnected): future.result(timeout=2)
            finally: release.set()
        assert g.halted is None and not g.active_requests
        assert g.rows[-1]['status']=='client_disconnected' and not g.rows[-1].get('usage')
    finally:
        release.set();g.client.close();server.shutdown();server.server_close();thread.join(2)
