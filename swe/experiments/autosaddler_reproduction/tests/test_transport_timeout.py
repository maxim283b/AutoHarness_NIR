from experiments.autosaddler_reproduction import go_gateway as go


def test_transport_ceiling_does_not_preempt_official_task_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("GO_GATEWAY_URL", "http://fixture")
    for model in ("opencode-go/qwen3.7-plus", "opencode-go/kimi-k2.7-code"):
        provider = go.provider_config(model)["opencode-go"]
        assert provider["options"]["timeout"] == 10800000
        assert provider["options"]["maxRetries"] == 0
        assert provider["models"][model.split("/", 1)[1]]["limit"]["output"] == 8192
    gateway = go.GoGateway(tmp_path)
    try:
        assert gateway.client.timeout.read == 10800
        assert gateway.client.timeout.connect == 30
    finally:
        gateway.client.close()
