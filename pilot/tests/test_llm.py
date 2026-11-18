import json

from llm import OpenAICompatibleChat, TokenLedger


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }).encode()


def test_openai_compatible_lm_studio_shape(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ledger = TokenLedger()
    client = OpenAICompatibleChat(
        "local-model", {"temperature": 0, "seed": 11, "num_predict": 64}, ledger,
        base_url="http://127.0.0.1:1234/v1/chat/completions",
    )
    result = client.chat([{"role": "user", "content": "ping"}])

    assert result["text"] == "ok"
    assert result["prompt_tokens"] == 7
    assert result["completion_tokens"] == 3
    assert seen["body"]["seed"] == 11
    assert seen["body"]["max_tokens"] == 64
    assert ledger.snapshot()["target"]["total_tokens"] == 10
