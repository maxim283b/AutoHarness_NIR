"""GLM optimizer admission while preserving the original Kimi defaults."""
import pytest

from experiments.autosaddler_reproduction import go_gateway as go
from experiments.autosaddler_reproduction.optimizer_provider import Transport


class Response:
    status_code = 200

    def json(self):
        return {"usage": {name: {"status": "ok", "percent": 99}
                          for name in ("rolling", "weekly", "monthly")}}


class Client:
    def get(self, *args, **kwargs):
        return Response()


def test_glm_model_and_user_authorized_quota_ceiling(tmp_path):
    assert go.MODELS["glm-5.3"]["family"] == "openai"
    assert go.quota(Client(), {"access": "dummy", "metadata": {"orgID": "dummy"}}, 100)["weekly"]["percent"] == 99
    with pytest.raises(RuntimeError, match="rolling"):
        go.quota(Client(), {"access": "dummy", "metadata": {"orgID": "dummy"}})
    h1 = go.GoGateway(tmp_path / "h1")
    h2 = go.GoGateway(tmp_path / "h2", optimizer_model="glm-5.3", quota_ceiling=100)
    assert (h1.optimizer_model, h1.quota_ceiling) == ("kimi-k2.7-code", 70)
    assert (h2.optimizer_model, h2.quota_ceiling) == ("glm-5.3", 100)
    h1.client.close()
    h2.client.close()
    assert Transport("opencode-go/glm-5.3").model == "opencode-go/glm-5.3"
