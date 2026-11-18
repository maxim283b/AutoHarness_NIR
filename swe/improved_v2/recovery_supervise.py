"""Use the frozen supervisor policy with operator recovery guards."""
import json
from datetime import datetime, timezone
from improved_v2 import supervise

supervise.SCRIPT = supervise.SCRIPT.with_name("resume_repaired.py")
_original_write = supervise.write
_worker_started_at = None


def _write_with_balance_guard(event, **fields):
    global _worker_started_at
    if event == "worker_start":
        _worker_started_at = datetime.now(timezone.utc)
    _original_write(event, **fields)
    if event != "worker_exit":
        return
    gateway_requests = supervise.OUT / "H2/gateway/requests.jsonl"
    if not gateway_requests.exists():
        return
    last = json.loads(gateway_requests.read_text().splitlines()[-1])
    created = last.get("created")
    request_time = datetime.fromisoformat(created.replace("Z", "+00:00")) if isinstance(created, str) else None
    if (last.get("status") == "http_error" and last.get("http_status") == 402
            and request_time is not None and _worker_started_at is not None
            and request_time >= _worker_started_at):
        _original_write("needs_reconciliation", reason="Insufficient balance (HTTP 402); hourly monitor will retry after top-up")
        raise SystemExit(2)


supervise.write = _write_with_balance_guard

if __name__ == "__main__":
    raise SystemExit(supervise.main())
