"""Operator-only choice between the two existing OpenCode workspaces."""
from __future__ import annotations

import json
import math
import subprocess
import time

import httpx

from experiments.autosaddler_reproduction.fullsource_host import checked_file

KNOWN_ORGS = {
    "org_01M6DEH67FHPJXQPZ8PSX4ERE7",
    "org_01M7CGWASNHKT5JFRG8QEK178J",
}
USAGE_URL = "https://opencode.ai/zen/go/v1/usage"


def _connections():
    cli = checked_file("desktop_cli")
    result = subprocess.run([str(cli), "auth", "export", "opencode"],
                            capture_output=True, text=True, timeout=30, check=True)
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("Expected a list of OpenCode credentials")
    found = {}
    for row in rows:
        if row.get("integrationID") != "opencode":
            continue
        value = row.get("value")
        if not isinstance(value, dict) or value.get("type") != "oauth":
            continue
        org = value.get("metadata", {}).get("orgID")
        if org in KNOWN_ORGS and isinstance(value.get("access"), str):
            if org in found:
                raise RuntimeError("Duplicate OpenCode workspace credential")
            found[org] = (value, bool(row.get("active")))
    return found


def credential_for(org):
    found = _connections()
    if org not in KNOWN_ORGS or org not in found:
        raise RuntimeError("Selected OpenCode workspace connection is unavailable")
    value = found[org][0]
    if value.get("expires", 0) / 1000 < time.time() + 60:
        raise RuntimeError("Selected OpenCode workspace credential has expired")
    return value


def availability():
    """Return sanitized quota metadata; never persist or print credentials."""
    found = _connections()
    snapshots = []
    with httpx.Client(timeout=25) as client:
        for org in sorted(KNOWN_ORGS):
            row = {"workspace": org, "active_connection": False, "available": False}
            if org not in found:
                row["reason"] = "connection_missing"
                snapshots.append(row)
                continue
            auth, active = found[org]
            row["active_connection"] = active
            if auth.get("expires", 0) / 1000 < time.time() + 60:
                row["reason"] = "credential_expired"
                snapshots.append(row)
                continue
            try:
                response = client.get(USAGE_URL, headers={
                    "Authorization": "Bearer " + auth["access"],
                    "x-opencode-org-id": org,
                    "User-Agent": "autoharness-opencode/1.19.5",
                })
                if response.status_code != 200:
                    row["reason"] = "quota_http_" + str(response.status_code)
                else:
                    usage = response.json()["usage"]
                    windows = {}
                    for name in ("rolling", "weekly", "monthly"):
                        item = usage[name]
                        percent = item["percent"]
                        if (isinstance(percent, bool) or not isinstance(percent, (int, float))
                                or not math.isfinite(percent) or percent < 0):
                            raise ValueError("invalid quota percent")
                        windows[name] = {"percent": percent, "status": item.get("status")}
                    row["windows"] = windows
                    row["available"] = all(item["status"] == "ok" and item["percent"] < 100
                                           for item in windows.values())
                    if not row["available"]:
                        row["reason"] = "go_quota_exhausted"
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                row["reason"] = "quota_unavailable"
            snapshots.append(row)
    return snapshots


def choose(snapshots):
    ready = [row for row in snapshots if row["available"]]
    ready.sort(key=lambda row: (row["windows"]["monthly"]["percent"],
                                row["windows"]["weekly"]["percent"],
                                row["windows"]["rolling"]["percent"],
                                not row["active_connection"], row["workspace"]))
    return ready[0]["workspace"] if ready else None
