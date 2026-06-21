from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any


def identify_session(messages: list[dict[str, Any]]) -> str:
    components: list[str] = []

    system_msgs = [m for m in messages if m.get("role") == "system"]
    if system_msgs:
        content = system_msgs[0].get("content") or ""
        components.append(str(content))

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if user_msgs:
        content = user_msgs[0].get("content") or ""
        components.append(str(content)[:500])

    if not components:
        for msg in messages[:3]:
            content = msg.get("content") or ""
            components.append(f"{msg.get('role', '')}:{str(content)[:200]}")

    session_str = "||".join(components)
    return hashlib.sha256(session_str.encode()).hexdigest()[:16]


_ACTIVE_THRESHOLD = timedelta(minutes=5)
_IDLE_THRESHOLD = timedelta(minutes=30)


def classify_session_age(last_request_time: datetime) -> str:
    age = datetime.utcnow() - last_request_time
    if age <= _ACTIVE_THRESHOLD:
        return "active"
    if age <= _IDLE_THRESHOLD:
        return "idle"
    return "stale"


def estimate_cache_status(last_request_time: datetime) -> str:
    age = datetime.utcnow() - last_request_time
    if age <= _ACTIVE_THRESHOLD:
        return "hot"
    if age <= _IDLE_THRESHOLD:
        return "cold"
    return "evicted"
