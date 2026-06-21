from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    model_config = {"extra": "allow"}


@dataclass
class RequestMetric:
    session_id: str
    timestamp: datetime
    ttft_ms: float | None
    cache_status: str  # "hit", "miss", "uncertain", "unknown"
    input_tokens: int
    output_tokens: int
    total_latency_ms: float
    message_count: int
    prefix_length: int
    model: str


@dataclass
class SessionMetrics:
    session_id: str
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_ttft_hit_ms: float = 0.0
    avg_ttft_miss_ms: float = 0.0
    last_request_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    estimated_cache_status: str = "cold"  # "hot", "cold", "evicted"
