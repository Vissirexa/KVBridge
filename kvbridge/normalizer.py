from __future__ import annotations

import re
from typing import Any

_STANDARD_MESSAGE_FIELDS = ("role", "content", "name", "tool_calls", "tool_call_id")
_STANDARD_TOOL_CALL_FIELDS = ("id", "type", "function")
_STANDARD_FUNCTION_FIELDS = ("name", "arguments")

_MULTI_SPACE_RE = re.compile(r"(?<=\S)[ \t]{2,}")  # only collapse spaces after a non-whitespace char, preserving indentation
_CRLF_RE = re.compile(r"\r\n|\r")
_TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_message(msg) for msg in messages]


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}

    for key in _STANDARD_MESSAGE_FIELDS:
        if key in msg:
            clean[key] = msg[key]

    if "content" in clean:
        clean["content"] = _normalize_content(clean["content"])

    if "tool_calls" in clean and clean["tool_calls"] is not None:
        clean["tool_calls"] = _normalize_tool_calls(clean["tool_calls"])

    clean = _normalize_reasoning_fields(msg, clean)

    return clean


def _normalize_content(content: Any) -> Any:
    if content is None:
        return ""

    if isinstance(content, str):
        return _normalize_string(content)

    if isinstance(content, list):
        normalized = [_normalize_content_part(part) for part in content]
        normalized.sort(key=lambda p: p.get("type", "") if isinstance(p, dict) else "")
        return normalized

    return content


def _normalize_content_part(part: Any) -> Any:
    if isinstance(part, dict):
        result = dict(part)
        if "text" in result and isinstance(result["text"], str):
            result["text"] = _normalize_string(result["text"])
        return result
    return part


def _normalize_string(s: str) -> str:
    s = _CRLF_RE.sub("\n", s)
    s = _TRAILING_WHITESPACE_RE.sub("", s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s


def _normalize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            clean_tc: dict[str, Any] = {}
            for key in _STANDARD_TOOL_CALL_FIELDS:
                if key in tc:
                    clean_tc[key] = tc[key]
            if "function" in clean_tc and isinstance(clean_tc["function"], dict):
                clean_tc["function"] = _normalize_function(clean_tc["function"])
        else:
            # Pydantic model with .model_dump() or dict-like object
            try:
                raw = tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            except Exception:
                raw = {}
            clean_tc = {}
            for key in _STANDARD_TOOL_CALL_FIELDS:
                if key in raw:
                    clean_tc[key] = raw[key]
            if "function" in clean_tc and isinstance(clean_tc["function"], dict):
                clean_tc["function"] = _normalize_function(clean_tc["function"])
        normalized.append(clean_tc)

    normalized.sort(key=lambda t: t.get("id", ""))
    return normalized


def _normalize_function(func: dict[str, Any]) -> dict[str, Any]:
    return {k: func[k] for k in _STANDARD_FUNCTION_FIELDS if k in func}


def _normalize_reasoning_fields(
    original: dict[str, Any], clean: dict[str, Any]
) -> dict[str, Any]:
    has_reasoning = "reasoning" in original
    has_reasoning_content = "reasoning_content" in original

    if has_reasoning and not has_reasoning_content:
        clean["reasoning_content"] = original["reasoning"]
    elif has_reasoning_content and not has_reasoning:
        clean["reasoning_content"] = original["reasoning_content"]
    elif has_reasoning and has_reasoning_content:
        clean["reasoning_content"] = original["reasoning_content"]

    return clean
