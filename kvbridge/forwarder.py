from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx


async def forward_streaming(
    payload: dict[str, Any],
    upstream_url: str,
    timeout: float = 300.0,
) -> AsyncIterator[bytes]:
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{upstream_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk


async def forward_request(
    payload: dict[str, Any],
    upstream_url: str,
    timeout: float = 300.0,
) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{upstream_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


async def passthrough_get(
    path: str,
    upstream_url: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{upstream_url}{path}",
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


async def passthrough_post(
    path: str,
    payload: dict[str, Any],
    upstream_url: str,
    timeout: float = 300.0,
) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{upstream_url}{path}",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
