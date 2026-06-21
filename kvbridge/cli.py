from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def _load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


@click.group()
def cli() -> None:
    """KVBridge — Cache-aware proxy for local LLM inference."""


@cli.command()
@click.option("--port", default=None, type=int, help="Port to listen on")
@click.option("--upstream", default=None, help="Upstream server URL")
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--host", default=None, help="Host to bind")
def start(port: int | None, upstream: str | None, config: str, host: str | None) -> None:
    """Start the KVBridge proxy server."""
    import uvicorn

    from .logger import setup_logger
    from .server import create_app
    from .tracker import TTFTTracker

    cfg = _load_config(config)

    _host = host or cfg.get("server", {}).get("host", "0.0.0.0")
    _port = port or cfg.get("server", {}).get("port", 8000)
    _upstream = upstream or cfg.get("upstream", {}).get("url", "http://localhost:1234")
    _timeout = cfg.get("upstream", {}).get("timeout_seconds", 300)
    _norm_enabled = cfg.get("normalization", {}).get("enabled", True)

    tracking_cfg = cfg.get("tracking", {})
    data_dir = tracking_cfg.get("data_dir", "./data")
    hit_thresh = tracking_cfg.get("cache_hit_threshold", 0.3)
    miss_thresh = tracking_cfg.get("cache_miss_threshold", 0.7)
    calib = tracking_cfg.get("calibration_requests", 5)

    log_cfg = cfg.get("logging", {})
    setup_logger(
        level=log_cfg.get("level", "info"),
        log_file=log_cfg.get("file"),
        console=log_cfg.get("console", True),
    )

    tracker = TTFTTracker(
        data_dir=data_dir,
        hit_threshold=hit_thresh,
        miss_threshold=miss_thresh,
        calibration_requests=calib,
    )

    app = create_app(
        upstream_url=_upstream,
        upstream_timeout=float(_timeout),
        tracker=tracker,
        normalization_enabled=_norm_enabled,
    )

    console.print(f"[green]KVBridge starting[/green] → upstream: {_upstream}")
    console.print(f"Listening on http://{_host}:{_port}")
    uvicorn.run(app, host=_host, port=_port, log_level="warning")


@cli.command()
@click.option("--config", default="config.yaml", help="Config file path")
def status(config: str) -> None:
    """Show current sessions and cache stats."""
    import httpx

    cfg = _load_config(config)
    port = cfg.get("server", {}).get("port", 8000)
    host = cfg.get("server", {}).get("host", "localhost")
    base = f"http://{host}:{port}"

    try:
        r = httpx.get(f"{base}/kvbridge/status", timeout=5)
        data = r.json()
    except Exception as exc:
        console.print(f"[red]Error connecting to KVBridge:[/red] {exc}")
        sys.exit(1)

    console.print(f"[bold]KVBridge Status[/bold]  upstream={data.get('upstream')}")
    console.print(f"  Sessions:     {data.get('total_sessions', 0)}")
    console.print(f"  Requests:     {data.get('total_requests', 0)}")
    console.print(f"  Hit rate:     {data.get('overall_hit_rate', 0):.1%}")
    if data.get("prefill_tps"):
        console.print(f"  Prefill TPS:  {data['prefill_tps']:.1f} tok/ms")


@cli.command("sessions")
@click.option("--config", default="config.yaml", help="Config file path")
def list_sessions(config: str) -> None:
    """List all tracked sessions."""
    import httpx

    cfg = _load_config(config)
    port = cfg.get("server", {}).get("port", 8000)
    host = cfg.get("server", {}).get("host", "localhost")
    base = f"http://{host}:{port}"

    try:
        r = httpx.get(f"{base}/kvbridge/sessions", timeout=5)
        data = r.json()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    table = Table(title="Sessions")
    table.add_column("Session ID")
    table.add_column("Requests", justify="right")
    table.add_column("Hits", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Avg TTFT Hit", justify="right")
    table.add_column("Cache State")

    for s in data.get("sessions", []):
        table.add_row(
            s["session_id"],
            str(s["total_requests"]),
            str(s["cache_hits"]),
            f"{s['hit_rate']:.1%}",
            f"{s['avg_ttft_hit_ms']:.0f}ms",
            s["estimated_cache_status"],
        )

    console.print(table)


@cli.command()
@click.option("--config", default="config.yaml", help="Config file path")
def metrics(config: str) -> None:
    """Show TTFT history and cache hit rates."""
    import httpx

    cfg = _load_config(config)
    port = cfg.get("server", {}).get("port", 8000)
    host = cfg.get("server", {}).get("host", "localhost")
    base = f"http://{host}:{port}"

    try:
        r = httpx.get(f"{base}/kvbridge/metrics", timeout=5)
        data = r.json()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    rows = data.get("metrics", [])
    console.print(f"[bold]{len(rows)} recorded requests[/bold]")

    if not rows:
        return

    table = Table(title="TTFT Metrics")
    table.add_column("Session")
    table.add_column("Timestamp")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Cache", justify="center")
    table.add_column("Tokens In", justify="right")
    table.add_column("Model")

    for m in rows[-50:]:  # show last 50
        table.add_row(
            m["session_id"],
            m["timestamp"][:19],
            f"{m['ttft_ms']:.0f}" if m["ttft_ms"] is not None else "—",
            m["cache_status"],
            str(m["input_tokens"]),
            m["model"],
        )

    console.print(table)


@cli.command()
@click.option("--config", default="config.yaml", help="Config file path")
def reset(config: str) -> None:
    """Clear all tracking data."""
    import httpx

    cfg = _load_config(config)
    port = cfg.get("server", {}).get("port", 8000)
    host = cfg.get("server", {}).get("host", "localhost")
    base = f"http://{host}:{port}"

    try:
        r = httpx.post(f"{base}/kvbridge/reset", timeout=5)
        console.print(f"[green]{r.json().get('status', 'done')}[/green]")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@cli.command()
def version() -> None:
    """Show version."""
    from . import __version__
    console.print(f"kvbridge {__version__}")
