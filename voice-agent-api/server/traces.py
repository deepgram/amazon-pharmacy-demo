"""Export OTel spans to Coval so tool-call and component-latency metrics work.

Coval's simulation transcript contains only the PERSONA's tool calls
(`tool_call_owner` is always "user"), so `METRIC_TOOLCALL` and the per-stage
latency metrics have nothing to read until the agent sends traces of its own.
Everything they need is already measured on this side:

    llm_tool_call   one span per client-side tool, from ToolBox.dispatch
    stt / llm / tts one span per turn carrying `metrics.ttfb`, straight off
                    Deepgram's LatencyReport (server/metrics.py::_REPORT_FIELDS)

Correlation comes from the WebSocket handshake. The Coval agent is configured
with

    custom_headers = {"X-Coval-Simulation-Id": "{{simulation_output_id}}"}

which Coval resolves per simulation, so `browser_ws` reads that header and hands
it here. No header, no export -- a human on the browser or phone path is not a
Coval simulation and has nothing to correlate against.

OFF BY DEFAULT. Set COVAL_TRACES=1 (and COVAL_API_KEY) to enable. Every failure
is swallowed and logged: a telemetry problem must never take down a call, and
this runs on the live demo agent.
"""

import json
import os
import random
import time
import urllib.error
import urllib.request

from loguru import logger

# Importing config for its load_dotenv side effect, so COVAL_* resolve from .env
# exactly like every other setting rather than only from the process env.
from . import config  # noqa: F401

ENABLED = os.environ.get("COVAL_TRACES", "").strip() in ("1", "true", "yes")
API_KEY = os.environ.get("COVAL_API_KEY", "").strip().strip("'\"")
ENDPOINT = os.environ.get("COVAL_TRACES_URL", "https://api.coval.dev/v1/traces")

# Deepgram LatencyReport field -> the span name Coval reads it from.
# https://docs.coval.ai/concepts/metrics/types/trace
_STAGE_SPANS = {
    "stt_latency": "stt",
    "ttt_token_latency": "llm",
    "tts_latency": "tts",
}


def _hex(n: int) -> str:
    return f"{random.getrandbits(n * 4):0{n}x}"


def _attr(key: str, value) -> dict:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


class CallTrace:
    """Spans for one call. Inert unless a Coval simulation id came in."""

    def __init__(self, simulation_id: str | None, label: str = "call"):
        self.simulation_id = (simulation_id or "").strip()
        self.label = label
        self.trace_id = _hex(32)
        self.started_ns = time.time_ns()
        self.spans: list[dict] = []

    @property
    def active(self) -> bool:
        return bool(ENABLED and API_KEY and self.simulation_id)

    # --- recording ---------------------------------------------------------

    def tool_call(self, name: str, arguments: str, result: str,
                  started_ns: int, ended_ns: int) -> None:
        """One client-side tool call, as the `llm_tool_call` span Coval counts."""
        if not self.active:
            return
        # Arguments and results are what METRIC_TOOLCALL judges against, but the
        # OTP result carries the live code -- never ship that to a third party.
        safe_result = result
        if name in ("send_otp", "verify_otp"):
            safe_result = json.dumps({"redacted": f"{name} result withheld"})
        self.spans.append({
            "traceId": self.trace_id,
            "spanId": _hex(16),
            "name": "llm_tool_call",
            "kind": 1,
            "startTimeUnixNano": str(started_ns),
            "endTimeUnixNano": str(ended_ns),
            "attributes": [
                _attr("tool.name", name),
                _attr("gen_ai.tool.name", name),
                _attr("tool.arguments", (arguments or "")[:4000]),
                _attr("tool.result", (safe_result or "")[:4000]),
                _attr("duration.ms", round((ended_ns - started_ns) / 1e6, 2)),
            ],
        })

    def latency_report(self, event: dict, turn: int) -> None:
        """A Deepgram LatencyReport, fanned out into stt / llm / tts spans."""
        if not self.active:
            return
        now = time.time_ns()
        for field, span_name in _STAGE_SPANS.items():
            seconds = event.get(field)
            if seconds is None:
                continue
            dur_ns = int(float(seconds) * 1e9)
            self.spans.append({
                "traceId": self.trace_id,
                "spanId": _hex(16),
                "name": span_name,
                "kind": 1,
                "startTimeUnixNano": str(now - dur_ns),
                "endTimeUnixNano": str(now),
                "attributes": [
                    # Coval reads time-to-first-byte off this attribute
                    _attr("metrics.ttfb", float(seconds)),
                    _attr("turn", turn),
                    _attr("source", "deepgram.LatencyReport"),
                ],
            })
        total = event.get("total_latency")
        if total is not None:
            dur_ns = int(float(total) * 1e9)
            self.spans.append({
                "traceId": self.trace_id,
                "spanId": _hex(16),
                "name": "turn",
                "kind": 1,
                "startTimeUnixNano": str(now - dur_ns),
                "endTimeUnixNano": str(now),
                "attributes": [_attr("metrics.ttfb", float(total)),
                               _attr("turn", turn)],
            })

    # --- export ------------------------------------------------------------

    def payload(self) -> dict:
        return {
            "resourceSpans": [{
                "resource": {"attributes": [
                    _attr("service.name", "pharmacy-voice-agent"),
                    _attr("coval.simulation_id", self.simulation_id),
                ]},
                "scopeSpans": [{
                    "scope": {"name": "pharmacy-voice-agent"},
                    "spans": self.spans,
                }],
            }],
        }

    def flush(self) -> bool:
        """POST the call's spans. Returns True on success; never raises."""
        if not self.active or not self.spans:
            return False
        body = json.dumps(self.payload()).encode()
        req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
            "X-Simulation-Id": self.simulation_id,
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                logger.info(f"[{self.label}] traces: sent {len(self.spans)} spans "
                            f"for simulation {self.simulation_id} ({r.status})")
                return True
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            logger.warning(f"[{self.label}] traces: {e.code} {detail}")
        except Exception as e:  # telemetry must never break a call
            logger.warning(f"[{self.label}] traces: export failed: {e}")
        return False
