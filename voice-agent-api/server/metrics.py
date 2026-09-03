"""Per-call latency. Prefer LatencyReport, then AgentStartedSpeaking, then wall clock.

Wall clock is invalid on turns where the server injected speech. Written to
ops/metrics/*.jsonl.
"""

import json
import statistics
import time
from pathlib import Path

from loguru import logger

METRICS_DIR = Path(__file__).resolve().parent.parent / "ops" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# LatencyReport fields worth keeping. ttt_tool/text/thinking separate tool-call
# overhead from generation, which is where this agent actually spends its time.
_REPORT_FIELDS = (
    "stt_latency", "ttt_token_latency", "ttt_text_latency", "ttt_tool_latency",
    "ttt_thinking_latency", "tts_latency", "total_latency",
)


def turn_total(turn: dict) -> float | None:
    """Best latency for one turn, or None if only an injected wall-clock exists."""
    for key in ("report_total_latency", "dg_total_latency"):
        value = turn.get(key)
        if value is not None:
            return value
    if turn.get("wall_ttfa_injected"):
        return None
    return turn.get("wall_ttfa_s")


def _stats(values: list[float]) -> dict:
    return {
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.mean(values), 3),
        "max": round(max(values), 3),
        "n": len(values),
    }


class CallMetrics:
    def __init__(self, label: str):
        self.label = label
        self.started = time.time()
        self.turns: list[dict] = []
        self.transcript: list[dict] = []
        self.listen_model: str | None = None
        self.duplicate_lines: list[str] = []
        self.narrated_lines: list[str] = []
        self.repeated_lines: list[tuple[str, int]] = []
        self.muted_reason: str | None = None
        self.api_warnings: list[str] = []
        self._t_user_text: float | None = None
        self._awaiting_first_audio = False

    # --- event hooks -------------------------------------------------------

    def on_user_speaking(self):
        # user barged in / started a new turn; next user text closes the turn
        self._awaiting_first_audio = False

    def on_text(self, role: str, content: str):
        self.transcript.append({"t": round(time.time() - self.started, 2), "role": role, "content": content})
        if role == "user":
            self.turns.append({"turn": len(self.turns) + 1, "user": content[:60]})
            self._t_user_text = time.monotonic()
            self._awaiting_first_audio = True

    def on_agent_audio(self):
        if self._awaiting_first_audio and self._t_user_text is not None:
            wall = time.monotonic() - self._t_user_text
            self._awaiting_first_audio = False
            turn = self._current_turn()
            turn["wall_ttfa_s"] = round(wall, 3)
            if turn.get("injected"):
                turn["wall_ttfa_injected"] = True

    def on_api_warning(self, etype: str, code: str | None, desc: str | None):
        """API Warning/Error kept with the call (PROMPT_TOO_LONG is a Warning)."""
        entry = f"{etype} {code}: {desc}"
        if entry not in self.api_warnings:
            self.api_warnings.append(entry)

    def on_injection(self, kind: str):
        """Record a server injection; it invalidates this turn's wall-clock TTFA."""
        self._current_turn().setdefault("injected", []).append(kind)

    def on_agent_started(self, event: dict):
        turn = self._current_turn()
        for k in ("total_latency", "tts_latency", "ttt_latency"):
            if k in event:
                turn[f"dg_{k}"] = event[k]

    def on_latency_report(self, event: dict):
        turn = self._current_turn()
        for k in _REPORT_FIELDS:
            if k in event:
                turn[f"report_{k}"] = event[k]

    # --- output -------------------------------------------------------------

    def _current_turn(self, new: bool = False) -> dict:
        if new or not self.turns:
            self.turns.append({"turn": len(self.turns) + 1})
        return self.turns[-1]

    def last_latency(self) -> float | None:
        return turn_total(self.turns[-1]) if self.turns else None

    def summary(self) -> dict:
        totals = [v for v in (
            t.get("report_total_latency", t.get("dg_total_latency")) for t in self.turns
        ) if v is not None]
        walls = [t["wall_ttfa_s"] for t in self.turns
                 if "wall_ttfa_s" in t and not t.get("wall_ttfa_injected")]
        ttfts = [t["report_ttt_token_latency"] for t in self.turns
                 if "report_ttt_token_latency" in t]
        injected = sum(1 for t in self.turns if t.get("injected"))

        out = {"label": self.label, "turns": len(self.turns)}
        if self.listen_model:
            out["listen_model"] = self.listen_model
        if totals:
            out["dg_total_latency"] = _stats(totals)
        if walls:
            out["wall_ttfa_s"] = _stats(walls)
        # LLM time-to-first-token is ~90% of total on this agent, so it is the
        # figure that moves when the prompt or the think model changes
        if ttfts:
            out["llm_ttft_s"] = _stats(ttfts)
        if injected:
            # how often the server had to cover for the model. A demo where this
            # climbs is a demo that sounds like it is stalling.
            out["injected_turns"] = injected
        if self.duplicate_lines:
            # the model narrating a tool call the server had already narrated.
            # The caller hears it twice; only the prompt can fix it.
            out["duplicate_lines"] = self.duplicate_lines
        if self.narrated_lines:
            # "I'll look that up for you" and friends: the prompt forbids the whole
            # class, so anything here is a prompt-adherence miss, not a bug.
            out["narrated_lines"] = self.narrated_lines
        if self.repeated_lines:
            # anything the caller heard twice, whoever said it. The headline
            # number for the repetition work: on a healthy call this is empty.
            out["repeated_lines"] = [{"line": line, "count": n}
                                     for line, n in self.repeated_lines]
        if self.muted_reason:
            # the agent stopped answering before the caller hung up, and why
            out["muted"] = self.muted_reason
        if self.api_warnings:
            out["api_warnings"] = self.api_warnings
        return out

    def finish(self):
        record = {
            "label": self.label,
            "started": self.started,
            "duration_s": round(time.time() - self.started, 1),
            "summary": self.summary(),
            "turns": self.turns,
            "transcript": self.transcript,
        }
        path = METRICS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{self.label}.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"[{self.label}] call summary: {json.dumps(self.summary())}")
        logger.info(f"[{self.label}] metrics written: {path}")


LAST_SUMMARIES: list[dict] = []
