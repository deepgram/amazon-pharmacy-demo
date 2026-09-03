"""Per-call latency for the LiveKit path. Written to ops/metrics/*.jsonl.

Ported from `pharmacy-voice-agent/server/metrics.py`, with the one part that
could not come across replaced.

WHAT CANNOT BE COMPARED DIRECTLY. Path A quotes Deepgram's
`LatencyReport.total_latency` -- one authoritative number per model turn, with an
STT / LLM / TTS breakdown, measured inside the Voice Agent session. There is no
such report here, because there is no such session: the three components are
separate services and LiveKit is the only thing that sees all of them. So this
uses LiveKit's own documented formula,

    total = EOUMetrics.end_of_utterance_delay + LLMMetrics.ttft + TTSMetrics.ttfb

which is the right analogue and is NOT the same stopwatch. Two known reasons the
figures will differ even with identical models:

- `end_of_utterance_delay` includes LiveKit's own endpointing delay, stacked on
  top of Nova's `endpointing=500`. Path A's STT latency has one of those, not two.
- the formula sums three independently measured stages; a LatencyReport measures
  one pipeline end to end, so queueing between stages lands in Path A's number
  and falls out of this one.

Report both, labelled, and do not present the difference as a model result.

Turns are keyed on `speech_id` rather than accumulated in arrival order. The
obvious implementation -- stash EOU, wait for LLM, wait for TTS, emit -- assumes
those three arrive strictly in sequence for one turn at a time, which stops being
true the moment preemptive generation is on: a speculative LLM run for turn N+1
starts while turn N is still synthesising, and its ttft lands in the wrong turn.
Every metric carries the speech id it belongs to, so use it.
"""

import json
import statistics
import time
from pathlib import Path

from loguru import logger

METRICS_DIR = Path(__file__).resolve().parent.parent / "ops" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def turn_total(turn: dict) -> float | None:
    """Best latency for one turn, or None if only an injected wall-clock exists."""
    value = turn.get("lk_total_s")
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
        self.think_model: str | None = None
        self.voice_model: str | None = None
        self.duplicate_lines: list[str] = []
        self.narrated_lines: list[str] = []
        self.repeated_lines: list[tuple[str, int]] = []
        self.muted_reason: str | None = None
        self.spurious_emergency = False
        self.errors: list[str] = []
        self._t_user_text: float | None = None
        self._awaiting_first_audio = False
        # speech_id -> the turn dict its stage metrics belong to. Populated by
        # whichever stage reports first; see the module docstring.
        self._by_speech: dict[str, dict] = {}

    # --- event hooks -------------------------------------------------------

    def on_user_speaking(self):
        # caller barged in / started a new turn; next user text closes the turn
        self._awaiting_first_audio = False

    def on_text(self, role: str, content: str):
        self.transcript.append({"t": round(time.time() - self.started, 2),
                                "role": role, "content": content})
        if role == "user":
            self.turns.append({"turn": len(self.turns) + 1, "user": content[:60]})
            self._t_user_text = time.monotonic()
            self._awaiting_first_audio = True

    def on_agent_audio(self):
        """First audio of an agent turn, for the wall-clock fallback."""
        if self._awaiting_first_audio and self._t_user_text is not None:
            wall = time.monotonic() - self._t_user_text
            self._awaiting_first_audio = False
            turn = self._current_turn()
            turn["wall_ttfa_s"] = round(wall, 3)
            if turn.get("injected"):
                turn["wall_ttfa_injected"] = True

    def on_error(self, message: str):
        if message not in self.errors:
            self.errors.append(message)

    def on_injection(self, kind: str):
        """Record a server-spoken line; it invalidates this turn's wall-clock TTFA.

        A hold phrase or a scripted verification line is OUR audio, so timing it
        reports the watchdog's reaction time as the agent's latency. Path A found
        this the same way and excludes the same turns.
        """
        self._current_turn().setdefault("injected", []).append(kind)

    def on_metrics(self, m) -> None:
        """One `metrics_collected` payload. Routed by speech id, not arrival order."""
        kind = type(m).__name__
        speech_id = getattr(m, "speech_id", None)
        turn = self._turn_for(speech_id)

        if kind == "EOUMetrics":
            turn["eou_delay_s"] = round(m.end_of_utterance_delay, 3)
            turn["transcription_delay_s"] = round(m.transcription_delay, 3)
        elif kind == "LLMMetrics":
            if m.cancelled:
                # A preemptive run the real turn did not use. Counting its ttft
                # would report a speculative request as the caller's wait.
                return
            if m.ttft >= 0:
                turn["llm_ttft_s"] = round(m.ttft, 3)
            turn["llm_tokens"] = m.total_tokens
        elif kind == "TTSMetrics":
            if m.cancelled:
                return
            # First segment only: a long reply synthesises in several, and the
            # caller waited for the first.
            turn.setdefault("tts_ttfb_s", round(m.ttfb, 3))
        elif kind == "STTMetrics":
            turn.setdefault("stt_audio_s", round(m.audio_duration, 3))
        else:
            return

        have = ("eou_delay_s", "llm_ttft_s", "tts_ttfb_s")
        if all(k in turn for k in have):
            turn["lk_total_s"] = round(sum(turn[k] for k in have), 3)

    def _turn_for(self, speech_id: str | None) -> dict:
        """The turn a stage metric belongs to, by speech id where one is given."""
        if not speech_id:
            return self._current_turn()
        known = self._by_speech.get(speech_id)
        if known is not None:
            return known
        turn = self._current_turn()
        self._by_speech[speech_id] = turn
        turn.setdefault("speech_id", speech_id)
        return turn

    # --- output -------------------------------------------------------------

    def _current_turn(self, new: bool = False) -> dict:
        if new or not self.turns:
            self.turns.append({"turn": len(self.turns) + 1})
        return self.turns[-1]

    def last_latency(self) -> float | None:
        return turn_total(self.turns[-1]) if self.turns else None

    def summary(self) -> dict:
        totals = [t["lk_total_s"] for t in self.turns if "lk_total_s" in t]
        walls = [t["wall_ttfa_s"] for t in self.turns
                 if "wall_ttfa_s" in t and not t.get("wall_ttfa_injected")]
        ttfts = [t["llm_ttft_s"] for t in self.turns if "llm_ttft_s" in t]
        eous = [t["eou_delay_s"] for t in self.turns if "eou_delay_s" in t]
        ttfbs = [t["tts_ttfb_s"] for t in self.turns if "tts_ttfb_s" in t]
        injected = sum(1 for t in self.turns if t.get("injected"))

        out: dict = {
            "label": self.label,
            "turns": len(self.turns),
            "orchestration": "livekit",
        }
        # All three models are ours here, not Deepgram-managed, so all three are
        # provenance worth keeping next to the numbers.
        for key, value in (("listen_model", self.listen_model),
                           ("think_model", self.think_model),
                           ("voice_model", self.voice_model)):
            if value:
                out[key] = value
        if totals:
            # LiveKit's formula, NOT Deepgram's LatencyReport. See the module
            # docstring before comparing this to Path A's dg_total_latency.
            out["lk_total_s"] = _stats(totals)
        if eous:
            # Where this path pays for stacking LiveKit endpointing on Nova's.
            out["eou_delay_s"] = _stats(eous)
        # LLM time-to-first-token is ~90% of total on this agent, so it is the
        # figure that moves when the prompt or the think model changes
        if ttfts:
            out["llm_ttft_s"] = _stats(ttfts)
        if ttfbs:
            out["tts_ttfb_s"] = _stats(ttfbs)
        if walls:
            out["wall_ttfa_s"] = _stats(walls)
        if injected:
            # how often the server had to cover for the model. A demo where this
            # climbs is a demo that sounds like it is stalling.
            out["injected_turns"] = injected
        if self.duplicate_lines:
            # the model repeating a line the server had already spoken. The caller
            # hears it twice; only the prompt can fix it.
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
        if self.spurious_emergency:
            out["spurious_emergency"] = True
        if self.errors:
            out["errors"] = self.errors
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
