"""Arbiter for server-spoken agent lines (filler, check-in, scripted lines).

Ported from `pharmacy-voice-agent/server/speech.py`. The arbitration is
unchanged -- same four priorities, same drop reasons, same per-call repeat cap,
same duplicate/narration counting. What changed is the two ends of it: where the
state comes from, and how a line gets spoken.

    Path A                              here
    ----------------------------------  ------------------------------------
    InjectAgentMessage over the socket  AgentSession.say()
    binary audio frames -> `speaking`   agent_state_changed -> `speaking`
    AgentAudioDone clears it            agent_state leaves "speaking"
    UserStartedSpeaking                 user_state_changed -> "speaking"
    assistant ConversationText          conversation_item_added (role assistant)

Two things get genuinely better on this transport, and both delete code rather
than adding it:

- `say()` returns a `SpeechHandle`. Path A could only find out whether a line was
  heard by watching for its words to come back in a ConversationText and
  re-injecting on a timer if they did not -- see `_ensure_greeting` in va.py, two
  attempts and three tunable timeouts. Here the handle says so directly:
  `await handle.wait_for_playout()` then `handle.interrupted`.
- A line that must not be talked over can say so: `allow_interruptions=False`.
  The compliance greeting carries the 911 instruction and the recording
  disclosure, and on Path A a caller saying "hello" during slow synthesis
  flushed it silently. It cannot be flushed here.

`speaking` still is not cleared by a new user turn, for the reason Path A found
the hard way: on a long reply the caller can start talking several seconds in,
and clearing it there let filler fire over audio that was still playing.
"""

import re
import time
from collections import Counter, deque
from enum import IntEnum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # avoid importing livekit at module scope for the offline harnesses
    from livekit.agents import AgentSession
    from livekit.agents.voice.speech_handle import SpeechHandle


class Priority(IntEnum):
    FILLER = 10    # drop if anything else is already covering the silence
    CHECKIN = 15   # caller went quiet after we finished
    ENFORCE = 20   # price disclaimer / day-supply conversion
    SCRIPTED = 30  # verification / transfer / greeting / closing lines


_ECHO_MEMORY = 12

# Openers whose only job is to say work is happening. Deliberately specific: "let me
# know if you need anything else" is a fine thing to say, "let me get that price" is
# not, so the verb matters and a bare "let me" would over-match.
_NARRATION = (
    "let me check", "let me look", "let me get", "let me pull", "let me verify",
    "let me find", "let me grab", "i'll look", "i'll check", "i'll get",
    "i'll pull", "i'll verify", "i'll find", "one moment", "just a moment",
    "give me a moment", "give me a second", "pulling that up", "pulling up",
    "bear with me", "hang on", "hold on while", "i'm checking", "i am checking",
    "let me see what", "allow me to",
)


def _is_narration(content: str) -> bool:
    """A short line that only announces a lookup, rather than answering anything.

    Length-bounded on purpose: "Let me check that for you" is narration, but a long
    sentence that happens to contain those words is usually carrying real content.
    """
    text = content.strip().lower()
    return len(text) <= 90 and any(m in text for m in _NARRATION)


# Below this, a repeat is not a stall. "Yes.", "Okay.", "Of course." (9),
# "no problem" (10), "sounds good" (11) recur in every healthy conversation and
# carry the turn forward.
#
# This was 16, which let the single most common loop through: "Take your time."
# normalises to 14 characters, so it was never counted at all -- the 68-repeat
# call in the first evaluation, and five consecutive repeats on a deployed smoke
# test. 12 is the line that catches it while leaving every ordinary short
# acknowledgement alone. The run also has to be CONSECUTIVE, so a phrase that
# recurs across a progressing call still never trips it.
_STALL_MIN_CHARS = 12

# How long a user-speaking state with no transcript keeps the server quiet.
_USER_SPEAKING_MAX_S = 15.0


def _norm(content: str) -> str:
    """Compare on words alone, so punctuation and casing cannot hide a repeat."""
    return re.sub(r"[^a-z0-9 ]+", "", content.lower()).strip()


class SpeechGate:
    def __init__(
        self,
        label: str,
        metrics,
        split_turn_grace_s: float,
        max_line_repeats: int = 2,
        stall_limit: int = 3,
    ):
        self.label = label
        self._metrics = metrics
        self._split_grace = split_turn_grace_s
        self._max_repeats = max_line_repeats
        self._stall_limit = stall_limit
        # Set by session.py once AgentSession exists. The gate is constructed
        # before the session so the tool layer can hold a reference to it.
        self._session: "AgentSession | None" = None

        self.turn = 0
        self._speaking = False
        self._user_speaking = False  # caller mid-utterance; never talk over them
        self._user_speaking_at = 0.0
        self._model_spoke = True     # no watchdog before the first user turn
        self._top_played = 0         # highest priority spoken on this turn
        self._kinds_this_turn: set[str] = set()
        self._last_agent_line_at = -1e9
        self._said_at: dict[str, float] = {}
        self._echo: deque[str] = deque(maxlen=_ECHO_MEMORY)
        self._spoken: deque[str] = deque(maxlen=_ECHO_MEMORY)
        # every line the caller heard this call, ours and the model's, normalised
        self._heard: Counter[str] = Counter()
        self._stall_run = 0          # consecutive agent turns that repeated a line
        self._turn_lines = 0         # agent lines heard on the turn in progress
        self._turn_had_new = False   # ...at least one of them not heard before
        self.duplicates: list[str] = []
        self.narrations: list[str] = []
        self._closed = False

    def attach(self, session: "AgentSession") -> None:
        self._session = session

    # --- state the session feeds in ------------------------------------------

    def on_agent_state(self, state: str) -> None:
        """`agent_state_changed`. "speaking" is the only state that gates us.

        Path A had to drive this from raw audio frames because
        AgentStartedSpeaking is not emitted on its shipped listen config. LiveKit
        maintains the state itself, so this is the whole implementation.
        """
        if state == "speaking":
            self._speaking = True
        else:
            self._speaking = False

    def on_user_state(self, state: str) -> None:
        """`user_state_changed`. Barge-in: LiveKit stops the agent for us."""
        if state == "speaking":
            self._speaking = False
            # And the CALLER now is. Nothing else knows to hold back while
            # someone is mid-sentence: on Path A a hold phrase fired straight
            # over a caller twice on one call, and the silence check-in would
            # happily start "Are you still there?" over someone still talking.
            self._user_speaking = True
            self._user_speaking_at = time.monotonic()

    def on_agent_text(self, content: str) -> None:
        """An assistant message landed. Ours, or the model's?"""
        self._last_agent_line_at = time.monotonic()
        echo = self._match_echo(content)
        if echo is not None:
            self._echo.remove(echo)   # our own line, counted at say() time
            return
        self._count_heard(content)
        if content in self._spoken:
            self.duplicates.append(content)
            logger.warning(f"[{self.label}] model repeated a server line: {content!r}")
        elif _is_narration(content):
            # Not one of our lines, but the same defect: a sentence whose only job
            # is to say work is happening. The prompt forbids the whole class, and
            # the model keeps inventing new wordings for it, so count them rather
            # than trust that the rule held.
            self.narrations.append(content)
            logger.warning(f"[{self.label}] model narrated a tool call: {content!r}")
        self._model_spoke = True

    def _match_echo(self, content: str) -> str | None:
        """The line WE said that this assistant message is an echo of, if any.

        Not an equality test, and that is the one place this transport is fussier
        than Path A's. A ConversationText carried the whole injected line; the
        message LiveKit records after `say()` carries `forwarded_text` -- what was
        actually spoken. Interrupt a hold phrase halfway and the echo comes back
        as "Let me check that" against "Let me check that for you.", which an
        exact match reads as the MODEL narrating a tool call. That would count a
        barge-in as a prompt-adherence defect on every interrupted server line.

        Prefix in either direction, because the tail can also be clipped by
        sentence tokenisation rather than by the caller.
        """
        text = content.strip()
        if not text:
            return None
        for line in self._echo:
            if line == text:
                return line
            stripped = line.strip()
            if len(text) >= 8 and (stripped.startswith(text) or text.startswith(stripped)):
                return line
        return None

    def begin_user_turn(self) -> bool:
        """Open a new turn. False if this looks like a split fragment of the last one."""
        # Cleared before the split-fragment return below: a transcript arriving
        # means that utterance ended, whether or not it opens a new turn. Leaving
        # it set on a run of fragments would suppress every filler for the rest
        # of the call.
        self._user_speaking = False
        if time.monotonic() - self._last_agent_line_at < self._split_grace:
            return False
        self._close_turn_for_stall()
        self.turn += 1
        # Do not clear _speaking: the agent may still be talking. Cleared only
        # when agent_state leaves "speaking", or on barge-in.
        self._model_spoke = False
        self._top_played = 0
        self._kinds_this_turn.clear()
        return True

    def _close_turn_for_stall(self) -> None:
        """Score the agent turn that just ended: did it say anything new?"""
        if self._turn_lines:
            if self._turn_had_new:
                self._stall_run = 0
            else:
                self._stall_run += 1
                if self._stall_run >= self._stall_limit:
                    logger.warning(
                        f"[{self.label}] stalled: {self._stall_run} consecutive turns "
                        f"with nothing new said"
                    )
        self._turn_lines = 0
        self._turn_had_new = False

    def close(self) -> None:
        self._closed = True

    def _count_heard(self, content: str) -> None:
        """Tally what the caller actually heard, and track a run of repeats.

        Whoever produced the line is irrelevant here: the caller hearing the same
        sentence four times is the defect, and the model can loop on its own
        wording just as easily as it can echo one of ours.
        """
        key = _norm(content)
        if len(key) < _STALL_MIN_CHARS:
            return
        self._heard[key] += 1
        # Per TURN, not per message. Counting every message meant the model's
        # stock closer -- "How can I help you today?" -- made small talk look
        # productive: on Path A 13 off-topic turns was enough to mute the agent,
        # which then answered nothing at all, including the pharmacy question
        # that followed. That is the exact shape of the stress test we are asked
        # to survive.
        #
        # A turn that contains ANY line not heard before is a turn that moved, and
        # a conversation that moves is not stalled. The 846-turn call this guard
        # was built for had turns containing nothing but the one repeated line, so
        # it is still caught.
        self._turn_lines += 1
        if self._heard[key] == 1:
            self._turn_had_new = True

    # --- queries the session asks --------------------------------------------

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def user_speaking(self) -> bool:
        """Caller is mid-utterance, as far as we know.

        Bounded the same way the internal check is: a VAD trigger whose transcript
        never arrives must not read as "still speaking" for the rest of the call.
        """
        return (self._user_speaking
                and time.monotonic() - self._user_speaking_at < _USER_SPEAKING_MAX_S)

    @property
    def model_spoke(self) -> bool:
        return self._model_spoke

    @property
    def stalled(self) -> bool:
        """The agent has said nothing new for `stall_limit` turns running."""
        return self._stall_run >= self._stall_limit

    @property
    def repeated_lines(self) -> list[tuple[str, int]]:
        """Every line the caller heard more than once, most repeated first."""
        return [(line, n) for line, n in self._heard.most_common() if n > 1]

    # --- the one way to speak -------------------------------------------------

    async def say(
        self,
        message: str,
        *,
        kind: str,
        priority: Priority,
        cooldown_s: float = 0.0,
        turn: int | None = None,
        interruptible: bool = True,
    ) -> "SpeechHandle | None":
        """Speak a server line, or drop it. `turn` drops a stale watchdog fire.

        Returns the SpeechHandle when the line went out, None when it was dropped.
        Callers that must know the line was HEARD -- the compliance greeting, the
        closing -- await the handle and check `interrupted`; everything else can
        ignore the return value and treat it as a bool.

        `interruptible=False` is the replacement for Path A's inject-and-confirm
        retry loop. A line that may not be talked over simply says so.
        """
        drop = self._why_not(kind, priority, cooldown_s, turn, message)
        if drop:
            logger.info(f"[{self.label}] speech[{kind}] dropped: {drop}")
            return None
        if self._session is None:
            logger.warning(f"[{self.label}] speech[{kind}] dropped: no session attached")
            return None

        self._said_at[kind] = time.monotonic()
        self._kinds_this_turn.add(kind)
        self._top_played = max(self._top_played, int(priority))
        # count now, not when the echo returns: two lines can be scheduled before
        # either conversation item lands, and the cap has to hold across that gap
        self._count_heard(message)
        self._echo.append(message)
        self._spoken.append(message)
        self._metrics.on_injection(kind)
        logger.info(f"[{self.label}] speech[{kind}]: {message!r}")
        try:
            return self._session.say(message, allow_interruptions=interruptible)
        except RuntimeError as e:
            # say() raises if the session is closing. Not worth a traceback on a
            # call that is already over.
            logger.warning(f"[{self.label}] speech[{kind}] failed: {e}")
            return None

    def _why_not(
        self, kind: str, priority: Priority, cooldown_s: float, turn: int | None,
        message: str = "",
    ) -> str | None:
        if self._closed:
            return "session closed"
        if turn is not None and turn != self.turn:
            return f"stale (minted for turn {turn}, now on {self.turn})"
        # A per-call ceiling on any one line, whatever its priority. The
        # per-turn and cooldown checks below both reset, which is how "Take your
        # time." reached 68 repetitions on a single call; this one does not.
        heard = self._heard.get(_norm(message), 0)
        if message and heard >= self._max_repeats:
            return f"line already spoken {heard}x this call"
        if cooldown_s:
            since = time.monotonic() - self._said_at.get(kind, -1e9)
            if since < cooldown_s:
                return f"cooldown ({since:.1f}s of {cooldown_s:.1f}s)"
        # Bounded: a VAD trigger whose transcript never arrives must not silence
        # the server for the rest of the call. 15s is past any real utterance.
        if (priority in (Priority.CHECKIN, Priority.FILLER) and self._user_speaking
                and time.monotonic() - self._user_speaking_at < _USER_SPEAKING_MAX_S):
            return "caller is still speaking"
        if priority is Priority.CHECKIN:
            if self._speaking:
                return "agent is already speaking"
            if kind in self._kinds_this_turn:
                return "already played this kind on this turn"
            return None
        if priority is Priority.FILLER:
            if self._speaking:
                return "agent is already speaking"
            if self._model_spoke:
                return "model already answered this turn"
            if self._top_played >= Priority.ENFORCE:
                return "a scripted line already covered this turn"
            if kind in self._kinds_this_turn:
                return "already played this kind on this turn"
        return None
