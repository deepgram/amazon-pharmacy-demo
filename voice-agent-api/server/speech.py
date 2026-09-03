"""Arbiter for server-injected agent speech (filler, check-in, scripted lines).

All InjectAgentMessage traffic goes through SpeechGate.say(). `speaking` is
driven by audio frames and AgentAudioDone; `model_spoke` is true when a
ConversationText is not an echo of something this gate injected.

The gate also counts every line the caller hears, whoever produced it, and
exposes `stalled` when the agent has spent several consecutive turns saying
things it has already said. The session uses that to stop feeding the socket --
see the turn-admission note in config.py.
"""

import re
import time
from collections import Counter, deque
from enum import IntEnum
from typing import Awaitable, Callable

from loguru import logger


class Priority(IntEnum):
    FILLER = 10    # drop if anything else is already covering the silence
    CHECKIN = 15   # caller went quiet after we finished
    ENFORCE = 20   # price disclaimer / day-supply conversion
    SCRIPTED = 30  # verification / transfer lines


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

# How long a UserStartedSpeaking with NO transcript behind it keeps the server
# quiet. This is a false-trigger bound, not a normal operating window: a real
# utterance produces a transcript within a second or two, so silence past that is
# the VAD having heard a breath or the room, not a caller mid-sentence.
#
# It was 15s, chosen only as a backstop against permanent silence, and it behaved
# like an operating window on a live call. The caller said "I actually need to
# refill my Levothyroxine", the VAD retriggered with nothing behind it, and the
# hold phrase was refused with "caller is still speaking" -- so they got 8.4
# seconds of dead air before the agent spoke, on a turn the agent answered in 1.3s
# once it had a complete question.
#
# 3s is past any plausible transcript delay and well short of a silence a caller
# would notice. The cost is that someone pausing longer than three seconds
# mid-sentence can be spoken over; barge-in stops the agent immediately, and that
# is a far cheaper failure than the agent going mute.
_USER_SPEAKING_MAX_S = 3.0


def _norm(content: str) -> str:
    """Compare on words alone, so punctuation and casing cannot hide a repeat."""
    return re.sub(r"[^a-z0-9 ]+", "", content.lower()).strip()


class SpeechGate:
    def __init__(
        self,
        label: str,
        send_json: Callable[[dict], Awaitable[None]],
        metrics,
        split_turn_grace_s: float,
        max_line_repeats: int = 2,
        stall_limit: int = 3,
    ):
        self.label = label
        self._send_json = send_json
        self._metrics = metrics
        self._split_grace = split_turn_grace_s
        self._max_repeats = max_line_repeats
        self._stall_limit = stall_limit

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
        self._turn_caller_spoke = False  # the CALLER said something real this turn
        self.duplicates: list[str] = []
        self.narrations: list[str] = []
        self._closed = False

    # --- state the session feeds in ------------------------------------------

    def on_agent_audio(self) -> None:
        """A binary frame arrived. The agent is audibly talking."""
        self._speaking = True

    def on_agent_started(self) -> None:
        self._speaking = True

    def on_agent_audio_done(self) -> None:
        self._speaking = False

    def on_user_started_speaking(self) -> None:
        """Barge-in. Deepgram stops the agent, so it is no longer speaking."""
        self._speaking = False
        # And the CALLER now is. `_speaking` only ever tracked the agent, so
        # nothing here knew to hold back while someone was mid-sentence: on a
        # long turn the hold phrase fired straight over them and the API threw it
        # away with INJECT_AGENT_MESSAGE_DURING_USER_SPEECH. Twice on one call.
        # Relying on Deepgram to reject it is luck; the silence check-in has no
        # such protection and would happily start "Are you still there?" over a
        # caller still talking.
        self._user_speaking = True
        self._user_speaking_at = time.monotonic()

    def on_agent_text(self, content: str) -> None:
        """An assistant ConversationText landed. Ours, or the model's?"""
        self._last_agent_line_at = time.monotonic()
        if content in self._echo:
            self._echo.remove(content)   # our own injection, counted at say() time
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
        # by AgentAudioDone / UserStartedSpeaking.
        self._model_spoke = False
        self._top_played = 0
        self._kinds_this_turn.clear()
        return True

    def note_user_line(self, content: str) -> None:
        """A real caller utterance this turn -- see the stall test below.

        Length-gated the same way agent lines are: a VAD trigger that produces a
        two-word fragment is noise, not a person talking.
        """
        if len(_norm(content)) >= _STALL_MIN_CHARS:
            self._turn_caller_spoke = True

    def _close_turn_for_stall(self) -> None:
        """Score the agent turn that just ended: is this conversation stuck?

        This guard exists for one thing: the agent talking to NOBODY. The call it
        was built for had no caller -- background noise tripped the VAD 846 times
        and the agent answered each one with the same sentence, degrading to
        "Please call 91 right way" by the end.

        So the test is whether a person is there, not whether the agent is being
        interesting. The scope redirect is deliberately short and repetitive ("I'm
        the Amazon Pharmacy agent and can only help with pharmacy questions"), and
        muting is a terrible answer to someone making small talk: the agent then
        answers NOTHING, including the pharmacy question that follows -- which is
        exactly the stress test this demo is graded on. Measured twice: muted at
        turn 15, and at turn 22 once repetition alone stopped counting.

        A caller producing real utterances keeps the call alive. MAX_TURNS still
        bounds a genuinely runaway conversation.
        """
        if self._turn_caller_spoke:
            self._stall_run = 0
            self._turn_lines = 0
            self._turn_had_new = False
            self._turn_caller_spoke = False
            return
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
        self._turn_caller_spoke = False

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
        # Per TURN, not per frame. This counted every ConversationText, and the
        # model ends most replies with a stock closer -- "How can I help you
        # today?" -- so on small talk the agent said something new every turn and
        # was muted anyway: 13 off-topic turns was enough, and it then answered
        # nothing at all, including the pharmacy question that followed. That is
        # the exact shape of the stress test we are asked to survive.
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
        behavior: str,
        cooldown_s: float = 0.0,
        turn: int | None = None,
    ) -> bool:
        """Speak a server line, or drop it. `turn` drops a stale watchdog fire."""
        drop = self._why_not(kind, priority, cooldown_s, turn, message)
        if drop:
            logger.info(f"[{self.label}] speech[{kind}] dropped: {drop}")
            return False

        self._said_at[kind] = time.monotonic()
        self._kinds_this_turn.add(kind)
        self._top_played = max(self._top_played, int(priority))
        # count now, not when the echo returns: two injections can land before
        # either ConversationText does, and the cap has to hold across that gap
        self._count_heard(message)
        self._echo.append(message)
        self._spoken.append(message)
        self._metrics.on_injection(kind)
        logger.info(f"[{self.label}] speech[{kind}/{behavior}]: {message!r}")
        await self._send_json({
            "type": "InjectAgentMessage", "message": message, "behavior": behavior,
        })
        return True

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
