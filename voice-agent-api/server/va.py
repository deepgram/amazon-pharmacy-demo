"""Deepgram Voice Agent session. One VoiceAgentSession per call; transport-agnostic.

Settings first, then FunctionCallRequest/Response for client-side tools.
KeepAlive every 8s while idle. InjectUserMessage is used for DTMF.
"""

import asyncio
import base64
import json
import re
import time
from typing import Awaitable, Callable

import websockets
from loguru import logger

from . import catalog, prompt, spoken, tools
from .config import (
    DEEPGRAM_API_KEY,
    DISCLAIMER_GRACE_S,
    EMERGENCY_END_S,
    EMERGENCY_DENIALS,
    EMERGENCY_EVIDENCE,
    EMERGENCY_MARKERS,
    GOODBYE_MARKERS,
    GOODBYE_MAX_CHARS,
    STILL_WORKING_MARKERS,
    OFFER_AGENT_MARKERS,
    DISCLAIMER_MAX_WAIT_S,
    DISCLAIMER_ESTIMATE_MARKERS,
    DISCLAIMER_VARIANCE_MARKERS,
    DISCLAIMER_WHOLE_MARKERS,
    EAGER_EOT_THRESHOLD,
    EOT_THRESHOLD,
    EOT_TIMEOUT_MS,
    EXPRESSIVITY,
    HOLD_AFTER_S,
    HOLD_PHRASE_2,
    HOLD_ATTEMPTS,
    HOLD_PHRASES,
    HOLD_RETRY_S,
    LINE_CLOSING,
    LINE_CLOSING_MARKER,
    LINE_LOCKOUT,
    LISTEN_ENDPOINTING_MS,
    LISTEN_KEYTERMS,
    LISTEN_MODEL,
    MAX_LINE_REPEATS,
    MAX_SILENCE_NUDGES,
    MAX_TURNS,
    MIP_OPT_OUT,
    OTP_SILENCE_FACTOR,
    PRICE_DISCLAIMER,
    REASONING_MODE,
    SCRIPTED_LINE_COOLDOWN_S,
    SCRIPTED_TOOL_LINES,
    SILENCE_AFTER_S,
    SILENCE_GOODBYE,
    SILENCE_NUDGES,
    SILENCE_NUDGES_2,
    SILENCE_NUDGES_OTP,
    SPEAK_SPEED,
    SPEAK_VERSION,
    SPLIT_TURN_GRACE_S,
    STALL_LIMIT,
    SUPPLY_CONVERSION_TEMPLATE,
    TAGS,
    THINK_MODEL,
    THINK_PROVIDER,
    THINK_TEMPERATURE,
    VA_BASIC_AUTH,
    VA_URL,
    VOICE_MODEL,
)
from .metrics import CallMetrics
from .traces import CallTrace
from .speech import Priority, SpeechGate

# Conversation phrases Flux tends to mangle over 8kHz telephony audio.
EXTRA_KEYTERMS = [
    "ninety days", "thirty days", "eighty four days", "day supply",
    "ninety day supply", "thirty day supply", "change my mind",
    "twelve point five", "milligrams", "micrograms", "verification code",
    "refill", "prescription", "insurance", "coupon",
]

SETTINGS_TIMEOUT_S = 10
LISTEN_FALLBACK_MODEL = "flux-general-en"

# First clause of prompt.GREETING. Enough to recognise the opening in a
# ConversationText frame; the API splits the full line across several.
GREETING_MARKER = "Thank you for calling Amazon Pharmacy"
# The verification question is now its own turn, so it needs its own proof of
# delivery -- the greeting landing no longer implies the caller was asked.
NUMBER_CHECK_MARKER = "associated with your account"

# The opening is re-sent until a ConversationText comes back carrying the marker.
# Two attempts, because the failure it covers is a one-off race with barge-in
# during slow synthesis, not a dead speak path -- if the second is lost too,
# something else is wrong and repeating it further only talks over the caller.
GREETING_ATTEMPTS = 2
GREETING_CONFIRM_S = 6.0     # past the measured 5s SLOW_SPEAK_REQUEST threshold
GREETING_QUIET_WAIT_S = 8.0  # how long to wait for a gap before injecting

# The price facts are enforced, not merely attempted: a caller talking over the
# estimate caveat had it discarded by the API and nothing noticed.
PRICE_FACTS_ATTEMPTS = 3
PRICE_FACTS_CONFIRM_S = 8.0     # wait for the words to come back before retrying
PRICE_FACTS_GAP_WAIT_S = 12.0   # hunt for silence before a retry

# The verification question is queued BEHIND the ~15s greeting, so "did it land?"
# cannot be answered by a short timer -- a 6s window expires while the line is
# still waiting its turn, and the retry then queues a second copy that the caller
# hears asked twice. Wait this long overall, and only treat it as lost once the
# channel has been quiet for a beat with the greeting already confirmed.
NUMBER_CHECK_ATTEMPTS = 4        # barge-in can swallow several in a row
NUMBER_CHECK_MAX_WAIT_S = 25.0   # how long to hunt for a gap before retrying
NUMBER_CHECK_CONFIRM_S = 20.0    # long enough to clear the greeting it queues behind
NUMBER_CHECK_QUIET_S = 1.0

# The closing is injected and then the channel is cut, so the mute has to wait
# for the line to be spoken. Generous, because the Aura-2 pool has taken over 5s
# to start a long utterance; bounded so a lost line cannot hold the socket open.
CLOSING_CONFIRM_S = 12.0
CLOSING_DRAIN_S = 3.0        # let the audio finish after the text lands


class SettingsRejected(RuntimeError):
    """The Voice Agent API would not start a session with these Settings."""


def _auth_header() -> str:
    """Token for production; HTTP Basic when VA_BASIC_AUTH is set (internal stems)."""
    if VA_BASIC_AUTH:
        return "Basic " + base64.b64encode(VA_BASIC_AUTH.encode()).decode()
    return f"Token {DEEPGRAM_API_KEY}"


def build_settings(audio_in: dict, audio_out: dict, greeting: bool = True,
                   think_provider: str | None = None, think_model: str | None = None,
                   listen_model: str | None = None, keyterms: bool | None = None) -> dict:
    listen_model = listen_model or LISTEN_MODEL
    is_flux = listen_model.startswith("flux")
    listen_provider: dict = {"type": "deepgram", "model": listen_model}
    # `auto` now means on for Nova as well as Flux. "RxPass" and "PillPack" have
    # no fallback -- the ASR splits them into two ordinary words and the
    # knowledge-base lookup lands on the wrong entry. connect() drops keyterms
    # and keeps the model if the API will not take them, so this cannot cost us
    # nova-3-medical_pharma.
    want_keyterms = LISTEN_KEYTERMS == "on" or (LISTEN_KEYTERMS == "auto" and keyterms is not False)
    if is_flux:
        listen_provider.update(
            version="v2",
            eot_threshold=EOT_THRESHOLD,
            eot_timeout_ms=EOT_TIMEOUT_MS,
        )
        if EAGER_EOT_THRESHOLD:
            listen_provider["eager_eot_threshold"] = min(EAGER_EOT_THRESHOLD, EOT_THRESHOLD)
    elif LISTEN_ENDPOINTING_MS:
        listen_provider["endpointing"] = LISTEN_ENDPOINTING_MS
    if want_keyterms:
        listen_provider["keyterms"] = catalog.keyterms() + EXTRA_KEYTERMS
    think_prov: dict = {
        "type": think_provider or THINK_PROVIDER,
        "model": think_model or THINK_MODEL,
    }
    is_gpt5 = think_prov["type"] == "open_ai" and think_prov["model"].startswith("gpt-5")
    if is_gpt5:
        if REASONING_MODE:
            think_prov["reasoning_mode"] = REASONING_MODE
    else:
        think_prov["temperature"] = THINK_TEMPERATURE

    speak_provider: dict = {
        "type": "deepgram",
        "version": SPEAK_VERSION,
        "model": VOICE_MODEL,
    }
    # Aura-2 does not support `speed`, so never send it for an aura-2-* voice --
    # the model returns 400, which SageMaker wraps as a 424 and which reads for
    # all the world like an unhealthy container. Measured on identical Settings:
    # with the field, 463 consecutive 400s and no audio; without it, 768 KB of
    # clean audio and zero errors. `version` IS required on that path; dropping
    # it 400s the same way.
    #
    # For voices that do support it, still omit the default: at 1.0 the field
    # buys nothing and only widens the surface.
    if not VOICE_MODEL.startswith("aura-2-") and SPEAK_SPEED != 1.0:
        speak_provider["speed"] = SPEAK_SPEED
    # expressivity is Flux v2 only (beta); Aura rejects it
    if SPEAK_VERSION == "v2" and EXPRESSIVITY:
        speak_provider["expressivity"] = EXPRESSIVITY

    settings: dict = {
        "type": "Settings",
        "audio": {"input": audio_in, "output": audio_out},
        "agent": {
            "language": "en",
            "listen": {"provider": listen_provider},
            "think": {
                "provider": think_prov,
                "prompt": prompt.build(),
                "functions": tools.DEFINITIONS,
            },
            "speak": {"provider": speak_provider},
        },
        "tags": TAGS,
        "mip_opt_out": MIP_OPT_OUT,
    }
    if greeting:
        settings["agent"]["greeting"] = prompt.GREETING
    return settings


class VoiceAgentSession:
    def __init__(
        self,
        label: str,
        audio_in: dict,
        audio_out: dict,
        toolbox: tools.ToolBox,
        on_audio: Callable[[bytes], Awaitable[None]],
        on_event: Callable[[dict], Awaitable[None]],
        on_tool: Callable[[], Awaitable[None]] | None = None,
        on_end_call: Callable[[str], Awaitable[None]] | None = None,
        think_provider: str | None = None,
        think_model: str | None = None,
        coval_simulation_id: str | None = None,
    ):
        self.label = label
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.toolbox = toolbox
        self.on_audio = on_audio        # agent speech -> transport
        self.on_event = on_event        # JSON events -> transport (barge-in, transcript, latency)
        self.on_tool = on_tool          # fired after each tool runs, so the UI can show OTP state
        # fired once with the reason when the call reaches a terminal state. The
        # scripted-call harnesses need this to know when to stop injecting turns;
        # both were passing it and neither VoiceAgentSession version ever
        # accepted it, so run_scenario.py and compare_think_models.py raised
        # TypeError on construction.
        self.on_end_call = on_end_call
        self._end_call_fired = False
        # per-session think override; used by scripts/compare_think_models.py
        self.think_provider = think_provider
        self.think_model = think_model
        self.metrics = CallMetrics(label)
        # Coval trace export. Inert unless a simulation id arrived on the
        # handshake AND COVAL_TRACES is on -- see server/traces.py.
        self.trace = CallTrace(coval_simulation_id, label)
        self.ws = None
        self.listen_model: str | None = None   # what actually started, post-fallback
        self._ready = asyncio.Event()
        self._settings_error: str | None = None
        self._closed = False
        self._last_sent = time.monotonic()
        self._tasks: list[asyncio.Task] = []
        # every server-spoken line goes through this; it owns "is the agent
        # talking", "did the model answer", and which turn we are on
        self.gate = SpeechGate(
            label, self._send_json, self.metrics, SPLIT_TURN_GRACE_S,
            max_line_repeats=MAX_LINE_REPEATS, stall_limit=STALL_LIMIT,
        )
        self._watchdog: asyncio.Task | None = None
        self._hold_i = 0                # rotates HOLD_PHRASES across turns
        self._idle_since: float | None = None   # when the caller last went quiet
        self._silence_stage = 0         # 0 nudge, 1 second nudge, 2 hangup
        self._nudge_i = 0               # rotates the check-in wording; never resets
        self._nudges_used = 0           # total check-ins this call, capped
        self._user_in_flight = False    # UserStartedSpeaking with no transcript yet
        self._user_started_at = 0.0
        self._disclaimer_said = False
        self._disclaimer_parts: set[str] = set()   # price disclaimer, once per call
        self._owed_supply: tuple[int, int] | None = None   # (requested, dispensed)
        self._supply_said = False
        self._supply_parts: set[str] = set()
        self._ending = False
        self._muted = False             # stop feeding caller audio to the socket
        self._greeting_said = False     # the opening carries the 911 and recording notices
        self._number_check_said = False  # ...and the verification question follows it
        self._awaiting_reply = False      # caller spoke; the model has not answered yet
        self._closing_said = False
        self._emergency_armed = False
        self._emergency_evidence = False   # caller gave a reason at some point      # the survey close, owed once on a normal end
        self._greeting_task: asyncio.Task | None = None
        self._mute_allowance: int | None = None   # replies still owed before muting
        self._mute_reason = ""

    async def connect(self):
        """Open the session, degrading one step at a time.

        Keyterms first, because they are what makes "RxPass" survive the channel;
        then the same model without them, because losing keyterms is a far smaller
        loss than dropping to flux-general-en, which mishears "Zepbound" as "sells
        that bound"; then the fallback model as the last resort.
        """
        ladder: list[tuple[str, bool | None]] = [(LISTEN_MODEL, None), (LISTEN_MODEL, False)]
        if LISTEN_MODEL != LISTEN_FALLBACK_MODEL:
            ladder.append((LISTEN_FALLBACK_MODEL, None))
        for i, (model, keyterms) in enumerate(ladder):
            try:
                await self._open(model, keyterms=keyterms)
                return
            except SettingsRejected as e:
                if i == len(ladder) - 1:
                    raise
                nxt_model, nxt_keyterms = ladder[i + 1]
                downgrade = (f"{nxt_model} without keyterms" if nxt_keyterms is False
                             else nxt_model)
                logger.error(
                    f"[{self.label}] Settings rejected for {model!r}"
                    f"{'' if keyterms is False else ' with keyterms'} ({e}); "
                    f"retrying with {downgrade}"
                )
            await self._teardown()

    async def _open(self, listen_model: str, keyterms: bool | None = None):
        self._ready.clear()
        self._settings_error = None
        self.ws = await websockets.connect(
            VA_URL,
            extra_headers={"Authorization": _auth_header()},
            max_size=2**23,
        )
        await self._send_json(build_settings(
            self.audio_in, self.audio_out,
            think_provider=self.think_provider, think_model=self.think_model,
            listen_model=listen_model, keyterms=keyterms,
        ))
        self._tasks.append(asyncio.create_task(self._pump()))
        self._tasks.append(asyncio.create_task(self._keepalive()))
        self._tasks.append(asyncio.create_task(self._caller_idle_watchdog()))
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=SETTINGS_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise SettingsRejected(f"no SettingsApplied within {SETTINGS_TIMEOUT_S}s")
        if self._settings_error:
            raise SettingsRejected(self._settings_error)
        self.listen_model = listen_model
        self.metrics.listen_model = listen_model
        logger.info(f"[{self.label}] VA session ready (listen={listen_model})")

    async def _teardown(self):
        """Drop a half-open session so connect() can retry on a clean socket."""
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:  # already dead; the retry is what matters
                pass
            self.ws = None

    async def send_audio(self, chunk: bytes):
        # Muted means the agent is done talking on this call: stop feeding the
        # socket so the model is never handed another turn. Dropping the audio
        # here rather than suppressing replies further down is the only place
        # that works, because the model answers every end of turn the Voice
        # Agent API detects and the server has no say in that.
        if self._muted:
            return
        if self.ws and not self._closed:
            await self.ws.send(chunk)
            self._last_sent = time.monotonic()

    def _mute(self, reason: str):
        """Stop giving the model turns. Irreversible for the rest of the call."""
        if self._muted:
            return
        self._muted = True
        self._mute_allowance = None
        self.metrics.muted_reason = reason
        self._cancel_watchdog()
        logger.info(
            f"[{self.label}] muted after {self.gate.turn} turns ({reason}); "
            "the transport stays open for the caller to disconnect"
        )

    def _arm_mute(self, reason: str, after_turns: int = 0):
        """Mute now, or after `after_turns` more replies for a script that owes one."""
        if self._muted:
            return
        if after_turns <= 0:
            self._mute(reason)
            return
        if self._mute_allowance is None or after_turns < self._mute_allowance:
            self._mute_allowance = after_turns
            self._mute_reason = reason
            logger.info(f"[{self.label}] muting after {after_turns} more reply ({reason})")

    async def inject_user_text(self, text: str):
        # Muting has to cover this too, not just audio: a keypad press after the
        # call has ended would otherwise hand the model a turn by the back door.
        if self._muted:
            logger.info(f"[{self.label}] dropped injected text after mute: {text!r}")
            return
        await self._send_json({"type": "InjectUserMessage", "content": text})

    async def close(self):
        self._closed = True
        self.gate.close()
        self.metrics.duplicate_lines = list(self.gate.duplicates)
        self.metrics.narrated_lines = list(self.gate.narrations)
        self.metrics.repeated_lines = self.gate.repeated_lines
        self._cancel_watchdog()
        for t in self._tasks:
            t.cancel()
        if self.ws:
            await self.ws.close()
        self.metrics.finish()

    async def _send_json(self, obj: dict):
        if self.ws and not self._closed:
            await self.ws.send(json.dumps(obj))
            self._last_sent = time.monotonic()

    async def _keepalive(self):
        # docs: one KeepAlive every 8 seconds while idle
        while not self._closed:
            await asyncio.sleep(4)
            if time.monotonic() - self._last_sent > 7:
                await self._send_json({"type": "KeepAlive"})

    async def _pump(self):
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    self.gate.on_agent_audio()
                    self.metrics.on_agent_audio()
                    await self.on_audio(msg)
                    continue
                event = json.loads(msg)
                await self._handle(event)
        except websockets.ConnectionClosed as e:
            if not self._closed:
                logger.warning(f"[{self.label}] VA socket closed: {e}")
            # a Settings rejection can close the socket instead of answering;
            # release connect() now rather than letting it burn the full timeout
            if not self._ready.is_set():
                self._settings_error = f"socket closed before SettingsApplied: {e}"
                self._ready.set()
        except asyncio.CancelledError:
            pass

    async def _handle(self, event: dict):
        etype = event.get("type")

        if etype == "SettingsApplied":
            self._ready.set()
            # Queue the verification question straight away. behavior="queue" puts
            # it behind the greeting rather than over it, and sending it now means
            # the API synthesises it while the greeting is still playing -- so the
            # caller hears one continuous opening with no gap between the two.
            self._tasks.append(asyncio.create_task(self._ask_number_check()))

        elif etype == "Welcome":
            logger.info(f"[{self.label}] VA request_id={event.get('request_id')}")

        elif etype == "UserStartedSpeaking":
            self.gate.on_user_started_speaking()
            self.metrics.on_user_speaking()
            self._user_in_flight = True
            self._user_started_at = time.monotonic()
            self._idle_since = None

        elif etype == "ConversationText":
            role, content = event.get("role"), event.get("content", "")
            self.metrics.on_text(role, content)
            logger.info(f"[{self.label}] {role}: {content}")
            if role == "user":
                # Keep the caller's latest complete transcript available to
                # verify_otp. This covers a model that heard the code but sent
                # incomplete function arguments.
                self.toolbox.latest_user_text = content
                self.gate.note_user_line(content)
                self._awaiting_reply = True
                if not self._emergency_evidence and self._is_emergency(content):
                    self._emergency_evidence = True
                self._user_in_flight = False
                self._idle_since = None
                self._silence_stage = 0
                self._cancel_watchdog()
                # A caller who says "hello" the moment the line opens cancels the
                # greeting the API was about to deliver, and the model improvises
                # an opening of its own -- one observed live was "Hello! Welcome
                # to Amazon Pharmacy. I'm here to help you with medications,
                # pricing, orders, and prescriptions." That is not a cosmetic
                # loss: the scripted opening is where the emergency instruction
                # and the call-recording disclosure live, so it must not be
                # skippable by talking first.
                if not self._greeting_said and not self._greeting_task:
                    self._greeting_task = asyncio.create_task(self._ensure_greeting())
                    self._tasks.append(self._greeting_task)
                # A deferred mute counts CALLER turns, not assistant frames: the
                # model splits one utterance across several ConversationText
                # messages, so counting those spent the emergency script's owed
                # confirmation on the second half of its own opening line. This
                # turn's audio is already through, so the reply to it still
                # happens -- that reply IS the confirmation -- and everything
                # after is dropped.
                if self._mute_allowance is not None:
                    self._mute_allowance -= 1
                    if self._mute_allowance <= 0:
                        self._mute(self._mute_reason)
                if self.gate.turn >= MAX_TURNS:
                    self._mute("turn ceiling")
                elif self.gate.begin_user_turn():
                    self._watchdog = asyncio.create_task(
                        self._hold_phrase_watchdog(self.gate.turn)
                    )
                else:
                    logger.info(f"[{self.label}] split user turn; not re-arming the watchdog")
            if role == "assistant":
                # The API delivers the greeting itself on a normal open, and it
                # arrives split across several frames, so match the distinctive
                # opening rather than the whole string.
                self._awaiting_reply = False
                if GREETING_MARKER in content:
                    self._greeting_said = True
                if NUMBER_CHECK_MARKER in content.lower():
                    self._number_check_said = True
                if LINE_CLOSING_MARKER in content.lower():
                    self._closing_said = True
                # The agent has just told this caller to hang up and dial 911.
                # Whether the model also calls end_call is now the server's
                # problem, not the prompt's -- see the note in config.py.
                # Record an offer of a live person, so a caller answering "yes"
                # to it still gets transferred. tools.transfer_to_agent refuses
                # a transfer nobody asked for.
                if any(m in content.lower() for m in OFFER_AGENT_MARKERS):
                    self.toolbox.offered_agent = True
                if (not self._emergency_armed
                        and any(m in content.lower() for m in EMERGENCY_MARKERS)):
                    self._emergency_armed = True
                    self._tasks.append(asyncio.create_task(self._emergency_backstop()))
                self.gate.on_agent_text(content)
                if self.gate.stalled:
                    self._mute("stalled")
                if not self._disclaimer_said:
                    # Both halves, tracked across frames: the model routinely splits
                    # "that's an estimate" and "your price may differ" into separate
                    # ConversationText messages, so neither one alone proves anything.
                    low = content.lower()
                    if any(m in low for m in DISCLAIMER_WHOLE_MARKERS):
                        self._disclaimer_parts.update({"estimate", "variance"})
                    if any(m in low for m in DISCLAIMER_ESTIMATE_MARKERS):
                        self._disclaimer_parts.add("estimate")
                    if any(m in low for m in DISCLAIMER_VARIANCE_MARKERS):
                        self._disclaimer_parts.add("variance")
                    if {"estimate", "variance"} <= self._disclaimer_parts:
                        self._disclaimer_said = True
                if self._owed_supply and not self._supply_said:
                    # Both numbers, not just the dispensed one. This checked only
                    # for "eighty four", so "dispensed as an eighty four day
                    # supply" counted as disclosure -- and a caller who asked for
                    # NINETY days never heard what became of their request, while
                    # the server concluded it had been told. The fact being
                    # disclosed is the relationship between the two numbers, so
                    # neither half alone can satisfy it.
                    #
                    # Tracked across messages: the model splits one sentence over
                    # several ConversationText frames, so "a ninety day request
                    # dispenses as" and "an eighty four day supply" often arrive
                    # separately and neither frame carries both.
                    # Hyphens folded to spaces: the model writes "ninety-day request"
                    # about as often as "ninety day request", and matching only the
                    # spaced form made the server repeat a conversion the model had
                    # just given -- the caller heard it twice.
                    low = re.sub(r"[\s\-]+", " ", content.lower())
                    for part, n in (("req", self._owed_supply[0]),
                                    ("disp", self._owed_supply[1])):
                        # Anchored on "day", not the bare number. Matching "ninety"
                        # alone, Zepbound's own cash price defeated this: "nine
                        # hundred ninety nine dollars" contains "ninety", so the
                        # server decided the ninety-to-eighty-four conversion had
                        # been disclosed and said nothing -- on a caller who asked
                        # for ninety days and was quoted eighty four. The fact is
                        # about a DAY SUPPLY, so the word "day" belongs in the test.
                        words = spoken.integer(n)
                        if f"{words} day" in low or f"{n} day" in low:
                            self._supply_parts.add(part)
                    if {"req", "disp"} <= self._supply_parts:
                        self._supply_said = True

        elif etype == "AgentStartedSpeaking":
            self.gate.on_agent_started()
            self.metrics.on_agent_started(event)

        elif etype == "AgentAudioDone":
            self.gate.on_agent_audio_done()
            if not self._closed:
                self._idle_since = time.monotonic()

        elif etype == "InjectionRefused":
            # visible trail: distinguishes "hold phrase played" from "refused"
            logger.warning(f"[{self.label}] InjectionRefused: {event.get('message')}")

        elif etype == "LatencyReport":
            self.metrics.on_latency_report(event)
            self.trace.latency_report(event, len(self.metrics.turns))

        elif etype == "FunctionCallRequest":
            for fn in event.get("functions", []):
                if not fn.get("client_side", False):
                    continue
                asyncio.create_task(self._run_function(fn))

        elif etype in ("Error", "Warning"):
            code, desc = event.get("code"), event.get("description")
            logger.error(f"[{self.label}] VA {etype}: {desc} ({code})")
            self.metrics.on_api_warning(etype, code, desc)
            if etype == "Error" and not self._ready.is_set():
                self._settings_error = f"{code}: {desc}"
                self._ready.set()

        await self.on_event(event)

    async def _ensure_price_facts(self):
        """Speak day-supply conversion and the estimate disclaimer if the model skipped them."""
        try:
            await asyncio.sleep(DISCLAIMER_GRACE_S)
            # Decide only once the model has actually stopped talking. Deciding on
            # a fixed 4s timer meant queueing our disclaimer while the model was
            # still mid-answer: on one call we queued at 21:25:10, the model said
            # its own version at 21:25:31, and ours played at 21:25:38 -- the
            # caller heard the estimate caveat twice, 7s apart. The model's price
            # answer routinely runs past the grace period, so wait it out.
            waited = 0.0
            while (self.gate.speaking and not self._closed
                   and waited < DISCLAIMER_MAX_WAIT_S):
                await asyncio.sleep(0.25)
                waited += 0.25
            if self._closed:
                return
            # These flags used to be set here, before the injection went out, which
            # made a REJECTED injection look exactly like a delivered one. On a live
            # call the caller talked over the estimate caveat, the API discarded it
            # with INJECT_AGENT_MESSAGE_DURING_USER_SPEECH, and the flag already said
            # it had been spoken -- so nothing retried and a graded disclosure was
            # silently missing. They are now set only by the ConversationText handler
            # when the words actually come back.
            for attempt in range(PRICE_FACTS_ATTEMPTS):
                missing = []
                if self._owed_supply and not self._supply_said:
                    req, disp = self._owed_supply
                    missing.append(SUPPLY_CONVERSION_TEMPLATE.format(
                        requested=spoken.integer(req), dispensed=spoken.integer(disp)))
                if PRICE_DISCLAIMER and not self._disclaimer_said:
                    missing.append(PRICE_DISCLAIMER)
                if not missing:
                    return
                if attempt:
                    # Retry only into a gap. The reason the first one was thrown away
                    # was that the caller was mid-sentence; sending again immediately
                    # just loses it the same way.
                    waited = 0.0
                    while waited < PRICE_FACTS_GAP_WAIT_S and not self._closed:
                        if not self.gate.speaking and not self.gate.user_speaking:
                            break
                        await asyncio.sleep(0.2)
                        waited += 0.2
                    if self._closed or self._muted:
                        return
                logger.info(
                    f"[{self.label}] enforcing price facts server-side: {len(missing)}"
                    f"{' (retry)' if attempt else ''}"
                )
                await self.gate.say(
                    " ".join(missing), kind="price_facts",
                    priority=Priority.ENFORCE, behavior="queue",
                )
                waited = 0.0
                while waited < PRICE_FACTS_CONFIRM_S and not self._closed:
                    if ((not self._owed_supply or self._supply_said)
                            and (not PRICE_DISCLAIMER or self._disclaimer_said)):
                        return
                    await asyncio.sleep(0.2)
                    waited += 0.2
            if not self._disclaimer_said and not self._closed:
                logger.warning(f"[{self.label}] price disclaimer never confirmed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[{self.label}] price-fact inject failed: {e}")

    def _cancel_watchdog(self):
        if self._watchdog and not self._watchdog.done():
            self._watchdog.cancel()

    @staticmethod
    def _is_goodbye(text: str) -> bool:
        """Is the caller signing off, rather than waiting on a lookup?

        Deliberately narrow. A question is never a sign-off however politely it is
        phrased ("thanks, what does that cost?"), and a long turn that merely ends
        warmly still deserves its filler -- so this needs a short utterance, no
        question, and an explicit farewell.
        """
        t = (text or "").strip().lower()
        if not t or len(t) > GOODBYE_MAX_CHARS or "?" in t:
            return False
        if any(m in t for m in STILL_WORKING_MARKERS):
            return False
        return any(m in t for m in GOODBYE_MARKERS)

    async def _hold_phrase_watchdog(self, turn: int):
        """Cover dead air after the caller stops, if nothing else already did."""
        try:
            await asyncio.sleep(HOLD_AFTER_S)
            if not self.toolbox.verified:
                return
            if self._ending or self.toolbox.end_call_reason:
                return
            # ...and the same check one beat earlier, from the caller's words.
            # The state check above only sees an end_call that has already
            # happened; on a sign-off the model is still composing the closing at
            # 2.8s, so nothing has been set yet and the filler goes out anyway.
            if self._is_goodbye(self.toolbox.latest_user_text):
                logger.info(f"[{self.label}] speech[hold] skipped: caller was signing off")
                return
            phrase = HOLD_PHRASES[self._hold_i % len(HOLD_PHRASES)]
            # A refusal here used to end the filler for the whole turn, and the most
            # common refusal is transient: "caller is still speaking", because the
            # VAD retriggered a second before the watchdog fired. On a live call that
            # cost 8.4 seconds of dead air -- the filler was refused once at 2.8s and
            # never asked again, while the caller waited. Keep asking while the turn
            # is still open and the model still has not answered; gate.say() drops a
            # stale attempt on its own once the turn moves on.
            spoke = False
            for _ in range(HOLD_ATTEMPTS):
                if self._closed or self._muted or self.gate.model_spoke:
                    return
                if self.gate.turn != turn:
                    return
                spoke = await self.gate.say(
                    phrase, kind="hold", priority=Priority.FILLER,
                    behavior="default", turn=turn,
                )
                if spoke:
                    break
                await asyncio.sleep(HOLD_RETRY_S)
            if not spoke:
                return
            self._hold_i += 1
            await asyncio.sleep(5)
            await self.gate.say(
                HOLD_PHRASE_2, kind="hold2", priority=Priority.FILLER,
                behavior="default", turn=turn,
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:  # e.g. socket closed mid-send; never crash the task
            logger.warning(f"[{self.label}] watchdog send failed: {e}")

    def _backstop_allowed(self) -> bool:
        """Should the backstop finish the job the model started?

        Defaults to YES, and that direction is the whole design. Requiring the
        caller to have used a recognised danger word was the first attempt and it
        failed on the second live case: someone said "my throat is closing up",
        the model correctly called it an emergency, and the guard did not
        recognise the words, so the call never ended. Symptoms are unbounded --
        any keyword list loses that race eventually, and losing it means the
        safety action does not complete.

        So the burden is reversed. The line was already spoken; stand down only
        where we can positively see it was spurious, which is the two shapes the
        150-call evaluation actually produced: fired into a caller who never
        spoke, or fired as the call was already ending. The prompt is the primary
        defence against the other two shapes (authorization pressure and
        third-party calls) and now handles them; this is the net beneath it.
        """
        if self._emergency_evidence:
            return True                      # caller described danger outright
        last = self.toolbox.latest_user_text
        if not last:
            return False                     # nobody has said anything at all
        if self._is_goodbye(last):
            return False                     # the call was wrapping up
        return True

    @staticmethod
    def _is_emergency(text: str) -> bool:
        """Did the caller give any reason to run the emergency script?

        Wide on purpose -- see the note in config.py. The only thing this rules out
        is silence and ordinary pharmacy talk.
        """
        t = (text or "").strip().lower()
        if not t:
            return False
        hits = [m for m in EMERGENCY_EVIDENCE if m in t]
        if not hits:
            return False
        # "Oh, no, this is not an emergency" matches only the word itself; that is a
        # denial, not a report. Any other hit and we take it at face value.
        if hits == ["emergency"] and any(d in t for d in EMERGENCY_DENIALS):
            return False
        return True

    async def _emergency_backstop(self) -> None:
        """End the call ourselves when the model says the emergency line but never
        calls end_call.

        The prompt asks for both: say "This sounds like an emergency..." and then
        call end_call with reason emergency. A model can do the first and skip the
        second, and then the session stays open and keeps answering turns after
        telling someone to dial 911.

        This does not decide anything the prompt would not have. It waits for the
        model's own end_call, and only if none arrives does it run the same ending
        the tool would have -- including the one owed follow-up turn, so "Did you
        hear me? Please call nine one one right away." can still be delivered.
        """
        await asyncio.sleep(EMERGENCY_END_S)
        if self._closed or self._ending or self.toolbox.end_call_reason:
            return
        # The line was spoken, but nobody asked for it. Ending the call here is how
        # a spurious trigger turns into a hang-up on a caller who is fine, so the
        # backstop stands down and the call carries on. Logged loudly: the line
        # should not have been said at all, and that is worth counting.
        if not self._backstop_allowed():
            logger.warning(
                f"[{self.label}] SPURIOUS emergency line -- fired into silence or a "
                f"sign-off; NOT ending the call (last caller turn: "
                f"{self.toolbox.latest_user_text!r})"
            )
            self.metrics.spurious_emergency = True
            return
        logger.warning(
            f"[{self.label}] emergency line spoken but no end_call after "
            f"{EMERGENCY_END_S:.0f}s; ending the call server-side"
        )
        self.toolbox.end_call_reason = "emergency"
        self._ending = True
        self._arm_mute("emergency:backstop", after_turns=1)
        if self.on_end_call and not self._end_call_fired:
            self._end_call_fired = True
            try:
                await self.on_end_call("emergency")
            except Exception as e:
                logger.warning(f"[{self.label}] on_end_call failed: {e}")

    async def _mute_after_closing(self, reason: str):
        """Hold the channel open until the closing has actually been spoken.

        Muting drops caller audio, and the Voice Agent API needs that stream to
        keep flowing to synthesise a queued injection -- so muting in the same
        breath as injecting the closing guarantees the caller never hears it.
        Wait for the marker to come back, then mute. Bounded, so a lost line
        cannot leave the socket open for the rest of the process.
        """
        waited = 0.0
        while (not self._closing_said and not self._closed
               and waited < CLOSING_CONFIRM_S):
            await asyncio.sleep(0.25)
            waited += 0.25
        if not self._closing_said:
            logger.warning(
                f"[{self.label}] closing never confirmed after {waited:.1f}s; "
                "muting anyway"
            )
        # a beat for the audio to drain after the text lands
        await asyncio.sleep(CLOSING_DRAIN_S)
        self._mute(reason)

    async def _ask_number_check(self) -> None:
        """Ask about the calling number, as a turn of its own after the greeting.

        Split out of GREETING so Amazon's compliance wording stays untouched while
        the verification question remains ours.

        The first attempt is queued at session open on purpose: the API synthesises
        it while the greeting plays, so the caller hears one continuous opening with
        no gap. Every attempt after that must wait for a GAP, which is the whole
        lesson of _ensure_greeting below -- a caller who talks over the opening
        causes barge-in to flush whatever is queued, and an injection sent into that
        window is discarded silently. Measured: on a barge-in three seconds in, both
        the original and a naive immediate retry were logged as sent and neither
        ever came back, so the caller was never asked their number and the call
        could not proceed.

        `_number_check_said` is set by the ConversationText handler when the line
        actually comes back -- and it matches the model's own phrasing too, so if
        the model asks the question itself we stop rather than ask it twice.
        """
        for attempt in range(NUMBER_CHECK_ATTEMPTS):
            if self._number_check_said or self._closed or self._muted:
                return
            # Never re-ask over an answer in flight. A caller barged into the
            # greeting with "I took too many pills and my chest hurts and I can't
            # breathe"; that flushed the queued question, this retry noticed and
            # re-asked -- 0.16s before the model produced the emergency line. The
            # caller heard "is this the mobile number on your account?" while the
            # transcript recorded "call nine one one right away". Nothing outranks
            # the emergency script, least of all a verification prompt.
            if self._ending or self.toolbox.end_call_reason or self._emergency_armed:
                return
            if attempt and self._awaiting_reply:
                waited = 0.0
                while self._awaiting_reply and waited < NUMBER_CHECK_MAX_WAIT_S:
                    if self._number_check_said or self._closed or self._muted:
                        return
                    if self._ending or self._emergency_armed:
                        return
                    await asyncio.sleep(0.2)
                    waited += 0.2
                if self._number_check_said or self._emergency_armed or self._ending:
                    return
            if attempt:
                # Retry only into silence, and only once the greeting itself has
                # landed -- the question follows the opening, it does not race it.
                waited = 0.0
                while waited < NUMBER_CHECK_MAX_WAIT_S and not self._closed:
                    if self._number_check_said or self._muted:
                        return
                    if self._greeting_said and not self.gate.speaking:
                        break
                    await asyncio.sleep(0.2)
                    waited += 0.2
                await asyncio.sleep(NUMBER_CHECK_QUIET_S)   # let the gap settle
                if self._number_check_said or self._closed or self._muted:
                    return
            await self.gate.say(
                prompt.LINE_NUMBER_CHECK, kind="number_check",
                priority=Priority.SCRIPTED, behavior="queue",
            )
            # Confirmation cannot be a short timer: the first attempt sits behind a
            # ~15s greeting, so a six second window expires while it is still queued.
            waited = 0.0
            while waited < NUMBER_CHECK_CONFIRM_S and not self._closed:
                if self._number_check_said:
                    return
                await asyncio.sleep(0.2)
                waited += 0.2
        if not self._number_check_said and not self._closed and not self._muted:
            logger.warning(f"[{self.label}] verification question never landed")

    async def _ensure_greeting(self):
        """Deliver the opening, and confirm it was actually heard.

        Injecting once is not enough on the self-hosted Aura-2 path. Greeting
        synthesis there runs past five seconds -- the API logs SLOW_SPEAK_REQUEST
        -- so a caller saying "hello" lands inside that window, barge-in cancels
        the pending greeting, and an injection sent in the same instant is
        flushed with it. Observed live: `speech[greeting/queue]` in the log with
        no ConversationText for it, and the model's improvised welcome playing
        instead. The 911 instruction and the recording disclosure ride on this
        line, so it has to be confirmed rather than assumed.

        `_greeting_said` is set by the ConversationText handler when the marker
        actually comes back, never here.
        """
        for attempt in range(GREETING_ATTEMPTS):
            if self._greeting_said or self._closed or self._muted:
                return
            # Land it in a gap. Injecting while the agent is mid-utterance is how
            # the first attempt was lost.
            waited = 0.0
            while self.gate.speaking and waited < GREETING_QUIET_WAIT_S and not self._closed:
                await asyncio.sleep(0.2)
                waited += 0.2
            if self._greeting_said or self._closed or self._muted:
                return
            await self.gate.say(
                prompt.GREETING, kind=f"greeting{attempt or ''}",
                priority=Priority.SCRIPTED, behavior="queue",
            )
            await asyncio.sleep(GREETING_CONFIRM_S)
        if not self._greeting_said:
            logger.warning(
                f"[{self.label}] greeting never confirmed after {GREETING_ATTEMPTS} "
                "attempts; the caller may not have heard the 911 or recording notice"
            )

    def _reading_code(self) -> bool:
        """Between send_otp and the first verify_otp: the caller is on their phone."""
        return (
            bool(self.toolbox.otp)
            and not self.toolbox.verified
            and self.toolbox.verify_attempts == 0
        )

    async def _caller_idle_watchdog(self):
        """Nudge, then hang up, if the caller stays quiet after we finish speaking."""
        if SILENCE_AFTER_S <= 0:
            return
        false_vad_s = 6.0
        try:
            while not self._closed:
                await asyncio.sleep(0.5)
                if self._closed or self._muted or self._ending or self.toolbox.end_call_reason:
                    return
                if self.toolbox.in_queue:
                    continue
                if self.gate.speaking:
                    continue
                now = time.monotonic()
                if self._user_in_flight and now - self._user_started_at > false_vad_s:
                    self._user_in_flight = False
                    self._idle_since = now
                if self._user_in_flight or self._idle_since is None:
                    continue
                threshold = SILENCE_AFTER_S
                if self._reading_code():
                    threshold *= OTP_SILENCE_FACTOR
                elif self.gate.turn == 0:
                    # Before the caller has said anything at all they are still
                    # taking in a greeting that runs about twenty seconds, and
                    # 15s of quiet after it is not an unresponsive caller. One
                    # live call opened with the greeting and then "Are you still
                    # there?" 15.2s later, before the caller had spoken once.
                    threshold *= OTP_SILENCE_FACTOR
                if now - self._idle_since < threshold:
                    continue
                await self._on_caller_idle()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[{self.label}] idle watchdog failed: {e}")

    def _next_nudge(self) -> tuple[str, str] | None:
        """The check-in to play, or None to stay quiet. `_nudge_i` never resets, so
        each firing gets fresh wording even though `_silence_stage` goes back to 0
        every time a user transcript lands."""
        if self._silence_stage >= 2:
            return SILENCE_GOODBYE, "silence_bye"
        # During the code-reading pause the server says nothing at all. Skipping
        # straight to the goodbye stage keeps genuine dead air terminating without
        # putting "Are you still there?" over someone squinting at their phone.
        if self._reading_code():
            return SILENCE_GOODBYE, "silence_bye"
        if self._nudges_used >= MAX_SILENCE_NUDGES:
            return SILENCE_GOODBYE, "silence_bye"
        pool = SILENCE_NUDGES if self._silence_stage == 0 else SILENCE_NUDGES_2
        kind = "silence" if self._silence_stage == 0 else "silence2"
        return pool[self._nudge_i % len(pool)], kind

    async def _on_caller_idle(self):
        """One stage of the caller-idle sequence. Resets the clock on success."""
        nudge = self._next_nudge()
        if nudge is None:
            self._idle_since = time.monotonic()
            return
        phrase, kind = nudge

        spoke = await self.gate.say(
            phrase, kind=kind, priority=Priority.CHECKIN, behavior="default",
        )
        if not spoke:
            self._idle_since = time.monotonic()
            return
        self._silence_stage += 1
        self._nudge_i += 1
        if kind != "silence_bye":
            self._nudges_used += 1
        self._idle_since = None   # AgentAudioDone will restart the clock
        if kind == "silence_bye" and not self._ending:
            self._ending = True
            self.toolbox.end_call_reason = self.toolbox.end_call_reason or "silence"
            self._mute("silence goodbye")
            logger.info(f"[{self.label}] closing after caller silence; waiting for caller to disconnect")
            # A call that ends on dead air is just as terminal as one that ends
            # on end_call, and a harness waiting on the callback would otherwise
            # sit until its own timeout.
            if self.on_end_call and not self._end_call_fired:
                self._end_call_fired = True
                try:
                    await self.on_end_call("silence")
                except Exception as e:
                    logger.warning(f"[{self.label}] on_end_call failed: {e}")

    async def _run_function(self, fn: dict):
        name = fn.get("name", "")
        line = SCRIPTED_TOOL_LINES.get(name)
        # "Transferring you to a live agent now" is a promise, not a filler, and
        # the tool can refuse a transfer nobody asked for. Ask before saying it.
        if line and name == "transfer_to_agent" and not self.toolbox.transfer_allowed():
            logger.info(f"[{self.label}] withholding the transfer line; no transfer was requested")
            line = None
        if line:
            spoke = await self.gate.say(
                line, kind=f"scripted:{name}", priority=Priority.SCRIPTED,
                behavior="queue", cooldown_s=SCRIPTED_LINE_COOLDOWN_S,
            )
            if spoke:
                self._cancel_watchdog()

        _t0 = time.time_ns()
        content = await self.toolbox.dispatch(name, fn.get("arguments", "{}"))
        self.trace.tool_call(name, fn.get("arguments", "{}"), content,
                             _t0, time.time_ns())

        locked_now = name == "verify_otp" and LINE_LOCKOUT and self.toolbox.locked
        if locked_now:
            await self.gate.say(
                LINE_LOCKOUT, kind="scripted:lockout",
                priority=Priority.SCRIPTED, behavior="queue",
            )
            self._cancel_watchdog()
            if not self._ending:
                self._ending = True
                self.toolbox.end_call_reason = (
                    self.toolbox.end_call_reason or "verification_failed"
                )
            self._mute("lockout")

        # Withhold the result for end_call too, for the same reason as lockout.
        # Sending it makes the API generate one more completion, and END_REPLY_RULE
        # ("reply with nothing at all") does not bind it: on a phone call the model
        # said "Glad I could help!", called end_call, and then delivered the entire
        # closing twice -- "Have a wonderful day... survey" and "You can disconnect
        # whenever you like" each landed a second time, 6s apart. Muting cannot
        # stop that, because the completion is driven by our response rather than
        # by caller audio. No response, no second completion.
        ending_now = name == "end_call" and self.toolbox.end_call_reason
        # send_otp is announced by the server WHILE the tool runs, so there is
        # nothing for the model to add afterwards -- and handing it the result gives
        # it a turn it then fills. Told to stay silent by both the prompt and the
        # tool's own note, it still announced the code a second time, and on one run
        # skipped ahead to "Are you calling about your Aspirin order?" before the
        # caller was verified. Withholding the result removes the turn itself. The
        # model does not need it: the next thing it sees is the caller's digits, and
        # verify_otp is driven by those.
        otp_sent = name == "send_otp" and not self.toolbox.verified
        if locked_now:
            logger.info(f"[{self.label}] lockout: withholding tool result, closing")
        elif otp_sent:
            logger.info(
                f"[{self.label}] send_otp: withholding tool result; the server has "
                f"already told the caller the code is coming"
            )
        elif ending_now:
            logger.info(
                f"[{self.label}] end_call: withholding tool result so the model gets "
                f"no turn to repeat the closing ({self.toolbox.end_call_reason})"
            )
        else:
            await self._send_json(
                {
                    "type": "FunctionCallResponse",
                    "id": fn.get("id"),
                    "name": fn.get("name"),
                    "content": content,
                }
            )
        if fn.get("name") == "check_price":
            try:
                res = json.loads(content)
                req, disp = res.get("requested_day_supply"), res.get("dispensed_day_supply")
                if req and disp and req != disp:
                    self._owed_supply = (int(req), int(disp))
                    self._supply_said = False
                    self._supply_parts = set()
            except (ValueError, TypeError):
                pass
            if (PRICE_DISCLAIMER and not self._disclaimer_said) or self._owed_supply:
                self._tasks.append(asyncio.create_task(self._ensure_price_facts()))

        if self.toolbox.end_call_reason and not self._ending:
            self._ending = True
            logger.info(
                f"[{self.label}] end_call completed; leaving transport open for caller "
                f"to disconnect ({self.toolbox.end_call_reason})"
            )
        if self._ending:
            # The transport stays open so the caller hangs up in their own time,
            # but the agent is finished. Without this the socket keeps receiving
            # audio, the API keeps detecting turns, and the model keeps answering
            # -- and after the emergency line the only permitted answer IS the
            # emergency line, which is how one call said it 353 times.
            reason = self.toolbox.end_call_reason
            # Say the close if the model did not. Withholding the tool result
            # stopped the closing being delivered twice, but it also removed the
            # model's last turn -- so a call where it went straight to end_call
            # ended on "How else can I help you?" and silence. The server owns
            # the line now: exactly once, whoever said it.
            if reason == "finished" and not self._closing_said:
                # Do NOT mark it said here. Sending an injection is not the same
                # as it being heard -- the greeting taught us that, and this line
                # repeated the mistake: the closing went out at 03:52:20.477 and
                # _mute() ran in the same millisecond, cutting the audio the API
                # needed to synthesise it. Injected, logged, never spoken.
                # `_closing_said` is set by the ConversationText handler when the
                # marker actually comes back.
                await self.gate.say(
                    LINE_CLOSING, kind="closing",
                    priority=Priority.SCRIPTED, behavior="queue",
                )
                self._tasks.append(asyncio.create_task(
                    self._mute_after_closing(f"end_call:{reason}")))
            else:
                # The emergency script is allowed exactly one follow-up: "Did you
                # hear me? Please call nine one one right away." if the caller
                # keeps talking. That needs one more turn, so the mute waits.
                self._arm_mute(f"end_call:{reason}",
                               after_turns=1 if reason == "emergency" else 0)
            if self.on_end_call and not self._end_call_fired:
                self._end_call_fired = True
                try:
                    await self.on_end_call(reason or "finished")
                except Exception as e:  # a harness callback must never kill a call
                    logger.warning(f"[{self.label}] on_end_call failed: {e}")

        if self.on_tool:
            try:
                await self.on_tool()
            except Exception as e:  # a dead UI socket must never break the call
                logger.warning(f"[{self.label}] on_tool failed: {e}")
