"""The agent and the call controller. Path B's answer to `server/va.py`.

`PharmacyAgent` publishes the nine tools on the exact schemas Path A publishes
(`tools.DEFINITIONS`, fed through `function_tool(raw_schema=...)`), and delegates
every one of them to `PharmacyCall.run_tool`, which is a port of va.py's
`_run_function`.

`PharmacyCall` owns everything that is not the model's job: the scripted lines,
the dead-air watchdog, the caller-idle sequence, the price-fact enforcement, the
emergency backstop, and the turn-admission bounds. Each of those exists because
of a specific live call on Path A, and the reasons are recorded next to the code
rather than in a commit message.

WHAT THE TRANSPORT CHANGES

Four Path A mechanisms have a LiveKit equivalent that is simply better, and in
each case the port is smaller than the original rather than larger:

  withholding a FunctionCallResponse   ->  raise StopResponse
  confirming a line was heard          ->  await handle.wait_for_playout()
  guaranteeing a line is not flushed   ->  say(..., allow_interruptions=False)
  muting caller audio into the socket  ->  session.input.set_audio_enabled(False)

The first three replace roughly 120 lines of retry-and-confirm scaffolding in
va.py (`_ensure_greeting`, `_ask_number_check`, `_mute_after_closing`), all of
which existed to work around injections that could be silently discarded. Where a
guard is gone rather than ported, the docstring says which one and why -- a
missing guard should be a decision, not an omission.

What does NOT change is every threshold, phrase and marker list: those live in
config.py, ported verbatim, because they are the parity surface.
"""

import asyncio
import json
import time

from livekit.agents import Agent, AgentSession, StopResponse, function_tool
from loguru import logger

from . import prompt, spoken, tools
from .config import (
    DISCLAIMER_ESTIMATE_MARKERS,
    DISCLAIMER_GRACE_S,
    DISCLAIMER_MAX_WAIT_S,
    DISCLAIMER_VARIANCE_MARKERS,
    DISCLAIMER_WHOLE_MARKERS,
    EMERGENCY_DENIALS,
    EMERGENCY_END_S,
    EMERGENCY_EVIDENCE,
    EMERGENCY_MARKERS,
    GOODBYE_MARKERS,
    GOODBYE_MAX_CHARS,
    HOLD_AFTER_S,
    HOLD_PHRASE_2,
    HOLD_PHRASES,
    LINE_CLOSING,
    LINE_CLOSING_MARKER,
    LINE_LOCKOUT,
    MAX_LINE_REPEATS,
    MAX_SILENCE_NUDGES,
    MAX_TURNS,
    OFFER_AGENT_MARKERS,
    OTP_SILENCE_FACTOR,
    PRICE_DISCLAIMER,
    SCRIPTED_LINE_COOLDOWN_S,
    SCRIPTED_TOOL_LINES,
    SILENCE_AFTER_S,
    SILENCE_GOODBYE,
    SILENCE_NUDGES,
    SILENCE_NUDGES_2,
    SPLIT_TURN_GRACE_S,
    STALL_LIMIT,
    STILL_WORKING_MARKERS,
    SUPPLY_CONVERSION_TEMPLATE,
)
from .metrics import CallMetrics
from .speech import Priority, SpeechGate

# The price facts are enforced, not merely attempted: on Path A a caller talking
# over the estimate caveat had it discarded by the API and nothing noticed. The
# discard cannot happen here -- an interrupted handle says so -- but the model
# skipping the disclosure entirely still can, which is what these cover.
PRICE_FACTS_ATTEMPTS = 3
PRICE_FACTS_CONFIRM_S = 8.0     # wait for the words to come back before retrying
PRICE_FACTS_GAP_WAIT_S = 12.0   # hunt for silence before a retry

# Let the closing's audio finish before muting the input.
CLOSING_DRAIN_S = 1.0


def _tool(name: str):
    """`@function_tool` on the schema Path A publishes for `name`.

    The whole reason for going through `raw_schema` rather than LiveKit's usual
    reflect-over-the-signature decorator: the tool name, description and
    parameter docs are prompt text the model reads on every turn, so they have to
    be the same bytes on both paths or the A/B is measuring the tool contract
    instead of the orchestration. `tools.definition` raises on an unknown name,
    at import time, so a typo here cannot register a tool the model can see and
    nothing can answer.

    Keeping the decorator here rather than in tools.py is deliberate too: it is
    the only LiveKit dependency in the tool layer, and tools.py stays importable
    without livekit installed so the offline harnesses can use ToolBox directly.
    """
    return function_tool(raw_schema=tools.definition(name))


class PharmacyAgent(Agent):
    """The nine tools, on Path A's schemas.

    Every method is a two-liner on purpose. The tool CONTRACT is
    `tools.DEFINITIONS` and the tool BEHAVIOUR is `tools.ToolBox`; this class is
    only the bridge between them and LiveKit's dispatcher, so there is nothing
    here to drift.

    `raw_arguments` is how LiveKit hands a raw-schema tool its arguments, and it
    arrives as the same dict Path A gets from `json.loads(fn["arguments"])`. It
    must stay annotated: the dispatcher resolves type hints by parameter name and
    raises on a bare parameter.
    """

    def __init__(self, call: "PharmacyCall"):
        super().__init__(instructions=prompt.build())
        self._call = call

    @_tool("send_otp")
    async def send_otp(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("send_otp", raw_arguments)

    @_tool("verify_otp")
    async def verify_otp(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("verify_otp", raw_arguments)

    @_tool("end_call")
    async def end_call(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("end_call", raw_arguments)

    @_tool("transfer_to_agent")
    async def transfer_to_agent(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("transfer_to_agent", raw_arguments)

    @_tool("lookup_medication")
    async def lookup_medication(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("lookup_medication", raw_arguments)

    @_tool("search_knowledge_base")
    async def search_knowledge_base(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("search_knowledge_base", raw_arguments)

    @_tool("get_account_info")
    async def get_account_info(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("get_account_info", raw_arguments)

    @_tool("fill_prescription")
    async def fill_prescription(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("fill_prescription", raw_arguments)

    @_tool("check_price")
    async def check_price(self, raw_arguments: dict[str, object]) -> str:
        return await self._call.run_tool("check_price", raw_arguments)


class PharmacyCall:
    """One call's worth of state, watchdogs and guards."""

    def __init__(self, label: str, caller_number: str | None):
        self.label = label
        self.toolbox = tools.ToolBox(caller_number=caller_number, session_label=label)
        self.metrics = CallMetrics(label)
        # every server-spoken line goes through this; it owns "is the agent
        # talking", "did the model answer", and which turn we are on
        self.gate = SpeechGate(
            label, self.metrics, SPLIT_TURN_GRACE_S,
            max_line_repeats=MAX_LINE_REPEATS, stall_limit=STALL_LIMIT,
        )
        self.session: AgentSession | None = None

        self._tasks: set[asyncio.Task] = set()
        self._watchdog: asyncio.Task | None = None
        self._hold_i = 0                # rotates HOLD_PHRASES across turns
        self._idle_since: float | None = None   # when the agent last stopped talking
        self._silence_stage = 0         # 0 nudge, 1 second nudge, 2 hangup
        self._nudge_i = 0               # rotates the check-in wording; never resets
        self._nudges_used = 0           # total check-ins this call, capped
        self._user_in_flight = False    # user_state "speaking" with no transcript yet
        self._user_started_at = 0.0
        self._disclaimer_said = False
        self._disclaimer_parts: set[str] = set()   # price disclaimer, once per call
        self._owed_supply: tuple[int, int] | None = None   # (requested, dispensed)
        self._supply_said = False
        self._supply_parts: set[str] = set()
        self._ending = False
        self._muted = False             # stop giving the model turns
        self._closing_said = False
        self._emergency_armed = False
        self._emergency_evidence = False   # caller gave a reason at some point
        self._mute_allowance: int | None = None   # replies still owed before muting
        self._mute_reason = ""
        self._closed = False

    # --- wiring ---------------------------------------------------------------

    def attach(self, session: AgentSession) -> None:
        """Subscribe to the session's events. Handlers are sync; work is spawned."""
        self.session = session
        self.gate.attach(session)

        session.on("agent_state_changed", self._on_agent_state)
        session.on("user_state_changed", self._on_user_state)
        session.on("user_input_transcribed", self._on_user_transcript)
        session.on("conversation_item_added", self._on_conversation_item)
        session.on("metrics_collected", self._on_metrics)
        session.on("error", self._on_error)

        self._spawn(self._caller_idle_watchdog())

    def _spawn(self, coro) -> asyncio.Task:
        """Fire and forget, but keep a reference so the task is not garbage
        collected mid-flight and is cancellable at shutdown."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def open(self) -> None:
        """Deliver the opening.

        Two lines, spoken in order, and `say()` queues the second behind the
        first so the caller hears one continuous opening -- the same effect Path A
        gets by queueing the verification question at session open.

        The compliance block is NOT interruptible. It carries the 911 instruction
        and the recording disclosure, and on Path A a caller saying "hello" during
        slow synthesis caused barge-in to flush it, after which the model
        improvised a welcome of its own: one observed live was "Hello! Welcome to
        Amazon Pharmacy. I'm here to help you with medications, pricing, orders,
        and prescriptions." That is not a cosmetic loss, so va.py re-injects the
        greeting up to twice and watches for its words to come back. Here the line
        simply declines to be interrupted, and `_ensure_greeting` has no reason to
        exist.

        The verification question IS interruptible, and deliberately so: a caller
        who answers "yes" halfway through it should be able to. Path A had to
        retry this one up to four times because an injection lost to barge-in was
        indistinguishable from one still queued behind a 15-second greeting. The
        handle tells the two apart, so an interruption is logged and left alone --
        the caller answering the question is not the question failing.
        """
        await self.gate.say(
            prompt.GREETING, kind="greeting",
            priority=Priority.SCRIPTED, interruptible=False,
        )
        handle = await self.gate.say(
            prompt.LINE_NUMBER_CHECK, kind="number_check",
            priority=Priority.SCRIPTED,
        )
        if handle is None:
            logger.warning(f"[{self.label}] verification question was not spoken")
            return
        await handle.wait_for_playout()
        if handle.interrupted:
            logger.info(
                f"[{self.label}] verification question interrupted; the caller is "
                "answering, not missing it"
            )

    async def finish(self) -> None:
        self._closed = True
        self.gate.close()
        self.metrics.duplicate_lines = list(self.gate.duplicates)
        self.metrics.narrated_lines = list(self.gate.narrations)
        self.metrics.repeated_lines = self.gate.repeated_lines
        self._cancel_watchdog()
        for task in list(self._tasks):
            task.cancel()
        self.metrics.finish()

    # --- event handlers -------------------------------------------------------

    def _on_agent_state(self, ev) -> None:
        self.gate.on_agent_state(ev.new_state)
        if ev.new_state == "speaking":
            self.metrics.on_agent_audio()
        elif ev.old_state == "speaking" and not self._closed:
            # the AgentAudioDone analogue: the caller-idle clock starts when the
            # agent stops talking, not when the turn began
            self._idle_since = time.monotonic()

    def _on_user_state(self, ev) -> None:
        self.gate.on_user_state(ev.new_state)
        if ev.new_state == "speaking":
            self.metrics.on_user_speaking()
            self._user_in_flight = True
            self._user_started_at = time.monotonic()
            self._idle_since = None

    def _on_user_transcript(self, ev) -> None:
        """A final caller transcript: va.py's `ConversationText` role=user path."""
        if not ev.is_final:
            return
        content = ev.transcript or ""
        self.metrics.on_text("user", content)
        logger.info(f"[{self.label}] user: {content}")

        # Keep the caller's latest complete transcript available to verify_otp.
        # This covers a model that heard the code but sent incomplete arguments.
        self.toolbox.latest_user_text = content
        if not self._emergency_evidence and self._is_emergency(content):
            self._emergency_evidence = True
        self._user_in_flight = False
        self._idle_since = None
        self._silence_stage = 0
        self._cancel_watchdog()

        # A deferred mute counts CALLER turns, not assistant messages: the model
        # splits one utterance across several messages, so counting those spent
        # the emergency script's owed confirmation on the second half of its own
        # opening line. This turn is already through, so the reply to it still
        # happens -- that reply IS the confirmation -- and everything after is
        # dropped.
        if self._mute_allowance is not None:
            self._mute_allowance -= 1
            if self._mute_allowance <= 0:
                self._mute(self._mute_reason)

        if self.gate.turn >= MAX_TURNS:
            self._mute("turn ceiling")
        elif self.gate.begin_user_turn():
            self._watchdog = self._spawn(self._hold_phrase_watchdog(self.gate.turn))
        else:
            logger.info(f"[{self.label}] split user turn; not re-arming the watchdog")

    def _on_conversation_item(self, ev) -> None:
        """va.py's `ConversationText` role=assistant path.

        `say()` reports what was actually forwarded, so an interrupted server line
        comes back clipped; SpeechGate matches echoes on a prefix for that reason.
        """
        item = ev.item
        if getattr(item, "role", None) != "assistant":
            return
        content = item.text_content or ""
        if not content:
            return
        self.metrics.on_text("assistant", content)
        logger.info(f"[{self.label}] assistant: {content}")
        low = content.lower()

        if LINE_CLOSING_MARKER in low:
            self._closing_said = True
        # Record an offer of a live person, so a caller answering "yes" to it
        # still gets transferred. tools.transfer_to_agent refuses a transfer
        # nobody asked for.
        if any(m in low for m in OFFER_AGENT_MARKERS):
            self.toolbox.offered_agent = True
        # The agent has just told this caller to hang up and dial 911. Whether
        # the model also calls end_call is the server's problem, not the
        # prompt's -- see the note in config.py.
        if not self._emergency_armed and any(m in low for m in EMERGENCY_MARKERS):
            self._emergency_armed = True
            self._spawn(self._emergency_backstop())

        self.gate.on_agent_text(content)
        if self.gate.stalled:
            self._mute("stalled")

        if not self._disclaimer_said:
            # Both halves, tracked across messages: the model routinely splits
            # "that's an estimate" and "your price may differ" into separate
            # messages, so neither one alone proves anything.
            if any(m in low for m in DISCLAIMER_WHOLE_MARKERS):
                self._disclaimer_parts.update({"estimate", "variance"})
            if any(m in low for m in DISCLAIMER_ESTIMATE_MARKERS):
                self._disclaimer_parts.add("estimate")
            if any(m in low for m in DISCLAIMER_VARIANCE_MARKERS):
                self._disclaimer_parts.add("variance")
            if {"estimate", "variance"} <= self._disclaimer_parts:
                self._disclaimer_said = True

        if self._owed_supply and not self._supply_said:
            # Both numbers, not just the dispensed one. An earlier version checked
            # only for "eighty four", so "dispensed as an eighty four day supply"
            # counted as disclosure -- and a caller who asked for NINETY days
            # never heard what became of their request, while the server concluded
            # it had been told. The fact being disclosed is the relationship
            # between the two numbers, so neither half alone satisfies it.
            for part, n in (("req", self._owed_supply[0]),
                            ("disp", self._owed_supply[1])):
                if spoken.integer(n) in low or str(n) in content:
                    self._supply_parts.add(part)
            if {"req", "disp"} <= self._supply_parts:
                self._supply_said = True

    def _on_metrics(self, ev) -> None:
        self.metrics.on_metrics(ev.metrics)

    def _on_error(self, ev) -> None:
        message = str(getattr(ev, "error", ev))
        logger.error(f"[{self.label}] session error: {message}")
        self.metrics.on_error(message)

    # --- keypad ---------------------------------------------------------------

    async def keypad(self, digits: str) -> None:
        """Hand the model typed digits as if they had been spoken.

        Muting has to cover this too, not just audio: a keypad press after the
        call has ended would otherwise hand the model a turn by the back door.
        """
        if self._muted or self._closed or self.session is None:
            logger.info(f"[{self.label}] dropped keypad entry after mute: {digits!r}")
            return
        logger.info(f"[{self.label}] keypad entry: {digits}")
        self.toolbox.latest_user_text = digits
        self.session.generate_reply(user_input=f"[keypad] {digits}")

    # --- muting ---------------------------------------------------------------

    def _mute(self, reason: str) -> None:
        """Stop giving the model turns. Irreversible for the rest of the call.

        The room stays connected: end_call does not disconnect the caller, and a
        caller who wants to say one more thing should not hear the line drop. So
        the audio INPUT is disabled rather than the session being closed -- the
        model is never handed another turn, and the caller hangs up in their own
        time.

        Path A had to drop audio at the socket for the same reason, because the
        Voice Agent API answers every end of turn it detects and the server has
        no say in that. An AgentSession behaves the same way, so the guard is the
        same guard.
        """
        if self._muted:
            return
        self._muted = True
        self._mute_allowance = None
        self.metrics.muted_reason = reason
        self._cancel_watchdog()
        if self.session is not None:
            try:
                self.session.input.set_audio_enabled(False)
            except Exception as e:  # a dying session must not break the ending
                logger.warning(f"[{self.label}] could not disable audio input: {e}")
        logger.info(
            f"[{self.label}] muted after {self.gate.turn} turns ({reason}); "
            "the room stays open for the caller to disconnect"
        )

    def _arm_mute(self, reason: str, after_turns: int = 0) -> None:
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

    async def _mute_after_closing(self, reason: str, handle) -> None:
        """Mute once the closing has actually been spoken.

        Path A could only wait for the closing's words to come back on a timer,
        having learned that muting in the same breath as injecting it cut the
        audio the API needed to synthesise it -- injected, logged, never spoken.
        The handle removes the guesswork.
        """
        if handle is not None:
            await handle.wait_for_playout()
            if handle.interrupted:
                logger.info(f"[{self.label}] closing was interrupted by the caller")
        await asyncio.sleep(CLOSING_DRAIN_S)
        self._mute(reason)

    # --- tool dispatch --------------------------------------------------------

    async def run_tool(self, name: str, args: dict) -> str:
        """va.py's `_run_function`, minus the websocket.

        The one structural difference: Path A can decline to answer a
        FunctionCallRequest at all, which is how it stops the model getting a
        turn it would fill with a duplicate line. Here the equivalent is raising
        `StopResponse` after the side effects have run -- so the order below is
        Path A's order, with the raise where the "don't send a response" branch
        used to be.
        """
        line = SCRIPTED_TOOL_LINES.get(name)
        # "Transferring you to a live agent now" is a promise, not a filler, and
        # the tool can refuse a transfer nobody asked for. Ask before saying it:
        # with the guard refusing the transfer, a caller once still heard
        # "Transferring you to a live agent now" and was then told their
        # prescription would be shipped.
        if line and name == "transfer_to_agent" and not self.toolbox.transfer_allowed():
            logger.info(
                f"[{self.label}] withholding the transfer line; no transfer was requested"
            )
            line = None
        if line:
            if await self.gate.say(
                line, kind=f"scripted:{name}", priority=Priority.SCRIPTED,
                cooldown_s=SCRIPTED_LINE_COOLDOWN_S,
            ):
                self._cancel_watchdog()

        content = await self.toolbox.dispatch(name, json.dumps(args))

        locked_now = name == "verify_otp" and LINE_LOCKOUT and self.toolbox.locked
        if locked_now:
            await self.gate.say(
                LINE_LOCKOUT, kind="scripted:lockout", priority=Priority.SCRIPTED,
            )
            self._cancel_watchdog()
            if not self._ending:
                self._ending = True
                self.toolbox.end_call_reason = (
                    self.toolbox.end_call_reason or "verification_failed"
                )
            self._mute("lockout")

        # Withhold the reply for end_call. Letting the model speak after it makes
        # one more completion, and END_REPLY_RULE ("reply with nothing at all")
        # does not bind it: on a Path A phone call the model said "Glad I could
        # help!", called end_call, and then delivered the entire closing twice --
        # "Have a wonderful day... survey" and "You can disconnect whenever you
        # like" each landed a second time, 6s apart. Muting cannot stop that,
        # because the completion is driven by the tool result rather than by
        # caller audio. No reply, no second completion.
        ending_now = name == "end_call" and self.toolbox.end_call_reason
        # send_otp is announced by the server WHILE the tool runs, so there is
        # nothing for the model to add afterwards -- and giving it a turn is
        # giving it something to fill. Told to stay silent by both the prompt and
        # the tool's own note, it still announced the code a second time, and on
        # one run skipped ahead to "Are you calling about your Aspirin order?"
        # before the caller was verified. The model does not need the result: the
        # next thing it sees is the caller's digits, and verify_otp is driven by
        # those.
        otp_sent = name == "send_otp" and not self.toolbox.verified

        if name == "check_price":
            try:
                res = json.loads(content)
                req = res.get("requested_day_supply")
                disp = res.get("dispensed_day_supply")
                if req and disp and req != disp:
                    self._owed_supply = (int(req), int(disp))
                    self._supply_said = False
                    self._supply_parts = set()
            except (ValueError, TypeError):
                pass
            if (PRICE_DISCLAIMER and not self._disclaimer_said) or self._owed_supply:
                self._spawn(self._ensure_price_facts())

        if self.toolbox.end_call_reason and not self._ending:
            self._ending = True
            logger.info(
                f"[{self.label}] end_call completed; leaving the room open for the "
                f"caller to disconnect ({self.toolbox.end_call_reason})"
            )
        if self._ending:
            await self._begin_ending()

        if locked_now:
            logger.info(f"[{self.label}] lockout: no reply for the model, closing")
            raise StopResponse
        if otp_sent:
            logger.info(
                f"[{self.label}] send_otp: no reply for the model; the server has "
                "already told the caller the code is coming"
            )
            raise StopResponse
        if ending_now:
            logger.info(
                f"[{self.label}] end_call: no reply for the model so it cannot repeat "
                f"the closing ({self.toolbox.end_call_reason})"
            )
            raise StopResponse
        return content

    async def _begin_ending(self) -> None:
        """The agent is finished, whoever decided so.

        The room stays open so the caller hangs up in their own time, but without
        this the session keeps detecting turns and the model keeps answering --
        and after the emergency line the only permitted answer IS the emergency
        line, which is how one Path A call said it 353 times.
        """
        reason = self.toolbox.end_call_reason
        # Say the close if the model did not. Withholding the tool result stopped
        # the closing being delivered twice, but it also removed the model's last
        # turn -- so a call where it went straight to end_call ended on "How else
        # can I help you?" and silence. The server owns the line now: exactly
        # once, whoever said it.
        if reason == "finished" and not self._closing_said:
            handle = await self.gate.say(
                LINE_CLOSING, kind="closing",
                priority=Priority.SCRIPTED, interruptible=False,
            )
            self._spawn(self._mute_after_closing(f"end_call:{reason}", handle))
        else:
            # The emergency script is allowed exactly one follow-up: "Did you
            # hear me? Please call nine one one right away." if the caller keeps
            # talking. That needs one more turn, so the mute waits.
            self._arm_mute(f"end_call:{reason}",
                           after_turns=1 if reason == "emergency" else 0)

    # --- price facts ----------------------------------------------------------

    async def _ensure_price_facts(self) -> None:
        """Speak the day-supply conversion and the estimate disclaimer if the
        model skipped them."""
        try:
            await asyncio.sleep(DISCLAIMER_GRACE_S)
            # Decide only once the model has actually stopped talking. Deciding on
            # a fixed 4s timer meant queueing the disclaimer while the model was
            # still mid-answer: on one Path A call we queued at 21:25:10, the
            # model said its own version at 21:25:31, and ours played at 21:25:38
            # -- the caller heard the estimate caveat twice, 7s apart. The price
            # answer routinely runs past the grace period, so wait it out.
            waited = 0.0
            while (self.gate.speaking and not self._closed
                   and waited < DISCLAIMER_MAX_WAIT_S):
                await asyncio.sleep(0.25)
                waited += 0.25
            if self._closed:
                return
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
                    # Retry only into a gap. Sending again immediately loses it
                    # the same way it was lost the first time.
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
                # The `_said` flags are set by the conversation-item handler when
                # the words actually come back, never here. Speaking a line is not
                # the same as it being heard, and treating the two as equal is how
                # a graded disclosure went silently missing on Path A.
                await self.gate.say(
                    " ".join(missing), kind="price_facts", priority=Priority.ENFORCE,
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
            logger.warning(f"[{self.label}] price-fact enforcement failed: {e}")

    # --- dead air -------------------------------------------------------------

    def _cancel_watchdog(self) -> None:
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

    async def _hold_phrase_watchdog(self, turn: int) -> None:
        """Cover dead air after the caller stops, if nothing else already did."""
        try:
            await asyncio.sleep(HOLD_AFTER_S)
            if not self.toolbox.verified:
                return
            if self._ending or self.toolbox.end_call_reason:
                return
            # ...and the same check one beat earlier, from the caller's words. The
            # state check above only sees an end_call that has already happened; on
            # a sign-off the model is still composing the closing at 2.8s, so
            # nothing has been set yet and the filler goes out anyway.
            if self._is_goodbye(self.toolbox.latest_user_text):
                logger.info(f"[{self.label}] speech[hold] skipped: caller was signing off")
                return
            phrase = HOLD_PHRASES[self._hold_i % len(HOLD_PHRASES)]
            if not await self.gate.say(
                phrase, kind="hold", priority=Priority.FILLER, turn=turn,
            ):
                return
            self._hold_i += 1
            await asyncio.sleep(5)
            await self.gate.say(
                HOLD_PHRASE_2, kind="hold2", priority=Priority.FILLER, turn=turn,
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:  # never crash the task
            logger.warning(f"[{self.label}] hold watchdog failed: {e}")

    # --- emergency ------------------------------------------------------------

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
        spoke, or fired as the call was already ending.
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
        telling someone to dial 911. gpt-4o-mini did exactly that on 3 of 3 runs.

        This decides nothing the prompt would not have. It waits for the model's
        own end_call, and only if none arrives does it run the same ending the
        tool would have -- including the one owed follow-up turn, so "Did you hear
        me? Please call nine one one right away." can still be delivered.
        """
        try:
            await asyncio.sleep(EMERGENCY_END_S)
            if self._closed or self._ending or self.toolbox.end_call_reason:
                return
            # The line was spoken, but nobody asked for it. Ending the call here is
            # how a spurious trigger turns into a hang-up on a caller who is fine,
            # so the backstop stands down and the call carries on. Logged loudly:
            # the line should not have been said at all, and that is worth counting.
            if not self._backstop_allowed():
                logger.warning(
                    f"[{self.label}] SPURIOUS emergency line -- fired into silence or "
                    f"a sign-off; NOT ending the call (last caller turn: "
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
        except asyncio.CancelledError:
            pass

    # --- caller silence -------------------------------------------------------

    def _reading_code(self) -> bool:
        """Between send_otp and the first verify_otp: the caller is on their phone."""
        return (
            bool(self.toolbox.otp)
            and not self.toolbox.verified
            and self.toolbox.verify_attempts == 0
        )

    async def _caller_idle_watchdog(self) -> None:
        """Nudge, then stop, if the caller stays quiet after we finish speaking."""
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
                    # taking in a greeting that runs about twenty seconds, and 15s
                    # of quiet after it is not an unresponsive caller. One live
                    # call opened with the greeting and then "Are you still
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

    async def _on_caller_idle(self) -> None:
        """One stage of the caller-idle sequence. Resets the clock on success."""
        nudge = self._next_nudge()
        if nudge is None:
            self._idle_since = time.monotonic()
            return
        phrase, kind = nudge

        spoke = await self.gate.say(phrase, kind=kind, priority=Priority.CHECKIN)
        if not spoke:
            self._idle_since = time.monotonic()
            return
        self._silence_stage += 1
        self._nudge_i += 1
        if kind != "silence_bye":
            self._nudges_used += 1
        self._idle_since = None   # the agent leaving "speaking" restarts the clock
        if kind == "silence_bye" and not self._ending:
            self._ending = True
            self.toolbox.end_call_reason = self.toolbox.end_call_reason or "silence"
            self._mute("silence goodbye")
            logger.info(
                f"[{self.label}] closing after caller silence; waiting for the caller "
                "to disconnect"
            )
