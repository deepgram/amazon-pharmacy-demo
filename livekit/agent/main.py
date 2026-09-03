"""Amazon Pharmacy demo agent -- Path B: LiveKit orchestration.

Same agent as the Voice Agent API demo (`../voice-agent-api`): same
prompt, same three knowledge bases, same nine tools on the same schemas, same
scripted lines and guardrails. Same three models, too -- which is the point of
the pairing. Only the orchestration differs.

    Path A  Twilio media stream -> Deepgram Voice Agent API
            (one websocket does listen, think, speak and turn-taking)

    Path B  Twilio SIP trunk -> LiveKit Cloud SIP -> LiveKit room -> this worker
            Deepgram nova-3-medical_pharma -> gpt-4o-mini -> Deepgram Flux TTS
            (three services, LiveKit doing the orchestration)

Run:
  uv run python -m agent.main console   # local terminal test, no LiveKit creds
  uv run python -m agent.main dev       # connect to LiveKit Cloud
  uv run python -m agent.main start     # production mode

The worker connects outbound to LiveKit Cloud, so it needs no tunnel and no
inbound port. Both demos can run at the same time.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from livekit import rtc  # noqa: E402
from livekit.agents import (  # noqa: E402
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    stt as stt_base,
)
from livekit.plugins import deepgram  # noqa: E402
from loguru import logger  # noqa: E402

from . import catalog  # noqa: E402
from .config import (  # noqa: E402
    AGENT_NAME,
    EAGER_EOT_THRESHOLD,
    ENDPOINTING_MAX_DELAY_S,
    ENDPOINTING_MIN_DELAY_S,
    EOT_THRESHOLD,
    EOT_TIMEOUT_MS,
    LISTEN_ENDPOINTING_MS,
    LISTEN_KEYTERMS,
    LISTEN_MODEL,
    MIP_OPT_OUT,
    SPEAK_IS_FLUX,
    TAGS,
    THINK_MODEL,
    THINK_PROVIDER,
    THINK_TEMPERATURE,
    VOICE_MODEL,
)
from .session import PharmacyAgent, PharmacyCall  # noqa: E402

# Phrases the ASR tends to mangle over a telephony channel. Kept in sync with
# `pharmacy-voice-agent/server/va.py` EXTRA_KEYTERMS -- change both or neither.
EXTRA_KEYTERMS = [
    "ninety days", "thirty days", "eighty four days", "day supply",
    "ninety day supply", "thirty day supply", "change my mind",
    "twelve point five", "milligrams", "micrograms", "verification code",
    "refill", "prescription", "insurance", "coupon",
]

LISTEN_FALLBACK_MODEL = "flux-general-en"


def _keyterms() -> list[str]:
    """The keyterm list, or empty when policy says not to send one.

    `auto` means on, for both model families. It was Flux-only at one point,
    because on nova-3-medical keyterms measured as a no-op and once as actively
    harmful -- a misheard Olmesartan became "omeprazole", a real catalog drug, so
    the error went silent where without them it stayed nonsense the fuzzy matcher
    caught. What changed the policy is that "RxPass" and "PillPack" have no
    fallback at all: the ASR splits them into two ordinary words and the
    knowledge-base lookup lands on the wrong entry.
    """
    if LISTEN_KEYTERMS == "off":
        return []
    return catalog.keyterms() + EXTRA_KEYTERMS


def _build_stt(model: str, *, keyterms: bool) -> stt_base.STT:
    """One rung of the listen ladder.

    The Flux/Nova split is a different plugin class here, not a `version` field:
    Flux is the /v2/listen websocket (`STTv2`, native end-of-turn), Nova is
    /v1/listen (`STT`, endpointing-driven). They configure differently and only
    one of the two accepts each set of keys, which is the same shape as Path A's
    `is_flux` branch in build_settings.
    """
    terms = _keyterms() if keyterms else []
    if model.startswith("flux"):
        opts: dict = {
            "eot_threshold": EOT_THRESHOLD,
            "eot_timeout_ms": EOT_TIMEOUT_MS,
        }
        if EAGER_EOT_THRESHOLD:
            # Starts the LLM on a medium-confidence turn end, ~150-250ms earlier.
            # Must be <= eot_threshold or the eager value never fires.
            opts["eager_eot_threshold"] = min(EAGER_EOT_THRESHOLD, EOT_THRESHOLD)
        if terms:
            opts["keyterm"] = terms
        return deepgram.STTv2(model=model, mip_opt_out=MIP_OPT_OUT, tags=TAGS, **opts)

    opts = {}
    if terms:
        opts["keyterm"] = terms
    return deepgram.STT(
        model=model,
        # Nova only, and not optional: without it 8 of 12 utterances split into
        # multiple turns and the agent answers half a sentence. 500ms is a
        # measured floor, not a guess -- see the note in config.py.
        endpointing_ms=LISTEN_ENDPOINTING_MS,
        mip_opt_out=MIP_OPT_OUT,
        tags=TAGS,
        **opts,
    )


def _listen() -> stt_base.STT:
    """The listen ladder, as a FallbackAdapter.

    The Voice Agent API path degrades one rung at a time when Settings is
    rejected: keyterms first, because they are what makes "RxPass" survive the
    channel; then the same model without them, because losing keyterms is a far
    smaller loss than dropping to flux-general-en, which mishears "Zepbound" as
    "sells that bound"; then the fallback model as the last resort.

    `stt.FallbackAdapter` is the same ladder and strictly stronger: that path can
    only recover at session open, because a rejection it never saw is a model
    that already started. This one also covers a model that fails mid-call.

    The cost is a caveat the other path does not have. Falling through to Flux changes
    turn-taking underneath a live call -- Flux decides end-of-turn itself, Nova
    leaves it to endpointing -- so a call that degrades gets a different feel as
    well as a different transcript. `GET`-style provenance matters more here than
    on Path A for that reason: the summary in ops/metrics records the model that
    was CONFIGURED, so on a suspicious latency run check the Deepgram request log
    for which one actually served.
    """
    rungs = [_build_stt(LISTEN_MODEL, keyterms=True)]
    if _keyterms():
        rungs.append(_build_stt(LISTEN_MODEL, keyterms=False))
    if LISTEN_MODEL != LISTEN_FALLBACK_MODEL:
        rungs.append(_build_stt(LISTEN_FALLBACK_MODEL, keyterms=True))
    if len(rungs) == 1:
        return rungs[0]
    return stt_base.FallbackAdapter(rungs)


def _think():
    """The think model.

    Imported lazily and per-provider so a missing plugin or key fails with the
    provider's own message, and so running the anthropic branch does not require
    the openai plugin or vice versa.

    Path A gets `gpt-4o-mini` from Deepgram as a managed provider and needs no
    OpenAI credential. LiveKit calls the provider directly, so this needs
    OPENAI_API_KEY in the environment. That is the one piece of configuration the
    two repos cannot share -- see the note on THINK_MODEL in config.py.
    """
    is_gpt5 = THINK_PROVIDER == "open_ai" and THINK_MODEL.startswith("gpt-5")
    if THINK_PROVIDER == "anthropic":
        from livekit.plugins import anthropic

        return anthropic.LLM(model=THINK_MODEL, temperature=THINK_TEMPERATURE)
    from livekit.plugins import openai

    if is_gpt5:
        # gpt-5 rejects any explicit temperature; reasoning above "low" is death
        # for voice latency.
        return openai.LLM(model=THINK_MODEL)
    return openai.LLM(model=THINK_MODEL, temperature=THINK_TEMPERATURE)


def _speak():
    """Flux TTS through /v2/speak, or an aura-2 voice through /v1.

    `speed` and `expressivity` have no equivalent on `TTSv2` in
    livekit-plugins-deepgram 1.7. Path A holds both at their defaults and omits
    them from Settings, so the shipped behaviour is identical; config.py says so
    where the env vars are read, because a non-default set there would silently
    do nothing.
    """
    if SPEAK_IS_FLUX:
        return deepgram.TTSv2(model=VOICE_MODEL, mip_opt_out=MIP_OPT_OUT)
    return deepgram.TTS(model=VOICE_MODEL, mip_opt_out=MIP_OPT_OUT)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # caller number from the SIP participant (absent in console/browser tests)
    caller = None
    label = "console"
    try:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=5)
        caller = participant.attributes.get("sip.phoneNumber") or None
        label = (participant.attributes.get("sip.twilio.callSid")
                 or participant.identity or "call")[-8:]
    except asyncio.TimeoutError:
        logger.info("no SIP participant within 5s; running without caller context")
    except Exception as e:  # console mode has no participant at all
        logger.info(f"no SIP participant ({e}); running without caller context")

    logger.info(f"[{label}] session starting, caller={caller}")
    call = PharmacyCall(label, caller)
    call.metrics.listen_model = LISTEN_MODEL
    call.metrics.think_model = f"{THINK_PROVIDER}/{THINK_MODEL}"
    call.metrics.voice_model = VOICE_MODEL

    session = AgentSession(
        stt=_listen(),
        llm=_think(),
        tts=_speak(),
        # "stt" delegates end-of-turn to the listen model, which is what Path A
        # does -- Nova's endpointing, or Flux's native decision. VAD is left at
        # its default (the bundled silero) because interruption detection still
        # needs it; only turn-taking comes from the STT.
        #
        # LiveKit's own endpointing delay stacks on top of the STT's. That is the
        # one turn-taking knob with no Path A counterpart, so it is the first
        # place to look when the two paths disagree on turn latency.
        turn_handling={
            "turn_detection": "stt",
            "endpointing": {
                "min_delay": ENDPOINTING_MIN_DELAY_S,
                "max_delay": ENDPOINTING_MAX_DELAY_S,
            },
        },
    )
    call.attach(session)
    ctx.add_shutdown_callback(call.finish)

    # keypad digits: accumulate, submit on 6 digits or '#', clear on '*'
    dtmf_digits: list[str] = []

    def _submit():
        digits = "".join(dtmf_digits)
        dtmf_digits.clear()
        if digits:
            asyncio.create_task(call.keypad(digits))

    def on_dtmf(dtmf: rtc.SipDTMF):
        digit = dtmf.digit
        if digit == "#":
            _submit()
        elif digit == "*":
            dtmf_digits.clear()
        elif digit:
            dtmf_digits.append(digit)
            if len(dtmf_digits) >= 6:
                _submit()

    ctx.room.on("sip_dtmf_received", on_dtmf)

    await session.start(room=ctx.room, agent=PharmacyAgent(call))
    await call.open()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))
