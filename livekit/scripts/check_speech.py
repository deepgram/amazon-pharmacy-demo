"""Exercise the speech gate and the watchdogs against a stub session.

    uv run python scripts/check_speech.py

`scripts/check_parity.py` proves this repo and Path A share the same prompt,
tools, knowledge bases and thresholds. It cannot prove they *behave* the same,
because the layer that decides who speaks and when was rebuilt on LiveKit
primitives rather than copied -- `InjectAgentMessage` became `AgentSession.say()`,
audio frames became `agent_state_changed`, and a withheld tool response became
`StopResponse`. A rebuild is exactly where a guard gets quietly lost.

So every case below is one of the live-call collisions the Path A gate was built
for, replayed through this one. Nothing here touches the network: the session is
a stub that records what the agent was made to say.

The stall-detection trip point deserves a note, because it looks off by one and
is not. `_stall_run` only advances when a turn CLOSES, and the mute is checked on
the assistant message after that, so the agent is muted on the STALL_LIMIT + 2nd
turn. Path A's gate trips on exactly the same turn -- both were driven through
the identical sequence and produced the same trajectory,
[(0,0,False), (1,0,False), (2,1,False), (3,2,False), (4,3,True)].

Exit status is 0 when every check passes, 1 when any fails.
"""

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

logger.remove()   # the drop-reason log is the thing under test, not the output

from agent import config, prompt  # noqa: E402
from agent.session import PharmacyCall  # noqa: E402
from agent.speech import Priority  # noqa: E402


class FakeHandle:
    def __init__(self):
        self.interrupted = False

    async def wait_for_playout(self):
        return None


class FakeInput:
    def __init__(self):
        self.audio_enabled = True

    def set_audio_enabled(self, value):
        self.audio_enabled = value


class FakeSession:
    """Records what the agent was made to say, and nothing else."""

    def __init__(self):
        self.said: list[tuple[str, bool]] = []      # (text, interruptible)
        self.replies: list[str] = []
        self.input = FakeInput()
        self._handlers: dict[str, list] = {}

    def on(self, event, fn):
        self._handlers.setdefault(event, []).append(fn)

    def emit(self, event, ev):
        for fn in self._handlers.get(event, []):
            fn(ev)

    def say(self, text, *, allow_interruptions=True):
        self.said.append((text, allow_interruptions))
        return FakeHandle()

    def generate_reply(self, *, user_input):
        self.replies.append(user_input)

    @property
    def texts(self) -> list[str]:
        return [t for t, _ in self.said]


def _ev(**kw):
    return types.SimpleNamespace(**kw)


def agent_state(s, old, new):
    s.emit("agent_state_changed", _ev(old_state=old, new_state=new))


def user_says(s, text):
    s.emit("user_input_transcribed", _ev(transcript=text, is_final=True))


def agent_says(s, text):
    s.emit("conversation_item_added", _ev(item=_ev(role="assistant", text_content=text)))


async def new_call(verified: bool = True) -> tuple[PharmacyCall, FakeSession]:
    call = PharmacyCall("check", "+15551230000")
    session = FakeSession()
    call.attach(session)
    if verified:
        # the hold watchdog deliberately stays quiet until the caller is verified
        call.toolbox.verified = True
        call.toolbox.otp = config.DEMO_OTP
        call.toolbox.verify_attempts = 1
    return call, session


results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(("  ok    " if ok else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


async def main() -> None:
    print("dead-air filler")

    call, s = await new_call()
    user_says(s, "how much is my Zepbound")
    await asyncio.sleep(config.HOLD_AFTER_S + 0.4)
    check("fires on genuine dead air", config.HOLD_PHRASES[0] in s.texts,
          f"said={s.texts}")

    # the 12-filler call: filler landing AFTER the answer it was covering
    call, s = await new_call()
    user_says(s, "how much is my Zepbound")
    agent_says(s, "That is fifty dollars with insurance for an eighty four day supply.")
    await asyncio.sleep(config.HOLD_AFTER_S + 0.4)
    check("dropped when the model already answered this turn",
          not any(t in config.HOLD_PHRASES for t in s.texts), f"said={s.texts}")

    call, s = await new_call()
    user_says(s, "how much is my Zepbound")
    agent_state(s, "listening", "speaking")
    await asyncio.sleep(config.HOLD_AFTER_S + 0.4)
    check("dropped while the agent is still speaking",
          not any(t in config.HOLD_PHRASES for t in s.texts), f"said={s.texts}")

    # "One moment." arriving after "have a wonderful day"
    call, s = await new_call()
    user_says(s, "that's all, thank you")
    await asyncio.sleep(config.HOLD_AFTER_S + 0.4)
    check("dropped on a caller sign-off",
          not any(t in config.HOLD_PHRASES for t in s.texts), f"said={s.texts}")

    call, s = await new_call()
    user_says(s, "I appreciate it, but first can you check the price")
    await asyncio.sleep(config.HOLD_AFTER_S + 0.4)
    check("still fires on a polite turn that is asking for something",
          any(t in config.HOLD_PHRASES for t in s.texts), f"said={s.texts}")

    print("\nrepetition bounds")

    # "Take your time." x68 on one call
    call, s = await new_call()
    for i in range(6):
        await call.gate.say("Are you still there?", kind=f"k{i}", priority=Priority.CHECKIN)
    spoken_n = s.texts.count("Are you still there?")
    check("one line cannot exceed MAX_LINE_REPEATS per call",
          spoken_n == config.MAX_LINE_REPEATS,
          f"spoken {spoken_n}x, cap {config.MAX_LINE_REPEATS}")

    call, s = await new_call()
    user_says(s, "let me change my mind")
    agent_says(s, "Of course, what would you like to change?")
    before = call.gate.turn
    user_says(s, "I want to look for ninety days")   # inside SPLIT_TURN_GRACE_S
    check("a split fragment does not open a new turn", call.gate.turn == before,
          f"turn {before} -> {call.gate.turn}")

    # the 846-turn call
    call, s = await new_call()
    for i in range(config.STALL_LIMIT + 2):
        user_says(s, f"unrelated chatter {i}")
        agent_says(s, "Please call 911 right away immediately.")
        # the gap belongs AFTER the agent line, or the next caller turn reads as a
        # split fragment of it and never closes the turn being scored
        await asyncio.sleep(config.SPLIT_TURN_GRACE_S + 0.05)
    check("STALL_LIMIT consecutive turns saying nothing new mutes the agent",
          s.input.audio_enabled is False, f"muted={call.metrics.muted_reason}")

    call, s = await new_call()
    call.gate.turn = config.MAX_TURNS
    user_says(s, "hello")
    check("MAX_TURNS mutes the agent", s.input.audio_enabled is False,
          f"muted={call.metrics.muted_reason}")

    print("\necho accounting (whose line was that?)")

    call, s = await new_call()
    await call.gate.say("Let me check that for you.", kind="hold",
                        priority=Priority.SCRIPTED)
    agent_says(s, "Let me check that for you.")
    check("our own line echoed back is not scored against the model",
          call.gate.narrations == [] and call.gate.duplicates == [],
          f"narrations={call.gate.narrations} duplicates={call.gate.duplicates}")

    # say() reports forwarded_text, so an interrupted line comes back clipped
    call, s = await new_call()
    await call.gate.say("Just a moment while I pull that up.", kind="hold",
                        priority=Priority.SCRIPTED)
    agent_says(s, "Just a moment while I")
    check("a clipped echo of an interrupted server line is not scored either",
          call.gate.narrations == [], f"narrations={call.gate.narrations}")

    call, s = await new_call()
    agent_says(s, "I'll pull that up for you.")
    check("the model inventing its own narration IS counted",
          len(call.gate.narrations) == 1, f"narrations={call.gate.narrations}")

    call, s = await new_call()
    await call.gate.say(config.PRICE_DISCLAIMER, kind="price_facts",
                        priority=Priority.ENFORCE)
    agent_says(s, config.PRICE_DISCLAIMER)   # the echo, consumed
    agent_says(s, config.PRICE_DISCLAIMER)   # the model saying it AGAIN
    check("the model repeating a server line IS counted as a duplicate",
          len(call.gate.duplicates) == 1, f"duplicates={call.gate.duplicates}")

    print("\nthe opening")

    call, s = await new_call(verified=False)
    await call.open()
    heard = dict(s.said)
    check("compliance greeting is spoken with interruptions disallowed",
          heard.get(prompt.GREETING) is False, f"said={s.said}")
    check("verification question stays interruptible",
          heard.get(prompt.LINE_NUMBER_CHECK) is True)

    print("\nthe emergency backstop")

    call, s = await new_call()
    user_says(s, "I think I took too much of my medication")
    agent_says(s, "This sounds like an emergency. Please hang up and dial "
                  "nine one one right away.")
    await asyncio.sleep(config.EMERGENCY_END_S + 0.5)
    check("ends a call the model announced but left open",
          call.toolbox.end_call_reason == "emergency",
          f"reason={call.toolbox.end_call_reason}")

    call, s = await new_call()
    agent_says(s, "This sounds like an emergency. Please call nine one one right away.")
    await asyncio.sleep(config.EMERGENCY_END_S + 0.5)
    check("stands down on a line fired into silence, rather than hanging up",
          call.toolbox.end_call_reason is None and call.metrics.spurious_emergency,
          f"reason={call.toolbox.end_call_reason} "
          f"spurious={call.metrics.spurious_emergency}")

    check("a denial alone is not emergency evidence",
          PharmacyCall._is_emergency("no, this is not an emergency") is False)
    check("a denial plus a real symptom still counts",
          PharmacyCall._is_emergency("this is not an emergency but I took too much")
          is True)

    print("\nenforced price facts")

    call, s = await new_call()
    agent_says(s, "The price is estimated to be fifty dollars.")
    check("an 'estimate' hedge alone does not satisfy the disclosure",
          call._disclaimer_said is False, f"parts={call._disclaimer_parts}")
    agent_says(s, "Your actual price may differ.")
    check("both halves together do", call._disclaimer_said is True)

    call, s = await new_call()
    call._owed_supply = (90, 84)
    agent_says(s, "That dispenses as an eighty four day supply.")
    check("the dispensed number alone does not satisfy the conversion notice",
          call._supply_said is False, f"parts={call._supply_parts}")
    agent_says(s, "Your ninety day request is what that covers.")
    check("both numbers together do", call._supply_said is True)

    print("\ncaller silence and the keypad")

    call, s = await new_call()
    call._idle_since = 0.0        # a long idle
    call.gate.turn = 3            # past the greeting allowance
    for _ in range(6):
        await call._on_caller_idle()
    check("check-ins rotate rather than repeating one line",
          len(set(s.texts)) > 1, f"said={s.texts}")
    check("the sequence ends in the goodbye and mutes",
          config.SILENCE_GOODBYE in s.texts and s.input.audio_enabled is False)

    call, s = await new_call()
    await call.keypad("936722")
    check("keypad digits reach the model before a mute",
          s.replies == ["[keypad] 936722"], f"replies={s.replies}")

    call, s = await new_call()
    call._mute("check")
    await call.keypad("936722")
    check("keypad digits are dropped after a mute -- no turn by the back door",
          s.replies == [], f"replies={s.replies}")

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks pass")
    if failed:
        print("\nFAILED:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
