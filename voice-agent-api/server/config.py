"""Environment configuration. Env-var names match the other demo repos so a shared .env carries over."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
VA_URL = os.environ.get("VA_URL", "wss://agent.deepgram.com/v1/agent/converse")
# Internal stems want HTTP Basic instead of Token. Empty = production Token auth.
VA_BASIC_AUTH = os.environ.get("VA_BASIC_AUTH", "")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "")

PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")

# Underscore in the id is required; the hyphenated form fails to start.
LISTEN_MODEL = os.environ.get("LISTEN_MODEL", "nova-3-medical_pharma")
EOT_THRESHOLD = float(os.environ.get("EOT_THRESHOLD", "0.7"))
EOT_TIMEOUT_MS = int(os.environ.get("EOT_TIMEOUT_MS", "5000"))
EAGER_EOT_THRESHOLD = float(os.environ.get("EAGER_EOT_THRESHOLD", "0.4"))  # 0 disables; Flux only
# Nova only. Silence Nova waits for before finalising a transcript -- dead air on
# every turn, before the LLM is even asked, so it looks like free latency to cut.
# It is not: this is also the only thing keeping a mid-sentence pause from being
# read as end-of-turn, and the model answers every end-of-turn it is given.
#
# 500 is the floor, measured rather than assumed. Against the run-sheet line
# "I think my insurance would cover, um, thirty days to get it started", on the
# production endpoint, 3 repeats each:
#
#     500ms   0/18 utterances split   median 1.76s
#     400ms   3/18 (all 3 disfluent)  median 1.67s
#     300ms   3/18 (all 3 disfluent)  median 1.57s
#     200ms   4/12 -- also ate the "I" in "Let me change my mind"
#
# The caller's pause after "um" lands between 400 and 500ms. Below 500 it splits
# into two turns and the agent starts answering over the caller. Do not read the
# lower medians as better: a split is answered sooner precisely because it is
# answering half a sentence.
#
# (An earlier 0.0% premature-endpoint rate was measured AT 500ms and says nothing
# about lower values. That mistake is what this note exists to prevent.)
LISTEN_ENDPOINTING_MS = int(os.environ.get("LISTEN_ENDPOINTING_MS", "500"))
LISTEN_KEYTERMS = os.environ.get("LISTEN_KEYTERMS", "auto")  # auto | on | off

# gpt-4o-mini since 2026-09-02, measured rather than assumed: it passes all six
# graded checks and reaches first token ~300ms sooner than claude-haiku-4-5, which
# is the largest remaining lever on turn latency now that Flux TTS costs 0.11s and
# endpointing is pinned at its measured floor.
#
# It took three tool-layer fixes to get there, and each one closed a gap that was
# only ever hidden by claude happening to behave: spoken strengths reaching
# check_price, a day supply the model pre-converted, and an emergency the model
# announced but never ended. Those fixes are model-independent and worth keeping
# whatever runs here.
#
# claude-haiku-4-5 remains a supported fallback and is one env var away, but do
# not treat it as a drop-in right now -- on the current prompt it intermittently
# fails the estimate-disclaimer check, and one run invented Zepbound strengths
# ("2.4 and 1.7 milligrams", which are Wegovy's). Re-run scripts/compare_think_models.py
# before switching back.
THINK_PROVIDER = os.environ.get("THINK_PROVIDER", "open_ai")
THINK_MODEL = os.environ.get("THINK_MODEL", "gpt-4o-mini")
REASONING_MODE = os.environ.get("REASONING_MODE", "low")  # gpt-5 only
THINK_TEMPERATURE = float(os.environ.get("THINK_TEMPERATURE", "0.4"))
VOICE_MODEL = os.environ.get("VOICE_MODEL", "flux-brooke-en")
SPEAK_VERSION = "v2" if VOICE_MODEL.startswith("flux-") else "v1"
SPEAK_SPEED = float(os.environ.get("SPEAK_SPEED", "1.0"))
EXPRESSIVITY = int(os.environ.get("EXPRESSIVITY", "0"))  # Flux v2 beta; 0 is the production value

TOOL_LATENCY_MS = int(os.environ.get("TOOL_LATENCY_MS", "0"))

# Filler after the caller stops and the model is still thinking.
#
# Tuned twice against measurement. At 2.5s it fired just BEFORE the answer landed
# rather than instead of it -- "Let me check that for you." at 21:24:21.712 and
# the answer at 21:24:22.571, 0.86s apart, which a listener hears as a filler in
# the wrong place. Raising it to 4.0s over-corrected: transcript-to-first-audio
# measures a 2.12s median, so 4.0s sat above almost every turn and the ordinary
# 2-3s gap got no cover at all -- silence with nothing in it.
#
# 2.8s is the gap between the two failures: clear of the 2.12s median so it does
# not pre-empt a normal answer, under the 3.2-5.0s a tool call takes so those are
# still covered.
HOLD_AFTER_S = float(os.environ.get("HOLD_AFTER_S", "2.8"))
# A caller signing off. The hold phrase exists to cover a lookup, and there is no
# lookup behind "that is all, thank you" -- but the watchdog fires on a timer, so
# on a live call it said "One moment." at 2.8s while the model was composing the
# closing, and the caller heard it AFTER "have a wonderful day". The session-state
# guard in _hold_phrase_watchdog cannot catch this: end_call had not been called
# yet at 2.8s, so there was no ending to detect. Only the caller's own words give
# it away in time.
GOODBYE_MARKERS = (
    "that is all", "that's all", "thats all", "that is it", "that's it",
    "nothing else", "no thanks", "no thank you", "that's everything",
    "thats everything", "all set", "we're done", "were done", "i'm done", "im done",
    "goodbye", "good bye", "bye", "have a good", "have a great", "take care",
    "thank you so much", "thanks so much", "thank you for your help",
    "thanks for your help", "appreciate it",
)
# Bounds the check so a long sentence that happens to end politely still gets its
# filler: a genuine sign-off is short and asks nothing.
GOODBYE_MAX_CHARS = 60
# Anything here means the caller is still working, whatever pleasantries surround
# it -- "I appreciate it, but first can you check the price" is not a sign-off.
# This carries the weight rather than the question mark, because speech-to-text
# routinely drops terminal punctuation and a question can arrive unmarked.
STILL_WORKING_MARKERS = (
    "can you", "could you", "would you", "will you", "do you", "did you", "is there",
    "check", "look up", "find", "tell me", "what", "when", "where", "which", "how",
    "why", "who", "price", "cost", "refill", "fill", "order", "transfer", "send",
    "i need", "i want", "i'd like", "id like", "first", "also", "but",
)

# The hold phrase asks again if it is refused, rather than giving up for the turn.
# Four tries a second apart covers the transient refusals -- the caller drawing
# breath, the agent finishing a syllable -- without chasing a turn that has moved on.
HOLD_ATTEMPTS = 4
HOLD_RETRY_S = 1.0

SPLIT_TURN_GRACE_S = float(os.environ.get("SPLIT_TURN_GRACE_S", "1.5"))
HOLD_PHRASES = [p.strip() for p in os.environ.get(
    "HOLD_PHRASES",
    "Let me check that for you.|One moment.|Just a moment while I pull that up.",
).split("|") if p.strip()]
HOLD_PHRASE_2 = os.environ.get("HOLD_PHRASE_2", "Thanks for your patience, I'm still checking on that.")

# Check-in after the agent finishes and the caller says nothing. 0 disables.
SILENCE_AFTER_S = float(os.environ.get("SILENCE_AFTER_S", "15"))


def _phrase_list(env: str, default: str) -> list[str]:
    return [p.strip() for p in os.environ.get(env, default).split("|") if p.strip()]


# Each stage rotates, the way HOLD_PHRASES already does. Fixed strings are why
# "Are you still there?" came back word for word on four calls and "Take your
# time." on nine, where the prompt allows each of them once.
SILENCE_NUDGES = _phrase_list(
    "SILENCE_NUDGES",
    "Are you still there?|I'm still here whenever you're ready.|Take your time, I'm here.",
)
SILENCE_NUDGES_OTP = _phrase_list(
    "SILENCE_NUDGES_OTP",
    "Take your time. I'm here when you're ready.|No rush at all.",
)
SILENCE_NUDGES_2 = _phrase_list(
    "SILENCE_NUDGES_2",
    "Are you still with me?|I can't quite hear you. Are you there?",
)
SILENCE_GOODBYE = os.environ.get(
    "SILENCE_GOODBYE",
    "I haven't heard from you, so I'll wait here. You can disconnect whenever you like. Take care.",
)
# Flat tuple of every line the server can speak during silence. Consumed by the
# offline transcript scripts to tell a server line from a model line.
SILENCE_PHRASES = (*SILENCE_NUDGES, *SILENCE_NUDGES_OTP, *SILENCE_NUDGES_2, SILENCE_GOODBYE)

# --- Turn admission -----------------------------------------------------------
# A Voice Agent session answers every detected end of turn, and the transport
# stays open after end_call so the caller can hang up in their own time. Put those
# together in a state that permits exactly one sentence -- the emergency line, the
# live-agent queue, the post-close silence -- and background noise triggering the
# VAD is enough to make the agent say that one sentence forever. One call ran to
# 846 turns against a 19-turn median and said "Please call 911 right away." 353
# times, degrading to "Please call 91 right way" by the end.
#
# None of these bounds fire on a healthy call: the ceiling is four times the
# longest legitimate call in a 135-call run, and three CONSECUTIVE turns that each
# repeat an earlier line is not something a progressing conversation does.
MAX_LINE_REPEATS = int(os.environ.get("MAX_LINE_REPEATS", "2"))
STALL_LIMIT = int(os.environ.get("STALL_LIMIT", "3"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "80"))
MAX_SILENCE_NUDGES = int(os.environ.get("MAX_SILENCE_NUDGES", "4"))

# While the caller is reading the code off their phone the server says nothing at
# all -- SILENCE_AFTER_S was set to 15s to sit above a 5-10s reading pause, and
# under cafe noise the pause still read as longer than it was and the agent asked
# "Are you still there?" over it. Real dead air still ends the call, just later.
OTP_SILENCE_FACTOR = float(os.environ.get("OTP_SILENCE_FACTOR", "3"))

# --- Emergency backstop ---------------------------------------------------------
# The emergency path was prompt-only: say the line, then call end_call. That is
# one instruction with two halves, and a model is free to obey the half it can
# say out loud. gpt-4o-mini did exactly that on 3 of 3 runs -- it told the caller
# to hang up and dial nine one one, then left the session open and kept taking
# turns. claude-haiku called end_call every time, so the gap was invisible until
# the think model changed.
#
# A safety behaviour that works only because the current model happens to be
# obedient is not a guardrail. The server watches for the agent's OWN emergency
# line and ends the call itself if no end_call follows.
#
# Keyed off the agent's line, not off classifying the caller: deciding whether a
# caller is really in danger is the judgement call the prompt explicitly refuses
# to make, and a false positive here hangs up on someone mid-sentence. Both
# markers are absent from the greeting, which says "dial nine one one" -- so the
# opening disclosure cannot arm this.
# Phrases in which the agent offers the caller a live person. A caller who then
# says just "yes" has asked for a transfer as clearly as one who said "agent",
# and only the preceding offer carries that meaning -- so the session records it
# and tools.transfer_to_agent accepts it. Kept in sync with the medical-advice
# line in the prompt: "I can have a pharmacist call you back, or connect you now."
OFFER_AGENT_MARKERS = (
    "connect you", "call you back", "callback", "call back",
    "speak to a pharmacist", "speak with a pharmacist", "talk to a pharmacist",
    "transfer you", "live agent", "someone else", "representative",
    # The catalog's out-of-stock rule tells the agent to "offer to have a pharmacist
    # follow up", and none of the wordings above matched that. So the agent offered,
    # the caller said "yes, a callback would be great", and the guard refused a
    # transfer the agent had just proposed -- ending in "I can't arrange a callback
    # directly". Keep this list in step with the offers the guidance actually asks for.
    "pharmacist follow up", "pharmacist follow-up", "have a pharmacist",
    "follow up with you", "follow-up with you",
)

# What counts as the caller giving us a reason. Deliberately WIDE: the prompt says
# "Do not try to decide whether it is truly an emergency", and that instinct is
# right -- a false negative here is someone in trouble being talked past, a false
# positive is someone being told to hang up and dial 911 when they need not. Those
# are not comparable, so this errs heavily toward triggering, "urgent" included.
#
# It exists only to catch the case where the caller said NOTHING of the kind: on 4
# of 150 evaluated calls the agent produced the emergency line into dead air or at
# a call's natural end, and since the backstop now terminates the call on that
# line, a trigger nobody asked for hangs up on a healthy caller.
EMERGENCY_EVIDENCE = (
    "emergency", "911", "nine one one", "urgent", "took too much", "take too much",
    "overdose", "overdosed", "too many pills", "chest pain", "can't breathe",
    "cant breathe", "cannot breathe", "trouble breathing", "not breathing",
    "passed out", "pass out", "unconscious", "unresponsive", "collapsed",
    "bleeding", "allergic reaction", "anaphyla", "heart attack", "stroke",
    "seizure", "suicid", "kill myself", "hurt myself", "poison", "severe pain",
    "help me", "ambulance", "hospital", "er ", "emergency room",
)
# A caller saying it is NOT an emergency is not evidence that it is. Only consulted
# when "emergency" itself is the sole match, so "this is not an emergency, but I
# took too much" still counts.
EMERGENCY_DENIALS = (
    "not an emergency", "isn't an emergency", "isnt an emergency", "no emergency",
    "not urgent", "nothing urgent", "not a real emergency",
)

EMERGENCY_MARKERS = ("sounds like an emergency", "nine one one right away")
# Long enough for a model that is simply slower to get its end_call in.
EMERGENCY_END_S = float(os.environ.get("EMERGENCY_END_S", "6.0"))

DISCLAIMER_GRACE_S = float(os.environ.get("DISCLAIMER_GRACE_S", "4.0"))
# After the grace period, keep waiting while the agent is still talking rather
# than queueing behind an answer the model has not finished -- that is how the
# estimate caveat got said twice on a live call. Capped so a stuck `speaking`
# flag cannot suppress the disclaimer entirely.
DISCLAIMER_MAX_WAIT_S = float(os.environ.get("DISCLAIMER_MAX_WAIT_S", "30"))
SUPPLY_CONVERSION_TEMPLATE = os.environ.get(
    "SUPPLY_CONVERSION_TEMPLATE",
    "Just to be clear, a {requested} day request dispenses as a {dispensed} day supply.",
)
PRICE_DISCLAIMER = os.environ.get(
    "PRICE_DISCLAIMER",
    "Just keep in mind, the average insurance price is an estimate based on what "
    "other customers paid, so your actual price may differ.",
)
# How we tell the model already covered the disclaimer, so the server does not say
# it a second time.
#
# The disclosure has TWO halves -- this number is an estimate, AND your price may
# differ -- and a caller needs both. Matching on any single marker could not tell
# them apart: the model saying "the price is estimated to be fifty dollars" tripped
# "estimate", the server concluded the job was done and stayed quiet, and the caller
# was never told the figure might not be theirs. That is a hedge on a number, not a
# disclosure, and it is what made the graded disclaimer check flap.
#
# So both halves must appear before we treat it as said. The one exception is the
# distinctive phrase from PRICE_DISCLAIMER itself, which only shows up when the
# model really is reproducing the disclosure.
#
# The opposite failure is live too: an earlier version matched only the literal word
# and a caller heard the disclaimer twice because the model had paraphrased the whole
# idea without it. Hence the paraphrases below rather than one exact string.
DISCLAIMER_ESTIMATE_MARKERS = (
    "estimate", "estimated", "approximate", "approximately", "ballpark",
    "not exact", "roughly",
)
DISCLAIMER_VARIANCE_MARKERS = (
    "may differ", "might differ", "could differ", "may vary", "might vary",
    "can vary", "could vary", "varies by", "not guaranteed", "actual price",
    "actual cost", "your price may", "differ from",
)
# Reproducing this needs no second half; it IS the disclosure.
DISCLAIMER_WHOLE_MARKERS = ("what other customers paid",)

# The mandated close, from the reference call. The server owns it for
# the same reason it owns the greeting: leaving a required line to the model gets
# it said twice or not at all. Both were observed -- first the whole closing
# delivered twice, then, once end_call stopped feeding the model a follow-up
# turn, a call that ended on "How else can I help you?" and nothing else.
# LINE_CLOSING_MARKER is how va.py tells whether the model already said it, so it
# is never doubled.
LINE_CLOSING = os.environ.get(
    "LINE_CLOSING",
    "Glad I could help! Have a wonderful day, and please stay on the line for a "
    "brief survey on your experience today. You can disconnect whenever you like.",
)
LINE_CLOSING_MARKER = "have a wonderful day"

SCRIPTED_LINE_COOLDOWN_S = float(os.environ.get("SCRIPTED_LINE_COOLDOWN_S", "6"))
LINE_LOCKOUT = os.environ.get(
    "LINE_LOCKOUT",
    "I'm sorry, that code didn't match either. You can verify your account in the "
    "Amazon app or on our website. You can disconnect whenever you like.",
)
SCRIPTED_TOOL_LINES = {
    # Spoken BEFORE the tool runs, so it covers the round-trip rather than following
    # it. It was "Perfect, just pulling up your account." -- pure filler, which left
    # the model to make the actual announcement in a second turn, so the caller
    # waited through two round-trips and heard two messages for one event. This line
    # does the whole job: what is happening, and what to do about it. The prompt
    # tells the model to stay silent after send_otp returns.
    "send_otp": os.environ.get(
        "LINE_SEND_OTP",
        "I'm sending a six-digit code to your email and mobile phone now. "
        "Read it back to me when it arrives, or enter it on your keypad.",
    ),
    "verify_otp": os.environ.get("LINE_VERIFY_OTP", "Let me verify that for you."),
}

_QUEUE_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)
QUEUE_POSITION = int(os.environ.get("QUEUE_POSITION", "3"))
QUEUE_WAIT_MIN = int(os.environ.get("QUEUE_WAIT_MIN", "15"))


def _queue_words(n: int) -> str:
    n = max(0, int(n))
    return _QUEUE_ONES[n] if n < len(_QUEUE_ONES) else str(n)


LINE_TRANSFER = os.environ.get(
    "LINE_TRANSFER",
    "Transferring you to a live agent now. You are number "
    f"{_queue_words(QUEUE_POSITION)} in line, and the estimated wait is about "
    f"{_queue_words(QUEUE_WAIT_MIN)} minutes.",
)
SCRIPTED_TOOL_LINES["transfer_to_agent"] = LINE_TRANSFER

# Every line the SERVER can speak. The offline harnesses subtract this from a
# transcript to get what the MODEL actually said.
#
# It lived as a hand-rebuilt literal in four scripts, and the duplication was not
# theoretical: LINE_LOCKOUT was missing from all four at once, so on a lockout
# run the server's own closing read as the model's reply. Adding a scripted line
# here now reaches every harness. Defined after the lines above so it picks up
# any env overrides.
SERVER_SPOKEN_LINES = frozenset(
    line for line in (
        *HOLD_PHRASES, HOLD_PHRASE_2,
        *SCRIPTED_TOOL_LINES.values(),
        *SILENCE_PHRASES,
        LINE_LOCKOUT,
        PRICE_DISCLAIMER,
        LINE_CLOSING,
    ) if line
)

DEMO_OTP = "".join(ch for ch in os.environ.get("DEMO_OTP", "936722") if ch.isdigit())
OTP_SMS = os.environ.get("OTP_SMS", "1") == "1"


CONSOLE_USER = os.environ.get("CONSOLE_USER", "demo")
CONSOLE_PASSWORD = os.environ.get("CONSOLE_PASSWORD", "")

TAGS = [t for t in os.environ.get("TAGS", "amazon-pharmacy-demo").split(",") if t]
MIP_OPT_OUT = os.environ.get("MIP_OPT_OUT", "1") == "1"

PORT = int(os.environ.get("PORT", "7860"))
