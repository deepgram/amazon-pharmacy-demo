# Pharmacy Voice Agent — Deepgram Voice Agent API

A pharmacy support agent built on the **Deepgram Voice Agent API**. One agent,
two front doors: a phone number (Twilio) and a browser client. It covers OTP
verification (spoken or keypad), a 247-medication knowledge base, mishear
recovery, natural number speech, and per-turn latency reporting.

The same agent orchestrated with LiveKit instead is in [`../livekit`](../livekit).

## How it works

```
Phone:    caller -> Twilio number -> <Connect><Stream> -> /ws  ─┐
                                                                ├─> Deepgram Voice Agent API
Browser:  mic -> /ws-browser ───────────────────────────────────┘    (nova-3-medical_pharma STT
                                                                      + gpt-4o-mini
                                                                      + flux-brooke-en TTS,
                                                                      one websocket)
```

The server is a thin bridge. All speech understanding, turn-taking, thinking, and
synthesis happen inside the Voice Agent API session; the server just moves audio
and answers tool calls (`send_otp`, `verify_otp`, `end_call`, `transfer_to_agent`,
`lookup_medication`, `check_price`, `search_knowledge_base`, `get_account_info`)
against the three knowledge bases.

The exact models live in `server/config.py` and are surfaced at
`GET /capabilities`, which is what the browser sidebar renders — don't hand-copy
them anywhere. The inbound phone number is served the same way, from
`TWILIO_FROM_NUMBER`.

Swapping Twilio for other telephony is a front-door change only: whatever feeds
audio can feed this same websocket, and nothing downstream changes.

## Run it

```bash
uv sync
cp .env.example .env          # fill in DEEPGRAM_API_KEY at minimum

# 1. start the server
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860

# 2. open http://localhost:7860 for the browser client
```

The browser client needs nothing but a Deepgram key. For the phone path, add
your Twilio credentials and a number, then point that number here:

```bash
ngrok http 7860                                   # PUBLIC_URL in .env must match
uv run python scripts/set_webhook.py set          # saves the old webhook first
uv run python scripts/set_webhook.py revert       # put it back
```

## Deploy (Fly)

Fly gives the app a stable `wss://` hostname, which replaces ngrok for the phone
path and gives the browser client the secure context `getUserMedia` requires.

```bash
fly launch --no-deploy                            # first time
fly secrets set DEEPGRAM_API_KEY=... CONSOLE_PASSWORD=...
fly deploy --remote-only                          # no local Docker daemon needed
fly scale count 1                                 # see below
```

**Stay at one machine.** `/state` and `/console` read per-call state out of
process memory, so a second machine serves a console that knows nothing about the
live call. Cold starts would also land on the exact latency this demo exists to
show, hence `min_machines_running = 1` and `auto_stop_machines = false`. Fly
re-adds a high-availability machine on some deploys — check afterwards.

Non-secret config (`PUBLIC_URL`, `TWILIO_FROM_NUMBER`, `OTP_SMS`) lives in
`fly.toml`. Rename the app there and in `PUBLIC_URL` together — the TwiML stream
URL is built from it.

## Knowledge bases

Three files, all loaded at startup, all reachable from tools. Together they exist
so the agent never has to say "I can't check that" — an earlier build had four
tools, two of them OTP, so anything that wasn't a drug price dead-ended.

**`catalog.json` — medications.** 247 medications, 711 strengths, **1,355 priced
strength/day-supply combinations**. Every strength on file carries a full price
row, so any dose a caller names has an answer. Covers the top-dispensed US drugs
across 19 therapeutic areas, and resolves brands to generics both ways (Lipitor →
Atorvastatin, "my water pill" → Hydrochlorothiazide).

| What it exercises | Entries |
|---|---|
| Mishear recovery | `nova-3-medical_pharma` transcription plus fuzzy resolution &mdash; no alias table, no keyterms |
| One name, several products | semaglutide → Ozempic / Wegovy / Rybelsus; tirzepatide → Zepbound / Mounjaro |
| Units that aren't milligrams | mcg, units per mL, mcg per puff, percent, grams, mEq |
| Out of stock | Wegovy |
| No insurance coverage | Wegovy (`insurance_avg: null` — never quote "zero dollars") |
| Not a monthly supply | 10-day courses, Z-Paks, tapers, Schedule II 30-day caps |
| Regulatory boundary | Adderall XR, Vyvanse, Concerta — non-transferable, 30-day cap |

**`knowledge-base.json` — pharmacy operations.** 92 entries across 15 topics:
delivery, orders, refills, transfers, prescriptions, insurance, pricing, payment,
PillPack, controlled substances, pharmacist, clinical, account, privacy, support.
Searched by `search_knowledge_base` with a scorer calibrated so real questions
come back high-confidence and off-topic chatter comes back low. **It never
returns empty** — a low-confidence answer plus an offer to connect someone beats
a refusal.

**`account.json` — the demo caller.** Prescriptions and orders (one in flight,
one waiting on a prescriber, one delivered), read by `get_account_info`. This is
what answers "where is my order", "when is my refill due", "what am I taking".
The record is synthetic and deliberately thin on identity — no name, no date of
birth, no address, because verification is the calling number plus a code and the
agent never asks for anything else. Dates are stored as **relative day offsets**
and rendered at call time, so the demo never goes stale sitting in the repo.

### What still gets refused

Clinical judgment — what to take, how much, whether two drugs are safe together,
what a symptom means — routes to a pharmacist, and emergencies route to 911. That
boundary is deliberate and stays: it's a compliance line for a pharmacy agent, not
a coverage gap. The prompt frames it as a warm handoff to a named expert who is
available around the clock, not as an apology.

### Keeping it honest

```bash
uv run python scripts/check_catalog.py
```

Checks all 1,355 price cells resolve, every alias lands on the right drug, the
reference call's numbers haven't drifted, 24 must-answer questions reach a
confident answer, junk queries don't, KB answers contain no characters that get
read aloud, and every prescription on the demo account exists in the catalog and
prices. Add `--aliases` to print the alias table. **Prices are demo placeholders,
not quotes.**

Adding a drug is a `catalog.json` edit and nothing else — STT keyterms, the
prompt's therapeutic-area list, and the pronunciation guide are all generated
from it.

**Prompt budget.** Only what tools can't supply goes in the system prompt. The
pronunciations for high-traffic drugs, the therapeutic-area list, and the policy
set are generated; everything else rides back on `lookup_medication` for the drug
actually in play. The 900+ aliases stay out entirely because the fuzzy match
already resolves them, and the per-drug rules — controlled schedules, cold chain,
refill and day-supply caps — come back as a `guidance` field on the tool result
rather than being stated in the abstract on every turn. Enumerating all of it
would cost roughly 4,000 tokens a turn, and time-to-first-audio is the metric
that matters.

`PROMPT_LIMIT` caps the assembled prompt and `check_catalog.py` fails above it
and warns within 5%, so a prompt that has grown past what the session will carry
is a build failure rather than a surprise on a live call.

The complete prompt is sent once in `Settings`. This use case does not need
`UpdatePrompt`; keeping one stable prompt removes a runtime state transition
while the verification flow and tool boundaries still prevent medication help
before verification.

## Choosing the listen model

**Shipped default: `nova-3-medical_pharma`, with an automatic fallback to
`flux-general-en`.** Note the underscore — `nova-3-medical-pharma` is rejected
with `FAILED_TO_START_LISTENING`.

`VoiceAgentSession.connect()` treats a missing `SettingsApplied` (or an `Error`
before one) as a rejection and reopens on `flux-general-en`, so a listen model
that will not start degrades to a working demo instead of a dead call. The model
that actually started is logged, returned in `GET /state`, and written into every
`ops/metrics/` summary as `listen_model` — **check it before trusting a latency
run**, because the two models make different trade-offs and a fallback you didn't
notice will be attributed to the wrong thing.

The trade-off between them is real and worth measuring on your own audio. The
medical model hears drug names the general model garbles; the general model
returns first audio sooner and has semantic end-of-turn, which is what lets a
caller go quiet for five to ten seconds while reading a verification code.
`LISTEN_ENDPOINTING_MS` is what stops the medical model answering half a
sentence; Flux rejects the key and uses `EOT_THRESHOLD` / `EOT_TIMEOUT_MS`
instead. To A/B live, set `LISTEN_MODEL=flux-general-en` — endpointing, `eot_*`
and keyterm handling all switch automatically.

**Measure on the path you will ship.** Twilio delivers `mulaw` 8 kHz (`app.py`
`MULAW_8K`); the browser widget is `linear16` 16 kHz. The two disagree enough to
invert the answer: at 16 kHz one model transcribes "escitalopram" where the other
returns "a citalopram", and at 8 kHz **both** return "the citalopram". An
evaluation run on the browser path is not evidence about the phone path.

**Keyterms.** `LISTEN_KEYTERMS=auto` sends them for Flux only, which is where
they measured as helping. On the medical model they were a no-op and once
actively harmful: with them on, a misheard *Olmesartan* became "omeprazole", a
real catalog drug, so the error went silent; without them it stayed nonsense the
fuzzy matcher caught. A mishearing that lands on a different real drug is worse
than one that lands on gibberish.

## Who gets to talk

Four things can make this agent speak: the model's pre-tool completion, its
post-tool completion, the scripted verification lines, and the dead-air watchdog.
Only the first two are the model's. The other two are ours, and they used to be
coordinated by six mutable flags and four independently tuned timeouts scattered
through `va.py` — which is how one live call ended up with ten filler phrases,
half of them landing *after* the answer they were covering.

`server/speech.py` now owns every server-spoken line. `SpeechGate.say()` is the
only path to `InjectAgentMessage`, and it decides using three things the old
flags could only approximate:

| | how it is known |
|---|---|
| `speaking` | set by binary audio frames, cleared only by `AgentAudioDone` or `UserStartedSpeaking` (barge-in stops the agent). **Not** `AgentStartedSpeaking` — that event is not emitted on the shipped listen config. A new turn deliberately does *not* clear it: on a long reply the caller can start talking several seconds in, and clearing it there let filler fire over audio that was still playing |
| `model_spoke` | an assistant `ConversationText` that does not match a line the gate itself injected |
| `turn` | incremented only on a genuine user turn, so a split fragment does not open a new one |

Three priorities, and only the lowest is ever dropped for being unnecessary:

- **`SCRIPTED`** — the verbatim verification lines. Always play, subject to a
  per-kind cooldown so a double-called tool cannot say them twice.
- **`ENFORCE`** — the price disclaimer and day-supply conversion. Play even if
  the model already spoke; that is the whole point of them.
- **`FILLER`** — hold phrases. Dropped if the agent is speaking, if the model
  already answered this turn, if a scripted line already covered the turn, if the
  same phrase already played this turn, or if the turn has moved on since the
  watchdog armed. Every one of those was a real collision in a recorded call.

Each decision is logged with its reason, so "why didn't the hold phrase play" is
answerable from the log instead of by inference:

```
[abc123] speech[hold] dropped: model already answered this turn
[abc123] speech[scripted:verify_otp/queue]: 'Let me verify that for you.'
```

**The gate also measures the two things it cannot fix.** Both are
prompt-adherence problems, and a gate can only count them:

- `duplicate_lines` — the model emitted a line the server had already spoken, so
  the caller hears it twice. The gate knows what it injected, so it can tell.
- `narrated_lines` — the model announced a lookup at all: "I'll look that up for
  you", "let me get that price". The prompt forbids the whole class of sentence,
  but the model keeps inventing wordings for it, so a blocklist cannot catch it
  and a matcher on the *shape* of the sentence does.

Watch both when changing the prompt's no-narration rule. The rule was a list of
banned phrases and the model simply produced phrases that were not on it;
rewriting it to forbid the category rather than the wording halved the rate, and
the residue is visible in `narrated_lines` rather than being assumed away.

Measured over six hostile runs: 0/6 with repetition, 0 duplicated lines, and 2.0
filler phrases per call against a historical worst of 12.

## Latency

**The number to quote is `LatencyReport.total_latency`.** It is emitted once per
model turn with a full STT / LLM / TTS breakdown, and `server/metrics.py` stores
the whole breakdown per turn. Call summaries print at hangup, persist to
`ops/metrics/`, and are served at `GET /latency`. The browser client shows the
last turn live and labels which stopwatch produced it.

Three sources, in descending order of authority — `metrics.turn_total()` picks:

1. `LatencyReport` — authoritative, and the only one with a stage breakdown.
2. `AgentStartedSpeaking` — carries `total_latency` too, but **is not emitted on
   every listen configuration**, so it cannot be the primary source.
3. Wall clock, transcript-in to first-audio-byte — last resort.

**Wall clock is discarded on any turn where the server injected speech.** A hold
phrase or a scripted verification line is *our* audio, so timing it reports the
watchdog's reaction time as the agent's latency. Every `InjectAgentMessage` goes
through `VoiceAgentSession._inject()`, which tags the turn; tagged turns are
excluded from the `wall_ttfa_s` summary and counted separately as
`injected_turns`. Watch that count — a call where it climbs is a call that sounds
like it is stalling, and it was previously invisible.

**Where the time goes.** The LLM dominates the total and TTS is nearly free, so
the prompt and the think model are the only levers that move the headline number.
Nothing else is worth optimising until those are.

The stage breakdown explains the spread. `ttt_token_latency` — first token of any
kind — is flat across turns. What varies is whether that first token is
speakable: on a turn where the model emits a tool call first and text only after
the result comes back, `ttt_text_latency` roughly doubles. Turns where the model
speaks *while* calling the tool stay fast. So the expensive turns are not "tool
turns", they are turns where the model chose to stay silent until the tool
returned — which is a prompt-shaped behaviour, not an infrastructure one.

`HOLD_AFTER_S` sits just above the tool-call turn time, so filler covers only the
genuinely slow tail. It was 1.0 s, which fired on nearly every turn — one live
call came back with twelve filler phrases.

## Knobs (.env)

Defaults below are the ones in `server/config.py`, which is the source of truth.
`fly.toml` overrides `PUBLIC_URL`, `TWILIO_FROM_NUMBER` and `OTP_SMS` in
production.

**Models**

| Var | Default | What |
|---|---|---|
| `LISTEN_MODEL` | `nova-3-medical_pharma` | STT. Underscore, not hyphen. Falls back to `flux-general-en` if it will not start |
| `THINK_MODEL` | `gpt-4o-mini` | LLM, provider `open_ai`, Deepgram-managed. Set `THINK_PROVIDER` to change family. `claude-haiku-4-5` (provider `anthropic`) is the supported fallback — see the note in `config.py` before switching |
| `VOICE_MODEL` | `flux-brooke-en` | Flux TTS voice (healthcare-tagged). A/B: `flux-sienna-en`, `flux-alexis-en`, `flux-haley-en`. Aura fallback: any `aura-2-*-en`, which switches `speak.version` to v1 automatically |
| `THINK_TEMPERATURE` | `0.4` | Omitted entirely for gpt-5, which rejects any explicit temperature |
| `REASONING_MODE` | `low` | OpenAI 5.0+ only. Anything higher is death for voice latency |

**Turn-taking**

| Var | Default | What |
|---|---|---|
| `LISTEN_ENDPOINTING_MS` | `500` | Nova-only: silence before the turn is called. Without it, utterances split and the agent answers half a sentence. Flux rejects it |
| `EOT_THRESHOLD` | `0.7` | Flux-only end-of-turn confidence |
| `EOT_TIMEOUT_MS` | `5000` | Flux-only hard turn timeout. This is what lets a caller go quiet while reading their code |
| `EAGER_EOT_THRESHOLD` | `0.4` | Flux-only. Starts the LLM on a medium-confidence turn end. `0` disables. Must be ≤ `EOT_THRESHOLD` |
| `LISTEN_KEYTERMS` | `auto` | `auto` = Flux only, which is where they measured as helping. `on`/`off` to force |
| `SPLIT_TURN_GRACE_S` | `1.5` | A user turn this soon after the agent spoke is treated as a split fragment, not a new turn |

**Speech and pacing**

| Var | Default | What |
|---|---|---|
| `SPEAK_SPEED` | `1.0` | Flux accepts discrete `0.85`–`1.15` only (Aura `0.7`–`1.5`). Not monotonic: 0.9/0.95/1.0 are all ~185 wpm, only 0.85 meaningfully slows the read |
| `EXPRESSIVITY` | `0` | Flux v2 only, `-2`..`2`, beta. Held at 0, which is the production-validated value; non-default values raise pronunciation-error risk, and that lands squarely on drug names. `0` omits the key entirely |
| `HOLD_AFTER_S` | `2.5` | Dead-air watchdog: agent silence after the caller stops before a hold phrase is injected. Sits just above tool-call turn time |
| `HOLD_PHRASES` | 3 phrases, `\|`-separated | Rotated so a long call does not repeat one |
| `HOLD_PHRASE_2` | "Thanks for your patience…" | Second stage, ~5 s later. Never loops |
| `SILENCE_AFTER_S` | `15` | Caller-idle check-in after the agent finishes and nobody speaks. Above the 5–10 s OTP-wait case. `0` disables. Three stages, then a polite hangup |
| `SCRIPTED_LINE_COOLDOWN_S` | `6` | Don't re-speak a scripted verification line inside this window |
| `DISCLAIMER_GRACE_S` | `4.0` | How long the model gets to say the price disclaimer itself before the server does |
| `PRICE_DISCLAIMER` / `SUPPLY_CONVERSION_TEMPLATE` | see config | Server-enforced because the model said them 1 time in 7. `""` to go back to trusting the prompt |

**Ops**

| Var | Default | What |
|---|---|---|
| `TOOL_LATENCY_MS` | `0` | Simulated tool latency, for feeling what a real backend costs |
| `OTP_SMS` | `1` | Text the OTP via Twilio. Needs an A2P 10DLC-registered number, or carriers drop it; the operator console is the fallback source |
| `DEMO_OTP` | `936722` | Fixed code phone callers get. A demo fixture against a synthetic account, not a secret — **there is no real verification here** |
| `CONSOLE_USER` / `CONSOLE_PASSWORD` | `demo` / unset | Basic auth for `/console`, `/state`, `/latency`. Unset = open, which is fine on localhost and not on a public deployment |
| `TAGS` | `pharmacy-demo` | Surfaces in Deepgram request logs |
| `MIP_OPT_OUT` | `1` | Keeps this traffic out of model training |

## Files

- `server/app.py` — FastAPI: TwiML webhook, Twilio WS bridge, browser WS bridge, `/capabilities`, `/state`, `/console`, `/latency`
- `server/va.py` — Voice Agent API session (Settings, listen-model fallback, event pump, function calls, injections, keepalive)
- `server/tools.py` — the client-side tools, per-call OTP state, the server-side verification gate
- `server/speech.py` — the gate that owns every server-spoken line
- `catalog.json` / `knowledge-base.json` / `account.json` — the three knowledge bases (see above)
- `server/catalog.py` — fuzzy drug lookup + pricing, 90-day requests dispense as 84
- `server/knowledge.py` — pharmacy FAQ search + the demo caller's account
- `server/prompt.py` + `prompts/system-prompt.md` — the agent's instructions
- `server/metrics.py` — per-turn latency capture + call summaries
- `web/index.html` — browser client (mic, transcript, live latency, keypad box); the stack line and phone number render from `/capabilities`
- `web/console.html` — operator console (live OTP, latency, transcript)
- `web/data.html` — browsable view of everything the agent can reach
- `scripts/check_catalog.py` — validates the knowledge bases; run it after any catalog edit
- `scripts/check_container_files.py` — asserts the Docker image actually contains the knowledge bases
- `scripts/set_webhook.py` — flip/revert the number's inbound webhook
