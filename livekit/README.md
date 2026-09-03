# Pharmacy LiveKit Agent — the same agent, orchestrated by LiveKit

The parallel demo to [`../voice-agent-api`](../voice-agent-api): **the same
agent**, orchestrated by LiveKit Agents instead of the Deepgram Voice Agent API.
Same prompt, same three knowledge bases, same nine tools on the same schemas,
same scripted lines and guardrails — and the same three models. It exists so the
two orchestration models can be compared with everything else held constant.

```
Voice Agent API:  Twilio media stream -> Deepgram Voice Agent API
                  (one websocket does listen, think, speak, turn-taking)

LiveKit:          Twilio SIP trunk -> LiveKit Cloud SIP -> LiveKit room
                  -> this worker: nova-3-medical_pharma -> gpt-4o-mini
                  -> flux-brooke-en    (three services, LiveKit orchestrating)
```

Both can run at the same time. This worker connects outbound to LiveKit Cloud, so
it needs no tunnel and no inbound port; LiveKit Cloud terminates SIP.

## The same agent, and how that is kept true

The pairing is only worth anything if the **only** difference is orchestration.
So the shared surface is not "kept in sync by hand" — it is asserted:

```bash
uv run python scripts/check_parity.py
```

81 checks across the prompt (byte-for-byte), all nine tool schemas, the three
knowledge bases, every server-spoken line, every threshold and marker list, and
the actual output of `catalog.lookup` / `catalog.price` / `knowledge.search` /
`spoken.*` on sample inputs. **Run it before quoting any comparison.** A reworded
tool description does not break anything; it just moves this path's behaviour, and
the next comparison then attributes the move to LiveKit — a wrong conclusion
presented as a measurement.

Parity of *configuration* is not parity of *behaviour*, though, and the layer that
decides who speaks and when was rebuilt rather than copied. So that layer gets
replayed through every live-call collision it was built for:

```bash
uv run python scripts/check_speech.py     # 27 checks, no network
uv run python scripts/check_catalog.py    # the same knowledge-base validation
```

### What is genuinely shared

| | |
|---|---|
| `catalog.json`, `knowledge-base.json`, `account.json` | copied, validated identically on both paths |
| `prompts/system-prompt.md` | copied |
| `agent/catalog.py`, `agent/knowledge.py`, `agent/spoken.py` | copied verbatim — none of them ever knew about the transport |
| `agent/tools.py` | copied verbatim, `ToolBox` guards and all |
| `agent/prompt.py`, `agent/config.py` | ported; every phrase and threshold verbatim |

The copies are deliberate rather than factored into a shared package: each path
stays independently runnable, and `check_parity.py` can only assert sameness
between two things that are genuinely separate.

### What is deliberately different

| | Voice Agent API | here |
|---|---|---|
| STT client | Voice Agent `listen` provider | `deepgram.STT` (Nova, /v1) or `STTv2` (Flux, /v2) |
| LLM client | Deepgram-managed `open_ai` | `livekit-plugins-openai` — **needs your own key** |
| TTS client | Voice Agent `speak` provider | `deepgram.TTSv2` |
| turn-taking | Deepgram's endpointing alone | Deepgram's **plus** LiveKit's own delay |
| latency | `LatencyReport.total_latency` | `eou_delay + llm_ttft + tts_ttfb` |
| server speech | `InjectAgentMessage` | `AgentSession.say()` |
| model fallback | ladder at session open | `stt.FallbackAdapter`, also mid-call |

## The one credential the two paths cannot share

The Voice Agent API path runs its LLM as a **Deepgram-managed provider**, so its
`.env` holds no LLM key at all. LiveKit calls the provider directly, so this path
needs `OPENAI_API_KEY`. Everything else in `.env` carries across unchanged — the
var names are deliberately identical.

`THINK_PROVIDER=anthropic` switches to `claude-haiku-4-5` (needs
`ANTHROPIC_API_KEY`) and is wired, but read the note in `agent/config.py` first:
on this prompt haiku intermittently fails the estimate-disclaimer check, and one
run invented drug strengths belonging to a different product.

## Numbers from one path do not transfer to the other

Two reasons, both structural, and both mean this path needs its own measurement
run rather than inheriting the other's.

**Latency is a different stopwatch.** There is no `LatencyReport` here, because
there is no single session that sees all three components. `agent/metrics.py`
uses LiveKit's documented formula instead. It sums three independently measured
stages where a `LatencyReport` measures one pipeline end to end, and its
`end_of_utterance_delay` contains LiveKit's endpointing delay *stacked on top of*
the STT's own `endpointing`. Report both, labelled; do not present the difference
as a model result.

**The audio is a different channel.** The Voice Agent API path's listen-model
evaluation is run at mulaw 8 kHz on purpose — an evaluation run on the browser
path is not evidence about the phone path, because at 16 kHz the answer inverts.
Here Twilio hands the audio to LiveKit Cloud over SIP and LiveKit resamples into
the room, so the STT sees neither of the two channels that path measured.
`HOLD_AFTER_S` in particular is tuned against the other path's turn-latency
median and tool-call time; re-measure before trusting it.

## One-time setup

Human steps (browser, ~3 minutes):

1. Create a LiveKit Cloud account at cloud.livekit.io (the free Build tier covers
   this: SIP included, 1,000 SIP + 1,000 agent minutes/month).
2. `lk cloud auth` (`brew install livekit-cli` first if needed).
3. Create an API key in the dashboard and fill `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
   `LIVEKIT_API_SECRET` in `.env`.

Scripted steps, in order:

```bash
uv sync
bash scripts/setup_livekit.sh                       # LiveKit trunk + dispatch rule; prints the SIP host
uv run python scripts/setup_twilio_trunk.py <host>  # Twilio trunk -> that host, attaches the number
```

## Run

```bash
uv run python -m agent.main console   # local terminal test, no LiveKit account needed
uv run python -m agent.main dev       # connect to LiveKit Cloud, then call your number
```

`console` needs only `DEEPGRAM_API_KEY` and `OPENAI_API_KEY` — it is the fastest
way to hear the agent without any telephony setup at all.

## Files

- `agent/main.py` — worker entrypoint: the model stack, the listen ladder, DTMF, wiring
- `agent/session.py` — `PharmacyAgent` (the nine tools) + `PharmacyCall` (watchdogs, backstops, ending, turn admission)
- `agent/speech.py` — the arbiter for every server-spoken line; the other path's `SpeechGate` on LiveKit primitives
- `agent/tools.py` — `ToolBox` and `DEFINITIONS`, verbatim
- `agent/catalog.py` / `knowledge.py` / `spoken.py` — verbatim
- `agent/prompt.py` + `prompts/system-prompt.md` — the agent's instructions
- `agent/config.py` — every knob, with the measurement behind it
- `agent/metrics.py` — per-turn latency on LiveKit's formula, call summaries to `ops/metrics/`
- `scripts/check_parity.py` — asserts this path and the other are the same agent
- `scripts/check_speech.py` — replays the live-call collisions through the rebuilt speech layer
- `scripts/check_catalog.py` — validates the knowledge bases
- `scripts/setup_livekit.sh` / `setup_twilio_trunk.py` — one-time telephony setup

## Where LiveKit made the port smaller

Three mechanisms on the other path exist only to work around injections that
could be silently discarded, and all three have a native answer here — so roughly
120 lines of retry-and-confirm scaffolding did not need porting:

- **`_ensure_greeting`** (two attempts, three timeouts, watch for the words to
  come back) → `say(..., allow_interruptions=False)`. The compliance opening
  carries the 911 instruction and the recording disclosure; on the other path a
  caller saying "hello" during slow synthesis flushed it and the model improvised
  a welcome of its own. It cannot be flushed here.
- **`_ask_number_check`** (four attempts, because a line lost to barge-in was
  indistinguishable from one still queued behind a 15-second greeting) → the
  `SpeechHandle` tells the two apart, so an interruption is logged and left alone.
- **`_mute_after_closing`** (wait for the closing's marker on a timer, having
  learned that muting in the same breath cut the audio needed to synthesise it) →
  `await handle.wait_for_playout()`.

Two more map rather than shrink: withholding a `FunctionCallResponse` so the
model gets no turn to repeat itself becomes `raise StopResponse`, and dropping
caller audio at the socket becomes `session.input.set_audio_enabled(False)`.

Everything a guard actually *decides* came across unchanged. Where a guard is
gone rather than ported, the docstring in `agent/session.py` says which one and
why — a missing guard should be a decision, not an omission.

## Not carried over

The other path's browser client, operator console, `/capabilities`, `/state` and
`/latency` endpoints, and its Fly deployment. This is a worker, not a web service:
it has no HTTP surface. The trace export is also absent — it correlates on a
Voice Agent websocket handshake header, so it would need rebuilding against
LiveKit's own telemetry rather than porting.
