# Pharmacy Voice Agent — two orchestration paths

A pharmacy support voice agent, built twice: once on the **Deepgram Voice Agent
API**, and once on **LiveKit Agents** driving the same Deepgram models. Same
prompt, same knowledge bases, same nine tools, same guardrails — the only
difference is who orchestrates the call.

| | | |
|---|---|---|
| [`voice-agent-api/`](voice-agent-api) | One websocket does listen, think, speak and turn-taking | Phone (Twilio) + browser client |
| [`livekit/`](livekit) | LiveKit orchestrates three separate Deepgram services | Phone (SIP via LiveKit Cloud) |

The two are held identical by assertion, not by discipline —
`livekit/scripts/check_parity.py` runs 81 checks across the prompt (byte-for-byte),
every tool schema, all three knowledge bases, every server-spoken line, and the
actual output of the shared lookup code on sample inputs. Run it before quoting
any comparison between the paths.

## What the agent does

Answers an inbound pharmacy call: verifies the caller with a code, looks up
medications and prices across a 247-drug catalog, answers operational questions
(delivery, refills, insurance, transfers, billing), reports its own per-turn
latency, and hands off to a pharmacist for anything clinical.

The things it is built to survive are the things a real pharmacy line gets: a
caller mispronouncing a drug name, going quiet for ten seconds while reading a
code off their phone, changing their mind mid-sentence, or talking over the
agent.

## Start here

The browser client is the fastest way to hear it, and needs only a Deepgram key —
no telephony setup:

```bash
cd voice-agent-api
uv sync
cp .env.example .env      # add DEEPGRAM_API_KEY
uv run uvicorn server.app:app --port 7860
```

Then open http://localhost:7860.

For the LiveKit path, `uv run python -m agent.main console` gives you the same
agent in a terminal with no LiveKit account required. Each directory's README
covers the phone paths, deployment, and every configuration knob.

## Demo data

Everything the agent knows is synthetic and lives in three JSON files per path:
`catalog.json` (medications, strengths, prices), `knowledge-base.json` (pharmacy
operations), and `account.json` (one demo caller). There is no real patient data
here, no real account, and no real verification — the demo caller's record
carries no name, date of birth, or address, and account dates are stored as
relative offsets so they never go stale.

**Prices are illustrative placeholders, not quotes.**

## License

MIT — see [LICENSE](LICENSE).
