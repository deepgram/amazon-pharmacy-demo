"""FastAPI app: Twilio phone bridge + browser client."""

import asyncio
import base64
import json
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from loguru import logger

from . import catalog, knowledge
from .config import (
    CONSOLE_PASSWORD,
    CONSOLE_USER,
    LISTEN_MODEL,
    PUBLIC_URL,
    THINK_MODEL,
    THINK_PROVIDER,
    TWILIO_FROM_NUMBER,
    VA_BASIC_AUTH,
    VA_URL,
    VOICE_MODEL,
)
from .metrics import LAST_SUMMARIES, turn_total
from .tools import ToolBox
from .va import VoiceAgentSession

app = FastAPI(title="pharmacy-voice-agent")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

class RevalidatingStatics(StaticFiles):
    """Serve /static with `Cache-Control: no-cache`.

    Without it Starlette sends only ETag and Last-Modified, and browsers fall
    back to heuristic freshness -- so a deploy that changes app.css can leave a
    demo machine rendering the previous stylesheet with no error anywhere. That
    happened: the data browser shipped correctly and came up unstyled, because
    the header rules were in the cached copy and the new ones were not.

    `no-cache` means revalidate, not "do not store". The ETag still answers 304
    on an unchanged file, so this costs one conditional request per asset, not a
    re-download. Right trade for a demo that gets redeployed repeatedly.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatingStatics(directory=WEB_DIR), name="static")


@app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
async def favicon():
    return FileResponse(WEB_DIR / "favicon.png", media_type="image/png")

# /console, /state, /latency: open locally, basic-auth when CONSOLE_PASSWORD is set.

_basic = HTTPBasic(auto_error=False)


def require_console(creds: HTTPBasicCredentials | None = Depends(_basic)) -> None:
    if not CONSOLE_PASSWORD:
        return
    ok = creds is not None and secrets.compare_digest(
        creds.username, CONSOLE_USER
    ) and secrets.compare_digest(creds.password, CONSOLE_PASSWORD)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="console auth required",
            headers={"WWW-Authenticate": 'Basic realm="pharmacy demo console"'},
        )


if not CONSOLE_PASSWORD:
    logger.warning("CONSOLE_PASSWORD unset: /console and /state are open (fine on localhost)")
if not PUBLIC_URL:
    logger.warning("PUBLIC_URL unset: the phone path is disabled (browser client still works)")

MULAW_8K = {"encoding": "mulaw", "sample_rate": 8000}
MULAW_8K_OUT = {"encoding": "mulaw", "sample_rate": 8000, "container": "none"}
LINEAR16_IN = {"encoding": "linear16", "sample_rate": 16000}
LINEAR16_OUT = {"encoding": "linear16", "sample_rate": 24000, "container": "none"}

# Seconds of keypad silence before the collected digits are submitted.
DTMF_PAUSE_S = 1.6


# --- Twilio ------------------------------------------------------------------


@app.api_route("/twiml-inbound", methods=["GET", "POST"])
async def twiml_inbound(request: Request):
    """Twilio inbound-voice webhook. Passes the caller's number to the stream
    so the console can show who is on the line."""
    form = await request.form() if request.method == "POST" else request.query_params
    caller = form.get("From", "")
    if not PUBLIC_URL:
        logger.error("PUBLIC_URL unset: cannot build the media-stream URL for Twilio")
        return Response(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>'
            "Demo server misconfigured. Public U R L is not set.</Say></Response>",
            media_type="text/xml",
        )
    ws_url = PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="caller" value="{caller}" />
    </Stream>
  </Connect>
  <Pause length="2"/>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/ws")
async def twilio_ws(websocket: WebSocket):
    await websocket.accept()
    stream_sid: str | None = None
    session: VoiceAgentSession | None = None
    label: str | None = None
    dtmf_digits: list[str] = []

    async def to_twilio_audio(chunk: bytes):
        if stream_sid:
            await websocket.send_json(
                {"event": "media", "streamSid": stream_sid,
                 "media": {"payload": base64.b64encode(chunk).decode()}}
            )

    async def on_va_event(event: dict):
        # barge-in: caller started talking -> flush Twilio's audio buffer
        if event.get("type") == "UserStartedSpeaking" and stream_sid:
            await websocket.send_json({"event": "clear", "streamSid": stream_sid})

    dtmf_timer: asyncio.Task | None = None

    async def submit_dtmf():
        nonlocal dtmf_timer
        dtmf_timer = None
        digits = "".join(dtmf_digits)
        dtmf_digits.clear()
        if digits and session:
            logger.info(f"[{stream_sid}] keypad entry: {digits}")
            await session.inject_user_text(f"[keypad] {digits}")

    async def submit_after_pause():
        # Fallback for an entry shorter than six digits: submit once the caller
        # stops pressing keys, the way an IVR does. '#' submits immediately,
        # '*' clears.
        try:
            await asyncio.sleep(DTMF_PAUSE_S)
            await submit_dtmf()
        except asyncio.CancelledError:
            pass

    def restart_dtmf_timer():
        nonlocal dtmf_timer
        if dtmf_timer and not dtmf_timer.done():
            dtmf_timer.cancel()
        dtmf_timer = asyncio.create_task(submit_after_pause())

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                caller = msg["start"].get("customParameters", {}).get("caller") or None
                label = (msg["start"].get("callSid") or stream_sid)[-8:]
                logger.info(f"[{label}] call started, caller={caller}")
                session = VoiceAgentSession(
                    label=label,
                    audio_in=MULAW_8K,
                    audio_out=MULAW_8K_OUT,
                    toolbox=ToolBox(caller_number=caller, session_label=label),
                    on_audio=to_twilio_audio,
                    on_event=on_va_event,
                )
                await session.connect()
                register_session(label, caller, session.toolbox, session.metrics)

            elif event == "media" and session:
                await session.send_audio(base64.b64decode(msg["media"]["payload"]))

            elif event == "dtmf" and session:
                digit = msg.get("dtmf", {}).get("digit", "")
                if digit == "#":
                    if dtmf_timer and not dtmf_timer.done():
                        dtmf_timer.cancel()
                    await submit_dtmf()
                elif digit == "*":
                    dtmf_digits.clear()
                    if dtmf_timer and not dtmf_timer.done():
                        dtmf_timer.cancel()
                elif digit:
                    dtmf_digits.append(digit)
                    # the verification code is exactly six digits, so submit on
                    # the sixth; the pause timer covers anything shorter
                    if len(dtmf_digits) >= 6:
                        if dtmf_timer and not dtmf_timer.done():
                            dtmf_timer.cancel()
                        await submit_dtmf()
                    else:
                        restart_dtmf_timer()

            elif event == "stop":
                logger.info(f"[{stream_sid}] call ended")
                break

    except WebSocketDisconnect:
        pass
    finally:
        if label and label in ACTIVE:
            ACTIVE[label]["ended"] = True
        if session:
            LAST_SUMMARIES.append(session.metrics.summary())
            await session.close()


# --- Browser -----------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    return (WEB_DIR / "index.html").read_text()


@app.websocket("/ws-browser")
async def browser_ws(websocket: WebSocket):
    """Browser client: binary frames = mic audio (linear16 @16k).
    We send binary agent audio (linear16 @24k) and JSON control messages.

    Coval's simulator connects here too. Its agent config sets
    custom_headers = {"X-Coval-Simulation-Id": "{{simulation_output_id}}"},
    so a simulated call arrives carrying the id its spans have to be filed
    under; a human on this endpoint sends no such header and exports nothing.
    """
    coval_sim = websocket.headers.get("x-coval-simulation-id")
    await websocket.accept()

    async def to_browser_audio(chunk: bytes):
        await websocket.send_bytes(chunk)

    async def on_va_event(event: dict):
        etype = event.get("type")
        if etype == "UserStartedSpeaking":
            # `clear` is barge-in. `listening` is for the transcript pane: the
            # Voice Agent API emits no interim transcript, so the caller's words
            # cannot appear until the whole utterance is finalised -- on a long
            # sentence that is seconds of apparently dead screen. This is the
            # earliest honest signal that speech is being heard.
            await websocket.send_json({"type": "clear"})
            await websocket.send_json({"type": "listening"})
        elif etype == "ConversationText":
            await websocket.send_json(
                {"type": "transcript", "role": event.get("role"), "content": event.get("content")}
            )
        elif etype in ("AgentStartedSpeaking", "LatencyReport", "AgentAudioDone"):
            # AgentStartedSpeaking is NOT emitted on every Voice Agent config
            # (the Flux listen path doesn't send it), so don't hang the UI's
            # latency on it -- push on any event that means a turn just landed,
            # and fall back to wall-clock TTFA exactly as /state does.
            await send_latency()

    async def send_latency():
        m = session.metrics
        summary = m.summary()
        # Deepgram's own LatencyReport is authoritative; the wall clock is only a
        # fallback, and summary() has already dropped the turns where the server
        # injected speech, because those measure our hold phrase not the agent.
        stats, source = summary.get("dg_total_latency"), "deepgram"
        if not stats:
            stats, source = summary.get("wall_ttfa_s") or {}, "wall clock"
        value = turn_total(m.turns[-1]) if m.turns else None
        if value is None and not stats:
            return
        await websocket.send_json({
            "type": "latency",
            "total_latency": value,
            "source": source,
            "turns": len(m.turns),
            "median": stats.get("median"),
            "max": stats.get("max"),
        })

    async def on_tool_update():
        tb = session.toolbox
        await websocket.send_json({
            "type": "tools",
            "otp": tb.otp,
            "verified": tb.verified,
            "attempts_remaining": tb.otp_attempts_remaining,
        })

    session = VoiceAgentSession(
        label="browser",
        audio_in=LINEAR16_IN,
        audio_out=LINEAR16_OUT,
        toolbox=ToolBox(caller_number=None, session_label="browser"),
        on_audio=to_browser_audio,
        on_event=on_va_event,
        on_tool=on_tool_update,
        coval_simulation_id=coval_sim,
    )
    try:
        await session.connect()
        register_session("browser", None, session.toolbox, session.metrics)
        await websocket.send_json({"type": "ready"})
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                await session.send_audio(msg["bytes"])
            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "keypad":
                    await session.inject_user_text(f"[keypad] {data.get('digits', '')}")
    except WebSocketDisconnect:
        pass
    except RuntimeError:
        pass  # receive after disconnect
    finally:
        if "browser" in ACTIVE:
            ACTIVE["browser"]["ended"] = True
        LAST_SUMMARIES.append(session.metrics.summary())
        # one POST per call rather than per span, and never on the hot path
        if session.trace.active:
            await asyncio.to_thread(session.trace.flush)
        await session.close()


# --- Metrics -----------------------------------------------------------------


def _shown_listen() -> str:
    """Listen model the UI should show: the live session's, else config."""
    for rec in globals().get("ACTIVE", {}).values():
        if rec.get("ended"):
            continue
        model = getattr(rec.get("metrics"), "listen_model", None)
        if model:
            return model
    return LISTEN_MODEL


def _shown_voice() -> str:
    """Voice the sidebar reports: always the one the session actually uses.

    Derived, never pinned. This briefly carried a PANEL_TTS_LABEL override so the
    panel could read "aura-2-callista-en-pharma" while the audio was something
    else; that existed only because production could not serve the pharma-tuned
    Aura-2, and Flux TTS approval on 2026-09-01 removed the reason. A knob whose
    only power is to make the panel disagree with the audio is not worth keeping.

    The one adjustment is honest: Aura-2 served from the self-hosted pharma pool
    is a different voice from production Aura-2 under the same name, so it gets a
    `-pharma` suffix. That is read off VA_URL, not asserted.

    Display only -- Settings always carries VOICE_MODEL, which is the name the API
    and the SageMaker container both expect.
    """
    on_pharma_pool = "aura2-pharma" in VA_URL or bool(VA_BASIC_AUTH)
    if on_pharma_pool and VOICE_MODEL.startswith("aura-2-"):
        return f"{VOICE_MODEL}-pharma"
    return VOICE_MODEL


def _shown_phone() -> dict | None:
    """The demo's inbound number, for the browser client's call-in panel.

    Deployment config, not source: an unset TWILIO_FROM_NUMBER means the phone
    path is not wired up, and the panel hides itself rather than printing a
    number nobody can dial.
    """
    e164 = (TWILIO_FROM_NUMBER or "").strip()
    if not e164:
        return None
    digits = "".join(c for c in e164 if c.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    display = (f"+1 ({digits[:3]}) {digits[3:6]}\u2212{digits[6:]}"
               if len(digits) == 10 else e164)
    return {"e164": e164, "display": display}


@app.get("/capabilities")
async def capabilities():
    """Catalog, knowledge topics, and the live STT/LLM/TTS stack for the sidebar."""
    in_stock = [m["name"] for m in catalog.MEDICATIONS if m.get("in_stock")]
    out_of_stock = [m["name"] for m in catalog.MEDICATIONS if not m.get("in_stock")]
    strengths = sum(len(m.get("strengths") or []) for m in catalog.MEDICATIONS)

    featured = [n for n in ("Zepbound", "Mounjaro", "Ozempic", "Aspirin", "Atorvastatin",
                            "Lisinopril", "Levothyroxine", "Adderall XR", "Alprazolam")
                if n in in_stock]

    topics: dict[str, list[str]] = {}
    for e in knowledge.ENTRIES:
        topics.setdefault(e["topic"], []).append(e["questions"][0])

    listen = _shown_listen()
    llm = f"{THINK_PROVIDER}/{THINK_MODEL}"
    return JSONResponse({
        "medications": {
            "total": len(catalog.MEDICATIONS),
            "strengths": strengths,
            "featured": featured,
            "out_of_stock": out_of_stock,
        },
        "knowledge": [
            {"topic": t, "count": len(qs), "example": qs[0]}
            for t, qs in sorted(topics.items())
        ],
        "account": {
            "prescriptions": [p.get("medication") for p in knowledge.PRESCRIPTIONS],
            "orders": len(knowledge.ORDERS),
        },
        "stack": {
            "stt": listen,
            "llm": llm,
            "tts": _shown_voice(),
        },
        "phone": _shown_phone(),
    })


@app.get("/health")
async def health():
    """Fly health check. Deliberately unauthenticated and dependency-free."""
    return {"ok": True}


@app.get("/latency", dependencies=[Depends(require_console)])
async def latency():
    return JSONResponse(LAST_SUMMARIES[-20:])


# Operator console: live OTP + latency. The Twilio number is not A2P-10DLC.

ACTIVE: dict[str, dict] = {}


def register_session(label: str, caller: str | None, toolbox: ToolBox, metrics) -> None:
    ACTIVE[label] = {"caller": caller, "toolbox": toolbox, "metrics": metrics,
                     "started": time.time(), "ended": False}
    # keep the console to the last handful of calls
    for k in list(ACTIVE)[:-5]:
        del ACTIVE[k]


@app.get("/state", dependencies=[Depends(require_console)])
async def state():
    out = []
    for label, s in ACTIVE.items():
        tb, m = s["toolbox"], s["metrics"]
        out.append({
            "label": label,
            "caller": s["caller"],
            "ended": s["ended"],
            "listen_model": m.listen_model,
            "otp": tb.otp,
            "verified": tb.verified,
            "attempts_remaining": tb.otp_attempts_remaining,
            "last_latency_s": m.last_latency(),
            "turns": len(m.turns),
            "transcript_tail": m.transcript[-6:],
        })
    return JSONResponse(out)


@app.get("/console", response_class=HTMLResponse, dependencies=[Depends(require_console)])
async def console():
    return (WEB_DIR / "console.html").read_text()


# --- data browser ------------------------------------------------------------
# Deliberately unauthenticated, like / and /capabilities: the point is that
# anyone watching the demo can see the data the agent is answering from. Nothing
# here is real -- the "account" is one fictional demo profile.


@app.get("/data", response_class=HTMLResponse)
async def data_page():
    return (WEB_DIR / "data.html").read_text()


def _med_row(m: dict) -> dict:
    sched = m.get("controlled_schedule")
    return {
        "name": m["name"],
        "generic": m.get("generic") or "—",
        "brand": m.get("brand_equivalent") or "—",
        "category": m.get("category") or "—",
        "form": m.get("form") or "—",
        "strengths": ", ".join(catalog._strengths(m)) or "—",
        "supplies": ", ".join(s.replace("_day", " day")
                              for s in catalog._supplies(m)) or "—",
        "stock": "in stock" if m.get("in_stock") else "out of stock",
        "schedule": sched or "—",
    }


@app.get("/data-api")
async def data_api():
    """Every row the agent can reach, in the shape the browser table renders.

    Pricing is summarised rather than shipped whole: catalog.json carries 1,349
    strength/supply combinations and sending them all would be a megabyte of
    JSON for a page nobody scrolls that far down.
    """
    meds = [_med_row(m) for m in catalog.MEDICATIONS]
    kb = [{"topic": e["topic"], "question": e["questions"][0],
           "also_asked": len(e["questions"]) - 1, "answer": e["answer"]}
          for e in knowledge.ENTRIES]
    rx = [{"medication": p["medication"], "strength": p["strength"],
           "day_supply": f"{p['day_supply']} day", "status": p["status"],
           "refills": p["refills_remaining"],
           "next_refill": knowledge._when(p.get("next_refill_in_days")),
           "prescriber": p.get("prescriber") or "—"}
          for p in knowledge.PRESCRIPTIONS]
    orders = [{"status": o["status"],
               "items": ", ".join(o["items"]),
               "placed": knowledge._when(-o["placed_days_ago"]),
               "arrives": knowledge._when(o["arrives_in_days"]),
               "total": f"${o['total']:.2f}",
               "tracking": o.get("tracking") or "—"}
              for o in knowledge.ORDERS]

    return JSONResponse({
        "collections": [
            {"id": "medications", "label": "medications", "icon": "pill",
             "source": "catalog.json",
             "columns": ["name", "generic", "category", "form", "strengths",
                         "supplies", "stock", "schedule"],
             "key": "name", "rows": meds},
            {"id": "knowledge", "label": "knowledge base", "icon": "book",
             "source": "knowledge-base.json",
             "columns": ["topic", "question", "also_asked", "answer"],
             "key": "topic", "rows": kb},
            {"id": "prescriptions", "label": "prescriptions", "icon": "flask",
             "source": "account.json",
             "columns": ["medication", "strength", "day_supply", "status",
                         "refills", "next_refill", "prescriber"],
             "key": "medication", "rows": rx},
            {"id": "orders", "label": "orders", "icon": "box",
             "source": "account.json",
             "columns": ["status", "items", "placed", "arrives", "total", "tracking"],
             "key": "status", "rows": orders},
        ],
        "profile": knowledge.PROFILE,
        "totals": {
            "strengths": sum(len(catalog._strengths(m)) for m in catalog.MEDICATIONS),
            "priced": sum(len(row) for m in catalog.MEDICATIONS
                          for row in (m.get("pricing") or {}).values()),
        },
    })
