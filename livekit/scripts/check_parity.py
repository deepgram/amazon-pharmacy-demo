"""Assert the two demo paths are still running the same agent.

    uv run python scripts/check_parity.py [--path-a ../voice-agent-api]

The pairing only means something if the ONLY difference between the two repos is
orchestration. Everything the model sees -- the prompt, the tool schemas, the
knowledge bases -- and everything the caller hears that the model did not choose
-- the scripted lines, the disclaimers, the phrase pools -- has to be identical.
Both are edited by hand in two places, so both drift.

The failure this is really for is the quiet one. A reworded tool description or a
retuned hold phrase does not break anything; it just moves Path B's behaviour,
and the next latency or repetition comparison attributes the move to LiveKit.
That is a wrong conclusion presented as a measurement, which is worse than a
crash.

What it does NOT check, because these are legitimately different:

    listen/think/speak wiring   different plugins, different classes
    turn-taking                 LiveKit stacks its own endpointing on the STT's
    latency accounting          no LatencyReport here; see agent/metrics.py
    the speech transport        InjectAgentMessage vs AgentSession.say()

Exit status is 0 when the shared surface matches, 1 when it does not.
"""

import argparse
import difflib
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load(repo: Path, package: str):
    """Import `package` out of `repo` as a top-level module."""
    if not repo.exists():
        sys.exit(f"path A repo not found: {repo}\nPass --path-a with its location.")
    sys.path.insert(0, str(repo))
    mods = {}
    for name in ("catalog", "knowledge", "prompt", "config", "tools", "spoken"):
        mods[name] = importlib.import_module(f"{package}.{name}")
    sys.path.pop(0)
    return mods


failures: list[str] = []
checks = 0


def same(what: str, a, b, *, show_diff: bool = False) -> None:
    global checks
    checks += 1
    if a == b:
        print(f"  ok    {what}")
        return
    failures.append(what)
    print(f"  DIFF  {what}")
    if show_diff and isinstance(a, str) and isinstance(b, str):
        for line in list(difflib.unified_diff(
            a.splitlines(), b.splitlines(),
            fromfile="path-a", tofile="path-b", lineterm="", n=1,
        ))[:40]:
            print(f"        {line}")
    else:
        print(f"        path A: {str(a)[:160]}")
        print(f"        path B: {str(b)[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-a", default=str(HERE.parent / "voice-agent-api"),
                    help="the Voice Agent API demo (../voice-agent-api)")
    args = ap.parse_args()

    repo_a = Path(args.path_a).expanduser().resolve()
    # Sibling checkout is the common case; fall back to it before giving up.
    if not repo_a.exists():
        sibling = HERE.parent / "voice-agent-api"
        if sibling.exists():
            repo_a = sibling

    print(f"path A: {repo_a}")
    print(f"path B: {HERE}\n")

    a = load(repo_a, "server")
    b = load(HERE, "agent")

    print("the prompt the model is given")
    same("prompt.build()", a["prompt"].build(), b["prompt"].build(), show_diff=True)
    same("prompt.GREETING", a["prompt"].GREETING, b["prompt"].GREETING)
    same("prompt.LINE_NUMBER_CHECK", a["prompt"].LINE_NUMBER_CHECK,
         b["prompt"].LINE_NUMBER_CHECK)
    same("prompt.PROMPT_LIMIT", a["prompt"].PROMPT_LIMIT, b["prompt"].PROMPT_LIMIT)

    print("\nthe tools the model is given")
    same("tool names", [d["name"] for d in a["tools"].DEFINITIONS],
         [d["name"] for d in b["tools"].DEFINITIONS])
    by_a = {d["name"]: d for d in a["tools"].DEFINITIONS}
    by_b = {d["name"]: d for d in b["tools"].DEFINITIONS}
    for name in sorted(set(by_a) | set(by_b)):
        same(f"schema {name}",
             json.dumps(by_a.get(name), sort_keys=True),
             json.dumps(by_b.get(name), sort_keys=True))
    same("gated tools", sorted(a["tools"].ToolBox.GATED_TOOLS),
         sorted(b["tools"].ToolBox.GATED_TOOLS))
    for rule in ("REPLY_RULE", "END_REPLY_RULE", "TRANSFER_REPLY_RULE"):
        same(f"tools.{rule}", getattr(a["tools"], rule), getattr(b["tools"], rule))

    print("\nthe knowledge bases")
    same("medication count", len(a["catalog"].MEDICATIONS), len(b["catalog"].MEDICATIONS))
    same("catalog payload", json.dumps(a["catalog"].MEDICATIONS, sort_keys=True),
         json.dumps(b["catalog"].MEDICATIONS, sort_keys=True))
    same("catalog policies", json.dumps(a["catalog"].POLICIES, sort_keys=True),
         json.dumps(b["catalog"].POLICIES, sort_keys=True))
    same("STT keyterms", a["catalog"].keyterms(), b["catalog"].keyterms())
    same("knowledge-base entries", json.dumps(a["knowledge"].ENTRIES, sort_keys=True),
         json.dumps(b["knowledge"].ENTRIES, sort_keys=True))
    same("kb topics", a["knowledge"].topics(), b["knowledge"].topics())
    same("demo account prescriptions",
         json.dumps(a["knowledge"].PRESCRIPTIONS, sort_keys=True),
         json.dumps(b["knowledge"].PRESCRIPTIONS, sort_keys=True))
    same("demo account orders", json.dumps(a["knowledge"].ORDERS, sort_keys=True),
         json.dumps(b["knowledge"].ORDERS, sort_keys=True))

    print("\nlines the caller hears that the model did not choose")
    for const in (
        "LINE_CLOSING", "LINE_CLOSING_MARKER", "LINE_LOCKOUT", "LINE_TRANSFER",
        "PRICE_DISCLAIMER", "SUPPLY_CONVERSION_TEMPLATE", "SILENCE_GOODBYE",
        "HOLD_PHRASE_2",
    ):
        same(f"config.{const}", getattr(a["config"], const), getattr(b["config"], const))
    for const in (
        "HOLD_PHRASES", "SILENCE_NUDGES", "SILENCE_NUDGES_2", "SILENCE_NUDGES_OTP",
    ):
        same(f"config.{const}", getattr(a["config"], const), getattr(b["config"], const))
    same("config.SCRIPTED_TOOL_LINES",
         json.dumps(a["config"].SCRIPTED_TOOL_LINES, sort_keys=True),
         json.dumps(b["config"].SCRIPTED_TOOL_LINES, sort_keys=True))
    same("config.SERVER_SPOKEN_LINES", sorted(a["config"].SERVER_SPOKEN_LINES),
         sorted(b["config"].SERVER_SPOKEN_LINES))

    print("\nthresholds and guards")
    for const in (
        # models: the whole point of the pairing is that these agree
        "LISTEN_MODEL", "THINK_PROVIDER", "THINK_MODEL", "VOICE_MODEL",
        "THINK_TEMPERATURE", "REASONING_MODE",
        # turn-taking that IS shared (the STT's own endpointing, not LiveKit's)
        "LISTEN_ENDPOINTING_MS", "LISTEN_KEYTERMS", "EOT_THRESHOLD",
        "EOT_TIMEOUT_MS", "EAGER_EOT_THRESHOLD", "SPLIT_TURN_GRACE_S",
        # pacing and guards
        "HOLD_AFTER_S", "SILENCE_AFTER_S", "SCRIPTED_LINE_COOLDOWN_S",
        "DISCLAIMER_GRACE_S", "DISCLAIMER_MAX_WAIT_S", "OTP_SILENCE_FACTOR",
        "MAX_LINE_REPEATS", "STALL_LIMIT", "MAX_TURNS", "MAX_SILENCE_NUDGES",
        "EMERGENCY_END_S", "GOODBYE_MAX_CHARS", "QUEUE_POSITION", "QUEUE_WAIT_MIN",
        "DEMO_OTP",
    ):
        same(f"config.{const}", getattr(a["config"], const, "<missing>"),
             getattr(b["config"], const, "<missing>"))
    for const in (
        "GOODBYE_MARKERS", "STILL_WORKING_MARKERS", "OFFER_AGENT_MARKERS",
        "EMERGENCY_EVIDENCE", "EMERGENCY_DENIALS", "EMERGENCY_MARKERS",
        "DISCLAIMER_ESTIMATE_MARKERS", "DISCLAIMER_VARIANCE_MARKERS",
        "DISCLAIMER_WHOLE_MARKERS",
    ):
        same(f"config.{const}", getattr(a["config"], const, "<missing>"),
             getattr(b["config"], const, "<missing>"))

    print("\nbehaviour on the same input")
    names = ["Zepbound", "zipbound", "sells that bound", "Lipitor", "Ozempic",
             "semaglutide", "Wegovy", "Adderall XR", "Levothyroxine"]
    same("catalog.lookup across sample names",
         json.dumps([a["catalog"].lookup(n) for n in names], sort_keys=True),
         json.dumps([b["catalog"].lookup(n) for n in names], sort_keys=True))
    prices = [("Zepbound", "12.5 mg", 90), ("Zepbound", "12.5 mg", 30),
              ("Levothyroxine", "100 mcg", 90), ("Wegovy", "2.4 mg", 30)]
    same("catalog.price across sample requests",
         json.dumps([a["catalog"].price(*p) for p in prices], sort_keys=True),
         json.dumps([b["catalog"].price(*p) for p in prices], sort_keys=True))
    questions = ["how fast is delivery", "can you transfer my prescription",
                 "what is RxPass", "do you take my insurance", "what is PillPack",
                 "what is the weather"]
    same("knowledge.search across sample questions",
         json.dumps([a["knowledge"].search(q) for q in questions], sort_keys=True),
         json.dumps([b["knowledge"].search(q) for q in questions], sort_keys=True))
    spoken_cases = [50.0, 12.5, 84, 90, 0.5, 349.0]
    same("spoken.money", [a["spoken"].money(v) for v in spoken_cases],
         [b["spoken"].money(v) for v in spoken_cases])
    same("spoken.integer", [a["spoken"].integer(int(v)) for v in (84, 90, 30, 6, 15)],
         [b["spoken"].integer(int(v)) for v in (84, 90, 30, 6, 15)])

    print()
    if failures:
        print(f"{checks - len(failures)}/{checks} checks match")
        print("\nDRIFT between the two demo paths:")
        for f in failures:
            print(f"  - {f}")
        print("\nFix the drift before quoting any A/B number: a comparison across "
              "these two repos is only about orchestration when this passes.")
        sys.exit(1)
    print(f"{checks}/{checks} checks match -- the two paths are running the same agent")


if __name__ == "__main__":
    main()
