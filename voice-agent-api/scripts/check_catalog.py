"""Validate catalog.json. Run after any edit to the knowledge base.

Catches the failure modes that only show up live, mid-call: a strength with no
price, an alias that resolves to the wrong drug, or a drifted transcript anchor.

    uv run python scripts/check_catalog.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import catalog as c  # noqa: E402
from server import knowledge as kb  # noqa: E402
from server import prompt  # noqa: E402

# The reference call's numbers, quoted back at us. These must never drift.
ANCHORS = [
    (("Zepbound", "12.5 mg", 90), "dispensed_day_supply", 84),
    (("Zepbound", "12.5 mg", 90), "insurance_avg_price", 50.0),
    (("Zepbound", "12.5 mg", 90), "as_low_as_with_coupon", 25.0),
    (("Zepbound", "12.5 mg", 90), "volume", "6 mL"),
    (("Zepbound", "12.5 mg", 84), "insurance_avg_price", 50.0),
]

failures: list[str] = []
warnings: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def check_shape() -> None:
    for m in c.MEDICATIONS:
        name = m.get("name", "<unnamed>")
        for field in ("name", "generic", "category", "form"):
            if not m.get(field):
                fail(f"{name}: missing {field}")

        strengths = c._strengths(m)
        if not strengths:
            fail(f"{name}: no strengths on file")
        keys = set(m.get("pricing", {}))

        for s in strengths:
            k = c._strength_key(s)
            if not k:
                fail(f"{name}: strength {s!r} has no parseable number")
            elif k not in keys:
                fail(f"{name}: strength {s!r} has no pricing row (looked for {k!r})")

        for k in keys - {c._strength_key(s) for s in strengths}:
            fail(f"{name}: pricing row {k!r} has no matching strength")

        for k, row in m.get("pricing", {}).items():
            if not row:
                fail(f"{name} {k}: empty pricing row")
            for supply, entry in row.items():
                if not supply.split("_")[0].isdigit():
                    fail(f"{name} {k}: supply key {supply!r} does not start with a number")
                if entry.get("cash") is None and entry.get("insurance_avg") is None:
                    fail(f"{name} {k} {supply}: neither a cash nor an insurance price")
                if entry.get("insurance_avg") == 0:
                    fail(f"{name} {k} {supply}: insurance_avg is 0, which the agent would "
                         "speak as 'zero dollars' -- use null for 'coverage not established'")


def check_pricing_roundtrip() -> None:
    """Every strength must price cleanly at every supply the drug carries."""
    for m in c.MEDICATIONS:
        for s in c._strengths(m):
            if not c.lookup(m["name"]).get("match"):
                continue
            for supply_key in c._supplies(m):
                days = int(supply_key.split("_")[0])
                res = c.price(m["name"], s, days)
                if res.get("error"):
                    fail(f"{m['name']} {s} {days}-day: {res['error']}")
                elif res.get("cash_price") is None and res.get("insurance_avg_price") is None:
                    fail(f"{m['name']} {s} {days}-day: priced to nothing")
            # a bare number, the way a caller says it
            bare = s.split()[0]
            if c.price(m["name"], bare, int(c._supplies(m)[0].split("_")[0])).get("error"):
                warnings.append(f"{m['name']}: bare strength {bare!r} is ambiguous, "
                                "the agent must say the unit")


def check_anchors() -> None:
    for (name, strength, days), field, expected in ANCHORS:
        got = c.price(name, strength, days).get(field)
        if got != expected:
            fail(f"transcript anchor {name} {strength} {days}-day {field}: "
                 f"expected {expected!r}, got {got!r}")


def check_prompt_surface() -> None:
    if len(c.keyterms()) > 100:
        warnings.append(f"{len(c.keyterms())} STT keyterms -- over Deepgram's budget")
    for m in c.MEDICATIONS:
        if not m.get("pronunciation"):
            warnings.append(f"{m['name']}: no pronunciation on file")


def check_prompt_budget() -> None:
    """The prompt grows with the catalog, so the limit has to be a test, not a habit.

    Over PROMPT_LIMIT the API truncates and only warns, so the failure is a call
    that sounds fine while missing the tail of its own instructions. That is worth
    catching here rather than in front of the customer.
    """
    size = len(prompt.build())
    if size > prompt.PROMPT_LIMIT:
        fail(f"prompt is {size:,} chars, over the {prompt.PROMPT_LIMIT:,} limit")
    elif size > prompt.PROMPT_LIMIT * 0.95:
        warnings.append(f"prompt is {size:,} chars, within 5% of the limit")
    print(f"prompt: {size:,} chars (limit {prompt.PROMPT_LIMIT:,})")


MUST_ANSWER = [
    "how much is delivery", "when will my order arrive", "where is my order",
    "how do I get a refill", "what is auto refill", "how do I transfer my prescription",
    "do you take blue cross", "what is a prior authorization", "why did my price go up",
    "what is rxpass", "what is pillpack", "how much does pillpack cost",
    "can I use my hsa", "how do I get a receipt", "can I talk to a pharmacist",
    "are you open on weekends", "is my information private", "I got the wrong medication",
    "can I cancel my order", "I do not have insurance", "can I refill early",
    "how do you ship insulin", "can you transfer my adderall", "what is your phone number",
]


def check_handling() -> None:
    """The knowledge base makes per-medication promises; the catalog has to back them.

    - delivery-signature: "Controlled medications and refrigerated items need
      someone to receive them"  -> signature_required must equal cold or controlled
    - delivery-cold: refrigerated items ship cold  -> only injectables are cold
    - controlled-refill: "Schedule two prescriptions have no refills" -> CII
      must carry refills_allowed 0, max_day_supply 30, transferable false
    - transfer-controlled: only Schedule II is non-transferable
    """
    INJECTABLE = ("injector", "syringe", "vial", "auto-injector", "prefilled pen")
    for m in c.MEDICATIONS:
        name, cold = m["name"], bool(m.get("refrigerated"))
        sched = m.get("controlled_schedule")

        if cold and not any(w in m["form"].lower() for w in INJECTABLE):
            fail(f"{name}: marked refrigerated but the form is {m['form']!r}")

        derived = c.lookup(name).get("signature_required")
        if derived != (cold or bool(sched)):
            fail(f"{name}: signature_required {derived} does not follow from "
                 f"refrigerated={cold} controlled={bool(sched)}")

        if sched == "CII":
            if m.get("refills_allowed") != 0:
                fail(f"{name}: Schedule II must have refills_allowed 0, got {m.get('refills_allowed')!r}")
            if m.get("max_day_supply") != 30:
                fail(f"{name}: Schedule II must cap at a 30 day supply, got {m.get('max_day_supply')!r}")
            if m.get("transferable") is not False:
                fail(f"{name}: Schedule II must be transferable=false")
        elif sched:
            if m.get("transferable") is False:
                fail(f"{name}: {sched} is transferable by law, but transferable=false")

    for key in ("cold_chain", "signature", "po_box"):
        if key not in c.POLICIES:
            fail(f"policies.{key} missing; the knowledge base promises it")


def check_internal_notes() -> None:
    """`note` fields reach the model verbatim, so build scaffolding in one can be
    read aloud on a live call. catalog.py withholds these at runtime; fail here so
    they get cleaned at the source instead of relying on the filter."""
    for m in c.MEDICATIONS:
        for label, note in (("note", m.get("note", "")),):
            if note and c.is_internal_note(note):
                fail(f"{m['name']}: {label} looks like internal scaffolding, "
                     f"not caller-facing copy: {note!r}")
        for strength, supplies in (m.get("pricing") or {}).items():
            for supply, entry in (supplies or {}).items():
                note = (entry or {}).get("note", "")
                if note and c.is_internal_note(note):
                    fail(f"{m['name']} {strength} {supply}: note looks internal: {note!r}")


def check_knowledge_base() -> None:
    ids = [e["id"] for e in kb.ENTRIES]
    for dup in {i for i in ids if ids.count(i) > 1}:
        fail(f"knowledge base: duplicate id {dup!r}")
    for e in kb.ENTRIES:
        if not e.get("questions"):
            fail(f"knowledge base {e['id']}: no question phrasings")
        if not e.get("answer"):
            fail(f"knowledge base {e['id']}: no answer")
        for ch in "*_#`|$%":
            if ch in e.get("answer", ""):
                fail(f"knowledge base {e['id']}: answer contains {ch!r}, which gets read aloud")

    # the whole point of this knowledge base: these must not dead-end
    for q in MUST_ANSWER:
        res = kb.search(q)
        if res["confidence"] == "low":
            fail(f"knowledge base: {q!r} finds no confident answer")

    # and junk must NOT come back confident
    for q in ("who won the game last night", "what is the weather", "tell me a joke"):
        if kb.search(q)["confidence"] == "high":
            warnings.append(f"knowledge base: off-topic {q!r} scored high confidence")


def check_account() -> None:
    for topic in ("orders", "prescriptions", "insurance", "profile"):
        res = kb.account_info(topic)
        if not res or len(res) < 2:
            fail(f"account: topic {topic!r} returned nothing useful")
    for o in kb.account_info("orders")["orders"]:
        if not o.get("status") or not o.get("items"):
            fail(f"account: order missing status or items: {o}")
    for rx in kb.account_info("prescriptions")["prescriptions"]:
        if c.lookup(rx["medication"]).get("match") != rx["medication"]:
            fail(f"account: prescription {rx['medication']!r} is not in the catalog")
        if c.price(rx["medication"], rx["strength"], rx["day_supply"]).get("error"):
            fail(f"account: {rx['medication']} {rx['strength']} does not price")


for check in (check_shape, check_pricing_roundtrip, check_anchors,
              check_prompt_surface, check_prompt_budget, check_knowledge_base, check_account,
              check_internal_notes, check_handling):
    check()

meds, strengths = len(c.MEDICATIONS), sum(len(c._strengths(m)) for m in c.MEDICATIONS)
cells = sum(len(r) for m in c.MEDICATIONS for r in m.get("pricing", {}).values())
print(f"{meds} medications, {strengths} strengths, {cells} priced strength/supply combinations")
print(f"{len(c.keyterms())} STT keyterms (of {len(c.all_names())} names), "
      f"{len(c.pronunciation_lines(keyterm_only=False))} pronunciations")
print(f"{len(kb.ENTRIES)} knowledge-base entries across {len(kb.topics())} topics; "
      f"{len(kb.PRESCRIPTIONS)} prescriptions and {len(kb.ORDERS)} orders on the demo account")


for w in warnings:
    print(f"  warn  {w}")
for f in failures:
    print(f"  FAIL  {f}")

if failures:
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1)
print(f"\nOK{f' ({len(warnings)} warning(s))' if warnings else ''}")
