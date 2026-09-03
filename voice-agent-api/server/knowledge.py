"""Pharmacy operations knowledge base + the demo caller's account.

Lookups never return empty: a low-confidence answer beats "I can't help."
"""

import json
import re
from pathlib import Path

from . import catalog, spoken
from .catalog import BE_BRIEF, FILL_IS_AVAILABLE

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "knowledge-base.json") as f:
    _KB = json.load(f)
with open(ROOT / "account.json") as f:
    _ACCOUNT = json.load(f)

ENTRIES = _KB["entries"]
PROFILE = _ACCOUNT["profile"]
PRESCRIPTIONS = _ACCOUNT["prescriptions"]
ORDERS = _ACCOUNT["orders"]

# words that carry no signal in "how do I get a refill" style questions
_STOP = {
    "a", "an", "the", "is", "are", "was", "do", "does", "did", "can", "could", "will", "would",
    "i", "my", "me", "you", "your", "we", "it", "its", "this", "that", "there", "of", "for",
    "to", "in", "on", "at", "and", "or", "but", "if", "how", "what", "when", "where", "why",
    "who", "get", "got", "have", "has", "am", "be", "been", "with", "about", "any", "some",
    "please", "just", "like", "want", "need", "tell", "know", "say", "said", "so", "up",
}


# Brand names the knowledge base stores closed up but nobody says that way. The
# tokenizer splits on non-alphanumerics, so a caller asking about RxPass produced
# the tokens {rx, pass} against a stored {rxpass} -- no overlap at all, so the
# query scored `low` and returned the delivery entry instead of the seven RxPass
# and PillPack entries that were sitting right there. The system prompt makes it
# worse by instructing the agent to say "pill pack" as two words, so both sides of
# the conversation hit it. Collapse the spoken form back before tokenising.
_COMPOUNDS = (
    (re.compile(r"\brx[\s-]+pass\b"), "rxpass"),
    (re.compile(r"\bpill[\s-]+pack\b"), "pillpack"),
    (re.compile(r"\bauto[\s-]+refill\b"), "autorefill"),
    (re.compile(r"\bco[\s-]+pay\b"), "copay"),
    (re.compile(r"\bmail[\s-]+order\b"), "mailorder"),
)


def _fold_compounds(text: str) -> str:
    low = (text or "").lower()
    for pattern, joined in _COMPOUNDS:
        low = pattern.sub(joined, low)
    return low


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _fold_compounds(text))
    return {w for w in words if w not in _STOP and len(w) > 1}


_INDEX = [
    (e, _tokens(" ".join(e["questions"])), _tokens(e["topic"]), _tokens(e["answer"]),
     [_tokens(p) for p in e["questions"]])
    for e in ENTRIES
]


# Calibrated against a held-out set of real and off-topic questions.
# density = match concentration; evidence = absolute overlap. Re-run after edits.
_LEN_K = 3.0
_EVIDENCE_FULL = 9.0
_W_EVIDENCE = 1.5
_W_PHRASE = 1.0

CONF_HIGH = 2.3
CONF_MEDIUM = 2.2


def _score(query: str, qt: set, tt: set, at: set, qsets: list[set]) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    # question phrasings carry the most signal, the answer body the least
    weighted = 3.0 * len(q & qt) + 1.5 * len(q & tt) + 0.5 * len(q & at)
    if not weighted:
        return 0.0
    density = weighted / (_LEN_K + len(q))
    evidence = min(1.0, weighted / _EVIDENCE_FULL)
    # How much of a stored phrasing the caller actually covered. Two length
    # biases had to go: the original difflib.SequenceMatcher compared raw
    # characters, so a 90-char spoken turn scored ~0.25 against a 30-char
    # canonical phrasing however well the words lined up; and an overlap
    # coefficient (dividing by min) let a single word saturate at 1.0. Dividing
    # by the phrasing length is robust to how much the caller said while still
    # requiring them to cover most of a phrasing to score well.
    phrase = max((len(q & ps) / len(ps) for ps in qsets if ps), default=0.0)
    return density + _W_EVIDENCE * evidence + _W_PHRASE * phrase


def search(query: str, limit: int = 3) -> dict:
    """Best matching knowledge-base entries. Always returns at least one, with a
    confidence the agent can use to decide how firmly to state the answer."""
    ranked = sorted(
        ((_score(query, qt, tt, at, qs), e) for e, qt, tt, at, qs in _INDEX),
        key=lambda pair: pair[0],
        reverse=True,
    )
    top = [(s, e) for s, e in ranked[:limit] if s > 0]
    if not top:
        top = ranked[:1]

    best = top[0][0]
    confidence = "high" if best >= CONF_HIGH else "medium" if best >= CONF_MEDIUM else "low"
    out = {
        "confidence": confidence,
        "answers": [
            {k: v for k, v in
             {"topic": e["topic"], "answer": e["answer"], "use_tool": e.get("use_tool")}.items()
             if v is not None}
            for _, e in top
        ],
    }
    if confidence == "low":
        out["guidance"] = (
            "Nothing on file matches this closely. Lead with what you do know, then offer a "
            "pharmacy representative or a pharmacist. Do not tell the caller you cannot help, "
            "and do not preface it -- no 'that's a great question', no 'let me check on that'. "
            "The hedge is the line most often said twice, so say it once and move on."
        )
    return out


# Rides on every account result, next to the data it governs, for the same
# reason the catalog's rules do: the prompt-only version of each of these held on
# most calls and not all of them.
SAY_IT_AS_WRITTEN = (
    "Read every _spoken field aloud exactly as written: no digits, no dollar "
    "signs, no decimals."
)
ONLY_THIS_CALLER = (
    "This is the verified caller's own account and nobody else's. If they ask "
    "about a spouse, parent, child or anyone else, do not look it up, do not "
    "take that person's phone number, and do not send or accept a code on their "
    "behalf. Say that person has to call us themselves."
)


def _when(days: int | None) -> str:
    """Relative day offset as spoken English, so the demo account never goes stale."""
    if days is None:
        return "not scheduled yet"
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if days == -1:
        return "yesterday"
    if days < 0:
        return f"{abs(days)} days ago"
    if days == 7:
        return "a week from today"
    return f"in {days} days"


def _spoken_list(names: list[str]) -> str:
    """['a','b','c'] -> 'a, b and c'. Spoken lists do not have Oxford commas."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def orders_summary(rows: list[dict]) -> str:
    """The order list as a sentence, for the same reason prescriptions has one.

    Asked "what is the status of my current medications?", the agent called this
    with topic=orders and read the whole table back -- five bulleted lines, each
    with status, arrival and price. The anti-list guidance was on the prescriptions
    payload only; orders was left as a raw table, and a table invites a table.
    """
    if not rows:
        return "You don't have any orders with us right now."

    def med(item: str) -> str:
        """"Aspirin 81 milligrams, 30 day supply" -> "Aspirin". A summary names the
        medication; the strength and supply belong to the follow-up about one of
        them, not to a sentence listing four."""
        return re.split(r"\s+\d", item, maxsplit=1)[0].strip().rstrip(",")

    filling = [med(i) for r in rows if r["status"] == "being filled" for i in r["items"]]
    moving = [med(i) for r in rows if r["status"] in ("shipped", "processing", "out for delivery")
              for i in r["items"]]
    done = [med(i) for r in rows if r["status"] == "delivered" for i in r["items"]]
    bits = []
    if filling:
        bits.append(f"{_spoken_list(filling)} {'is' if len(filling) == 1 else 'are'} being filled now")
    if moving:
        bits.append(f"{_spoken_list(moving)} {'is' if len(moving) == 1 else 'are'} on the way")
    if done:
        bits.append(f"{_spoken_list(done)} already arrived")
    if not bits:
        return f"You have {spoken.integer(len(rows))} orders on file."
    joined = _spoken_list(bits)
    return joined[0].upper() + joined[1:] + "."


def prescriptions_summary(rows: list[dict]) -> str:
    """One sentence a person would actually say, instead of a table to read out.

    Handed five records of eleven fields each, the model read all five aloud --
    name, strength, day supply, status and refills, one after another, with the
    list markers "2." and "3." going to text-to-speech as their own turns. That is
    about a minute of recitation for a question that wanted a sentence.

    So the payload now leads with the sentence. The per-record fields stay for the
    follow-up about one medication; they are just no longer the first thing.
    """
    if not rows:
        return "You don't have any prescriptions on file with us."
    names = [r["medication"] for r in rows]
    out = [f"You have {spoken.integer(len(rows))} on file: {_spoken_list(names)}."]

    ready = [r["medication"] for r in rows if r["status"] == "ready to fill"]
    waiting = [r["medication"] for r in rows if r["status"] == "awaiting prescriber"]
    filling = [r["medication"] for r in rows if r["status"] == "being filled"]
    tail = []
    if filling:
        tail.append(f"{_spoken_list(filling)} {'is' if len(filling) == 1 else 'are'} being filled now")
    if ready:
        tail.append(f"{_spoken_list(ready)} {'is' if len(ready) == 1 else 'are'} ready to fill")
    if waiting:
        tail.append(f"{_spoken_list(waiting)} {'is' if len(waiting) == 1 else 'are'} waiting on your prescriber")
    if tail:
        # Not .capitalize(): it lowercases the rest of the string, which turned
        # "...and Lisinopril is waiting" into "...and lisinopril is waiting".
        joined = _spoken_list(tail)
        out.append(joined[0].upper() + joined[1:] + ".")
    return " ".join(out)


def refillable_summary(names: list[str], any_filled: bool) -> str:
    """The answer to "what can I refill?" as a sentence, not a list to assemble.

    Asked "do I have any other medications to refill?", the agent read all five
    prescriptions back and only produced the refillable ones a turn later -- the
    data was right, the answer was to a different question. Giving that question
    its own spoken line stops the model reaching for the general summary.
    """
    if names:
        lead = "You can refill" if not any_filled else "The others you can refill are"
        return f"{lead} {_spoken_list(names)}."
    if any_filled:
        # Deliberately not "that was the only one" -- true after one fill, wrong
        # after four, and this line cannot see how many there were.
        return ("Nothing else is ready to refill. What's left is either already "
                "being filled or waiting on your prescriber.")
    return ("Nothing is ready to refill right now -- everything is either waiting on "
            "your prescriber or not due yet.")


def fill_prescription(name: str) -> dict:
    """Fill a prescription already on this caller's account.

    The agent had no way to do this, and a caller whose Zepbound was sitting at
    "ready to fill" asked four times -- "just fill my order" -- and was told "I
    can't place the order directly" every time. The missing capability is also
    what produced the surprise transfer: with nothing to call, the model reached
    for the only tool that moves a call forward.

    Nothing here invents a prescription. It fills one that is already on file,
    which is exactly what the account payload already says is ready.
    """
    want = re.sub(r"[^a-z0-9]+", "", (name or "").lower())
    if not want:
        return {"filled": False, "error": "no medication named",
                "guidance": "Ask which medication they want filled."}

    match = next((p for p in PRESCRIPTIONS
                  if want in re.sub(r"[^a-z0-9]+", "", p["medication"].lower())), None)
    if match is None:
        return {
            "filled": False, "error": "not on file", "heard": name,
            "on_file": [p["medication"] for p in PRESCRIPTIONS],
            "guidance": [
                f"There is no prescription for {name} on this account. Say so, list what "
                "IS on file, and offer to transfer a prescription from another pharmacy "
                "or have their prescriber send one. Do not say you are unable to place "
                "orders -- that is not the reason.",
                ONLY_THIS_CALLER,
            ],
        }

    if match["status"] == "awaiting prescriber" or match["refills_remaining"] <= 0:
        return {
            "filled": False, "error": "no refills remaining",
            "medication": match["medication"], "status": match["status"],
            "guidance": [
                f"{match['medication']} is out of refills and we have already asked their "
                "prescriber for a new one. Say that, and that we will fill it as soon as "
                "it arrives. Do not say you are unable to place orders.",
                BE_BRIEF,
            ],
        }

    # What this fill costs, so "fill it and apply any coupons" gets an answer rather
    # than silence. Priced at the strength and day supply actually on file, and
    # carried whether or not a coupon exists -- a caller who asks about one is owed
    # "there isn't one for this" just as much as "yes, applied".
    money: dict = {}
    try:
        priced = catalog.price(match["medication"], match["strength"], match["day_supply"])
    except Exception:
        priced = {}
    if priced.get("match") and "error" not in priced:
        for k in ("insurance_avg_price_spoken", "cash_price_spoken",
                  "coupon_savings_up_to_spoken", "as_low_as_with_coupon_spoken"):
            if priced.get(k):
                money[k] = priced[k]
        money["coupon_available"] = bool(priced.get("as_low_as_with_coupon_spoken"))

    return {
        "filled": True,
        "medication": match["medication"],
        "say_the_name_as": match["medication"],
        "strength_spoken": spoken.strength(match["strength"]),
        "day_supply_spoken": spoken.day_supply(match["day_supply"]),
        "delivery": "free two-day delivery",
        "refills_remaining_after": max(0, match["refills_remaining"] - 1),
        **money,
        "guidance": [
            SAY_IT_AS_WRITTEN,
            "It is done. Confirm it in one sentence -- say it is being filled and will "
            "ship with free two-day delivery. Do not hedge, do not say you cannot place "
            "orders, and do not offer a transfer for this.",
            "Only if the caller mentioned a coupon, savings, or the price: when "
            "coupon_available is true, say the coupon is applied and quote "
            "as_low_as_with_coupon_spoken; when it is false, say plainly that there is "
            "no coupon for this one and give insurance_avg_price_spoken instead. Never "
            "leave a coupon question unanswered, and never volunteer any of this if "
            "they did not ask.",
            BE_BRIEF,
        ],
    }


def account_info(topic: str = "profile") -> dict:
    """One tool covering the whole account: profile, prescriptions, orders, refills."""
    t = (topic or "profile").strip().lower()

    if t in ("orders", "order", "order_status", "shipping", "tracking", "delivery"):
        return {
            "topic": "orders",
            "orders": [
                {
                    "status": o["status"],
                    "placed": _when(-o["placed_days_ago"]),
                    "arrives": _when(o["arrives_in_days"]) if o["status"] != "delivered"
                    else f"delivered {_when(o['arrives_in_days'])}",
                    "items": o["items"],
                    "delivery_speed": o.get("delivery_speed"),
                    "total_dollars": o["total"],
                    "total_spoken": spoken.money(o["total"]),
                    "tracking": o.get("tracking"),
                    "note": o.get("note"),
                }
                for o in ORDERS
            ],
            "spoken_summary": orders_summary([
                {"status": o["status"], "items": o["items"]} for o in ORDERS
            ]),
            "guidance": [
                SAY_IT_AS_WRITTEN,
                "Answer with spoken_summary and stop. Never walk the list: no bullets, "
                "no dashes, no numbering, and do not recite each order's status, "
                "arrival date and price in turn. The per-order fields are for the "
                "follow-up about ONE medication, once the caller names it.",
                BE_BRIEF,
            ],
        }

    if t in ("prescriptions", "prescription", "medications", "refills", "refill", "list"):
        return {
            "topic": "prescriptions",
            "prescriptions": [
                {
                    "medication": p["medication"],
                    "strength": p["strength"],
                    "strength_spoken": spoken.strength(p["strength"]),
                    "day_supply": p["day_supply"],
                    "day_supply_spoken": spoken.day_supply(p["day_supply"]),
                    "status": p["status"],
                    "refills_remaining": p["refills_remaining"],
                    "next_refill": _when(p["next_refill_in_days"]),
                    "auto_refill": p["auto_refill"],
                    "prescriber": p["prescriber"],
                    "note": p.get("note"),
                }
                for p in PRESCRIPTIONS
            ],
            # "Which of those can I refill?" was answered from the model's own
            # reading of the table, and it silently dropped Zepbound -- which was
            # ready to fill with two refills left. Compute it instead.
            "refillable_now": [
                p["medication"] for p in PRESCRIPTIONS
                if p["refills_remaining"] > 0 and p["status"] != "awaiting prescriber"
            ],
            "refillable_spoken": refillable_summary(
                [p["medication"] for p in PRESCRIPTIONS
                 if p["refills_remaining"] > 0 and p["status"] != "awaiting prescriber"],
                any_filled=False,
            ),
            "spoken_summary": prescriptions_summary([
                {"medication": p["medication"], "status": p["status"]} for p in PRESCRIPTIONS
            ]),
            "guidance": [
                SAY_IT_AS_WRITTEN,
                "This payload answers a question; it is not something to announce. If "
                "the caller did not ask what they have or what they can refill -- if you "
                "called this tool to check something for yourself -- use the fields "
                "silently and say nothing about the list. Reading it out unasked is how "
                "a caller who asked to fill one prescription got all five read back.\n"
                "When they DID ask, answer with ONE of these two lines and stop:\n"
                "  - asked what they can refill, what else they can refill, or what "
                "is due -> say refillable_spoken\n"
                "  - asked what they have, what is on file, or what they are taking "
                "-> say spoken_summary\n"
                "Never walk the list: no numbering, no reciting each medication's "
                "strength, day supply, status and refill count in turn. The "
                "per-prescription fields are for the follow-up about ONE medication, "
                "once the caller names it.",
                BE_BRIEF, ONLY_THIS_CALLER, FILL_IS_AVAILABLE,
            ],
        }

    if t in ("insurance", "coverage", "plan", "payment", "billing"):
        return {
            "topic": "insurance and payment",
            "insurance_plan": PROFILE["insurance_plan"],
            "insurance_status": PROFILE["insurance_status"],
            "payment_method": PROFILE["payment_method"],
            "prime_member": PROFILE["prime_member"],
            "rxpass": PROFILE["rxpass"],
            "guidance": [BE_BRIEF, ONLY_THIS_CALLER],
        }

    return {"topic": "profile", **PROFILE, "guidance": [BE_BRIEF, ONLY_THIS_CALLER]}


def topics() -> list[str]:
    return sorted({e["topic"] for e in ENTRIES})
