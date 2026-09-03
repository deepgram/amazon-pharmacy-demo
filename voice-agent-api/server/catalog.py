"""Medication catalog: fuzzy lookup and 90-to-84-day pricing from catalog.json."""

import difflib
import json
import re
from pathlib import Path

from . import spoken

CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog.json"

with open(CATALOG_PATH) as f:
    _DATA = json.load(f)

MEDICATIONS = _DATA["medications"]
POLICIES = _DATA["policies"]

_UNITS = {
    "milligram": "mg", "milligrams": "mg", "mg": "mg",
    "microgram": "mcg", "micrograms": "mcg", "mcg": "mcg", "ug": "mcg",
    "unit": "unit", "units": "unit",
    "milliliter": "ml", "milliliters": "ml", "ml": "ml",
    "gram": "g", "grams": "g", "g": "g",
    "milliequivalent": "meq", "milliequivalents": "meq", "meq": "meq",
    "%": "pct", "percent": "pct",
}


def _norm_strength(text: str) -> tuple[str, str]:
    """'12.5 milligrams' -> ('12.5', 'mg'). '90 mcg per actuation' -> ('90', 'mcg').
    '12.5' -> ('12.5', ''). Returns ('', '') when there is no number to work with."""
    raw = str(text or "")
    m = re.search(r"([\d]+(?:\.[\d]+)?)\s*(%|[a-zA-Z]+)?", raw)
    if not m:
        # No digit anywhere: the caller's own words may have been passed straight
        # through ("twelve point five milligrams"). Retry on the digit form
        # rather than reporting no such strength for one we stock.
        m = re.search(r"([\d]+(?:\.[\d]+)?)\s*(%|[a-zA-Z]+)?", spoken.numeric_from(raw))
    if not m:
        return "", ""
    num = m.group(1).lstrip("0") or "0"
    if num.startswith("."):
        num = "0" + num
    num = num.rstrip(".") if "." in num and num.endswith(".") else num
    unit = _UNITS.get((m.group(2) or "").lower(), "")
    return num, unit


# Withhold demo-scaffolding notes so the model cannot read them aloud.
_INTERNAL_NOTE_MARKERS = (
    "on purpose", "demo", "exercises", "scaffold", "todo", "fixme",
    "placeholder", "test only", "internal",
)


def is_internal_note(note: str) -> bool:
    low = (note or "").lower()
    return any(m in low for m in _INTERNAL_NOTE_MARKERS)


def _handling(med: dict) -> dict:
    """Delivery flags. signature_required follows from refrigerated | controlled."""
    cold = bool(med.get("refrigerated"))
    controlled = bool(med.get("controlled_schedule"))
    out: dict = {
        "refrigerated": cold,
        "signature_required": cold or controlled,
    }
    if cold:
        out["cold_chain"] = POLICIES["cold_chain"]
    if out["signature_required"]:
        out["delivery_note"] = POLICIES["signature"]
        out["po_box_ok"] = False
    return out


def _safe_note(note: str) -> str:
    return "" if is_internal_note(note) else (note or "")


_SCHEDULE_WORD = {"CII": "two", "CIII": "three", "CIV": "four", "CV": "five"}


# The clinical boundary rides on the tool result that supplies the strengths,
# rather than living only in the system prompt. Prompt-only, it held on one call
# and not the next: the agent volunteered "the most common starting dose is two
# point five milligrams" and then priced that strength. The prompt already
# declares that a tool result's `guidance` outranks any general habit, so putting
# the rule next to the data it governs is the strongest place available.
# Every field ending in `_spoken` is already the sentence-ready form. The model
# used to be asked to convert 50.0 into "fifty dollars" itself, and on one run
# 126 assistant turns went out with a dollar sign or a bare decimal still in them
# -- text-to-speech then reads "$4.99" as "four ninety nine". It also confused a
# strength for a price once, quoting a coupon "as low as $12.50" off "12.5 mg".
SAY_IT_AS_WRITTEN = (
    "Read every _spoken field aloud exactly as written and never re-render the "
    "number yourself: no digits, no dollar signs, no decimals. Say the medication "
    "by its say_the_name_as value, never the name you heard."
)

# The caller who says "my doctor already sent it" is telling you the strength and
# day supply are on file. On a live call the agent answered that by asking which
# strength they wanted -- of the six Zepbound strengths -- while account.json had
# Zepbound 12.5 mg, 30 day, "ready to fill" sitting in it. Asking someone to
# recite their own prescription back is the moment a pharmacy demo loses a room.
FILL_IS_AVAILABLE = (
    "If the caller wants a prescription that is on file filled, sent, ordered or shipped, "
    "call fill_prescription with the medication name and confirm what it returns. Never "
    "tell the caller you are unable to place, fill or order anything: on a live call that "
    "sentence was said four times to someone whose prescription was sitting at ready to "
    "fill, and each refusal made them ask again. Your own limits are not the caller's "
    "problem to solve."
)

BE_BRIEF = (
    "Two short sentences, then stop and let the caller speak. Ask at most one question. "
    "Do not read the strength list unless they asked what strengths we carry: if they "
    "named a strength, use it; if they did not, just ask which one. Do not staple the "
    "day-supply list, the prescription requirement, delivery or storage onto an answer "
    "that was not about them -- each is its own question, and the rules below say when "
    "each one is actually owed."
)

CHECK_ACCOUNT_FIRST = (
    "If the caller says they already have a prescription with us, or that their "
    "prescriber has sent one, call get_account_info with topic prescriptions "
    "BEFORE asking them for a strength or a day supply -- both are on file. Only "
    "ask them if the lookup comes back without that medication."
)

ACCEPT_NINETY = (
    "If the caller asks for ninety days, that is a valid request: never say ninety "
    "is unavailable and never re-offer the day-supply list. Call check_price with "
    "day_supply ninety, then tell them a ninety day request dispenses as an eighty "
    "four day supply and quote that price. Do not make them choose again."
)

NO_DOSE_ADVICE = (
    "Never call a strength typical, common, standard, starting, recommended, or "
    "what most people take, and never rank these strengths. If the caller asks "
    "which one they should take or which is usual, say you cannot advise on dose "
    "and offer a pharmacist callback or connection. List strengths only as the "
    "options we carry, and only price the one the CALLER names."
)


def _speaking_rules(med: dict, handling: dict) -> list[str]:
    """Per-drug speak/don't-speak lines, returned on the lookup so they are not in the prompt."""
    rules: list[str] = []

    if not med.get("in_stock", False):
        rules.append(
            "Out of stock. Say so plainly, never suggest a substitute drug -- that is a "
            "prescribing decision -- and offer to have a pharmacist follow up."
        )

    sched = med.get("controlled_schedule")
    if sched == "CII":
        rules.append(
            "Schedule two. The prescription must come from the prescriber electronically "
            "each time, cannot be transferred from another pharmacy, and cannot exceed a "
            "thirty day supply."
        )
    elif sched:
        rules.append(
            f"Schedule {_SCHEDULE_WORD.get(sched, sched)}. Refills and pharmacy transfers "
            "ARE allowed for this one -- do not apply the Schedule two rules to it."
        )

    if med.get("requires_prescription", True):
        rules.append(
            "Prescription required. Before any talk of placing an order, say it has to come "
            "from their prescriber, and offer to transfer it from another pharmacy for them."
        )
        rules.append(CHECK_ACCOUNT_FIRST)

    if med.get("refills_allowed") == 0:
        rules.append(
            "No refills on file. The prescriber has to send a new prescription each time, and "
            "can post-date them so the caller is covered month to month."
        )

    if med.get("max_day_supply"):
        rules.append(
            f"Never quote or offer more than a {med['max_day_supply']} day supply for this one."
        )

    detail = [d for d, on in (("cold-pack shipping", handling.get("refrigerated")),
                              ("signature on delivery", handling.get("signature_required")))
              if on]
    if detail:
        rules.append(
            f"Mention {' and '.join(detail)} only once the caller shows purchase intent or "
            "asks about delivery. Do not volunteer it alongside the price."
        )

    rules.append(NO_DOSE_ADVICE)
    return rules


def _strength_key(text: str) -> str:
    num, unit = _norm_strength(text)
    return f"{num}{unit}" if num else ""


def _index() -> dict[str, list[dict]]:
    """Lowercase name/generic/brand -> medications. Ambiguous names stay as a list."""
    idx: dict[str, list[dict]] = {}

    def add(alias: str | None, med: dict) -> None:
        if not alias:
            return
        bucket = idx.setdefault(alias.lower(), [])
        if med not in bucket:
            bucket.append(med)

    for m in MEDICATIONS:
        add(m["name"], m)
        add(m.get("generic"), m)
        add(m.get("brand_equivalent"), m)
    for m in MEDICATIONS:
        idx[m["name"].lower()] = [m]
    return idx


_INDEX = _index()
_KEYS = list(_INDEX)          # hoisted: rebuilt per lookup before

# Silent accept only if the top fuzzy hit beats the runner-up by this much.
_MATCH_MARGIN = 0.08


def _resolve(q: str) -> tuple[dict | None, list[str]]:
    """(medication, suggestions). Exactly one of the two is meaningful."""
    hits = _INDEX.get(q)
    if hits:
        if len(hits) == 1:
            return hits[0], []
        return None, [m["name"] for m in hits]

    close = difflib.get_close_matches(q, _KEYS, n=3, cutoff=0.6)
    if not close:
        return None, []

    best = difflib.SequenceMatcher(None, q, close[0]).ratio()
    runner_up = difflib.SequenceMatcher(None, q, close[1]).ratio() if len(close) > 1 else 0.0
    tie_is_same_drug = (
        len(close) > 1
        and {m["name"] for m in _INDEX[close[0]]} == {m["name"] for m in _INDEX[close[1]]}
    )
    if (
        best >= 0.8
        and len(_INDEX[close[0]]) == 1
        and (best - runner_up >= _MATCH_MARGIN or tie_is_same_drug)
    ):
        return _INDEX[close[0]][0], []

    seen, suggestions = set(), []
    for cand in close:
        for m in _INDEX[cand]:
            if m["name"] not in seen:
                seen.add(m["name"])
                suggestions.append(m["name"])
    return None, suggestions


def _strengths(med: dict) -> list[str]:
    """Display strengths, e.g. ['12.5 mg', '15 mg']. Falls back to the pre-v2
    strengths_mg / strengths_mcg shape so a hand-edited entry still loads."""
    if med.get("strengths"):
        return list(med["strengths"])
    for field, unit in (("strengths_mg", "mg"), ("strengths_mcg", "mcg")):
        if med.get(field):
            return [f"{s} {unit}" for s in med[field]]
    return []


def _supplies(med: dict) -> list[str]:
    """Day-supply keys on file, e.g. ['30_day', '84_day'], in numeric order."""
    seen = {k for row in med.get("pricing", {}).values() for k in row}
    return sorted(seen, key=lambda k: int(k.split("_")[0]))


def _spoken_supplies(med: dict) -> list[str]:
    """Sentence-ready, because the model reads this list straight out when it asks
    which supply the caller wants: 'thirty days or eighty four days?'"""
    return [f"{spoken.integer(int(k.split('_')[0]))} days" for k in _supplies(med)]


def all_names() -> list[str]:
    """Brand names plus brand equivalents."""
    names = []
    for m in MEDICATIONS:
        names.append(m["name"])
        if m.get("brand_equivalent"):
            names.append(m["brand_equivalent"])
    return names


# Programme and policy vocabulary. These are brand names the ASR has never seen
# and cannot guess from context, and they were missing entirely: the keyterm list
# was 68 drug names and nothing else, so "RxPass" came through as "rx pass" and
# the knowledge-base lookup went to the wrong entry.
PROGRAM_KEYTERMS = [
    "RxPass", "PillPack", "Amazon Pharmacy", "Prime membership",
    "prior authorization", "auto refill", "day supply", "prescription transfer",
    "flexible spending account", "health savings account", "copay",
    "formulary", "deductible", "mail order",
]


def keyterms(limit: int = 100) -> list[str]:
    """STT keyterms: catalog rows flagged `keyterm`, plus the programme vocabulary."""
    names: list[str] = []
    for m in MEDICATIONS:
        if not m.get("keyterm"):
            continue
        names.append(m["name"])
        if m.get("brand_equivalent"):
            names.append(m["brand_equivalent"])
    # programme terms first: they are the ones with no fallback if the list is cut
    return (PROGRAM_KEYTERMS + names)[:limit]


# Roll per-drug categories up for the prompt. First match wins.
_AREAS: list[tuple[tuple[str, ...], str]] = [
    (("weight loss",), "weight loss"),
    (("diabetes", "insulin"), "diabetes"),
    (("cholesterol", "triglycerides"), "cholesterol"),
    (("blood thinner", "antiplatelet"), "blood thinners"),
    (("blood pressure", "heart", "chest pain", "fluid retention"), "heart and blood pressure"),
    (("thyroid", "hormone", "birth control", "osteoporosis"), "thyroid and hormones"),
    (("adhd",), "ADHD"),
    (("depression", "anxiety", "bipolar", "schizophrenia", "sleep", "insomnia"), "mental health and sleep"),
    (("seizure", "nerve", "migraine", "alzheimer", "parkinson"), "neurology"),
    (("pain", "anti-inflammatory", "muscle relaxant", "gout", "arthritis", "psoriasis"), "pain and inflammation"),
    (("antibiotic", "antiviral", "antifungal", "hiv"), "infections"),
    (("acid reflux", "nausea", "bowel", "colitis", "constipation", "diarrhea", "ulcer", "stool"), "stomach and digestion"),
    (("asthma", "copd", "allerg", "cough"), "asthma and allergies"),
    (("eye", "glaucoma"), "eye care"),
    (("skin", "acne", "topical", "scabies", "steroid"), "skin"),
    (("prostate", "bladder", "erectile", "urinary"), "urology"),
    (("vitamin", "supplement", "iron", "potassium"), "vitamins and supplements"),
    (("smoking",), "smoking cessation"),
    (("alcohol", "dependence"), "addiction treatment"),
    (("corticosteroid", "biologic", "transplant"), "specialty medications"),
]


def categories() -> list[str]:
    """Broad therapeutic areas for the prompt."""
    seen, out = set(), []
    for m in MEDICATIONS:
        cat = m.get("category", "").lower()
        area = next((a for keys, a in _AREAS if any(k in cat for k in keys)), "other")
        if area not in seen:
            seen.add(area)
            out.append(area)
    return sorted(out)


def pronunciation_lines(keyterm_only: bool = True, limit: int | None = None,
                        pin: tuple[str, ...] = ()) -> list[str]:
    """Pronunciation lines for the prompt. `pin` keeps named drugs in the budget."""
    lines: list[tuple[str, str]] = []
    for m in MEDICATIONS:
        if keyterm_only and not m.get("keyterm"):
            continue
        if m.get("pronunciation"):
            lines.append((m["name"], f'{m["name"]} is "{m["pronunciation"]}".'))
    if limit is None or len(lines) <= limit:
        return [t for _, t in lines]
    pinned = [t for n, t in lines if n in pin]
    rest = [t for n, t in lines if n not in pin]
    return (rest[:max(0, limit - len(pinned))] + pinned)


# Policies the agent now learns per-drug, from the `guidance` on a tool result or
# from _handling, rather than in the abstract on every turn. Stating them twice
# costs prompt budget we do not have -- see prompt.PROMPT_CEILING -- and the static
# wording is the less accurate of the two: "Schedule II cannot be transferred" as a
# standing rule is what made the agent apply it to Schedule IV alprazolam.
PER_DRUG_POLICIES = frozenset({
    "controlled_substances", "out_of_stock", "cold_chain", "signature", "day_supply",
    # These three now ride on tool results too, and the prompt has no room to say
    # anything twice: `price_disclaimer` comes back on every check_price (and the
    # server speaks it itself if the model skips it), and _speaking_rules already
    # emits the prescription-required and offer-to-transfer line per drug.
    "price_disclaimer", "prescription_transfer", "new_prescription",
})


def policy_lines(skip: frozenset[str] = PER_DRUG_POLICIES) -> list[str]:
    return [f"{k.replace('_', ' ').capitalize()}: {v}"
            for k, v in POLICIES.items() if k not in skip]


def lookup(name: str) -> dict:
    """Exact/alias match first, then fuzzy. Returns either the med summary or
    {'match': None, 'suggestions': [...]} so the agent can ask 'did you mean X'."""
    q = (name or "").strip().lower()
    # callers say "my water pill" and "the z pack"; the article is never part of
    # the drug name and only drags the fuzzy ratio down
    q = re.sub(r"^(the|my|a|an|some|his|her|their|our)\s+", "", q).strip()
    if not q:
        return {"match": None, "suggestions": []}

    med, suggestions = _resolve(q)
    if med is None:
        out = {"match": None, "heard": name, "suggestions": suggestions}
        if len(suggestions) > 1:
            out["guidance"] = [
                "These are different products. Ask which one the caller means and "
                "never pick for them."
            ]
        elif suggestions:
            top = _INDEX[suggestions[0].lower()][0]
            out["guidance"] = [
                f"Ask \"did you mean {suggestions[0]}\" and say what it treats "
                f"({top.get('category')}). Never pretend to recognise a name."
            ]
        else:
            out["guidance"] = [
                "Nothing matched. Ask the caller to say the name again or describe what it "
                "treats. Do not ask them to spell it, and do not say we don't carry it."
            ]
        return out

    out = {
        "match": med["name"],
        "say_the_name_as": med["name"],
        "generic": med.get("generic"),
        "brand_equivalent": med.get("brand_equivalent"),
        "category": med.get("category"),
        "form": med.get("form"),
        "strengths": _strengths(med),
        "strengths_spoken": [spoken.strength(s) for s in _strengths(med)],
        "day_supplies_available": _spoken_supplies(med),
        "in_stock": med.get("in_stock", False),
        "requires_prescription": med.get("requires_prescription", True),
        "pronunciation": med.get("pronunciation", ""),
        "note": _safe_note(med.get("note", "")),
    }
    handling = _handling(med)
    out.update(handling)
    if med.get("controlled_schedule"):
        out["controlled_schedule"] = med["controlled_schedule"]
        out["transferable"] = bool(med.get("transferable", True))
        out["max_day_supply"] = med.get("max_day_supply")
        out["refills_allowed"] = med.get("refills_allowed")
    rules = _speaking_rules(med, handling)
    # day_supplies_available reads as a closed menu, and once 60-day pricing was
    # added it read as a complete one -- so a caller who said "I want ninety days
    # instead" got "we have thirty, sixty, or eighty four, which works best?" and
    # had to pick 84 themselves. Ninety is a valid REQUEST that converts; it is
    # not a missing option. The rule belongs next to the list it qualifies.
    if any(k == "84_day" for k in _supplies(med)):
        out["ninety_day_requests"] = "accepted; dispensed as an eighty four day supply"
        rules.insert(0, ACCEPT_NINETY)
    rules.insert(0, SAY_IT_AS_WRITTEN)
    rules.insert(1, BE_BRIEF)
    out["guidance"] = rules
    return {k: v for k, v in out.items() if v not in (None, "")}


def dispensed_supply(supply: int) -> int:
    """What a requested day supply actually dispenses as. 90 -> 84, everything else
    unchanged (insurance dispensing cycles).

    Named rather than inline because two callers need the same rule: price() applies
    it, and tools.check_price runs it backwards to spot a model that already applied
    it and would otherwise leave the caller un-told.
    """
    return 84 if supply >= 85 else supply


def price(name: str, strength: str, day_supply: int) -> dict:
    """Price for a drug/strength/day-supply. 90-day requests dispense as 84."""
    res = lookup(name)
    if not res.get("match"):
        return res

    med = _INDEX[res["match"].lower()][0]
    pricing = med.get("pricing", {})

    # match the caller's strength against the keys on file. "12.5 mg", "12.5mg"
    # and a bare "12.5" all land on the same row; a bare number only matches when
    # exactly one strength shares it, so 5 mg and 5 mcg can never be confused.
    num, unit = _norm_strength(strength)
    matched = None
    if num:
        exact = [k for k in pricing if _norm_strength(k) == (num, unit)] if unit else []
        numeric = [k for k in pricing if _norm_strength(k)[0] == num]
        if exact:
            matched = exact[0]
        elif len(numeric) == 1:
            matched = numeric[0]
    if matched is None:
        return {"match": med["name"], "error": f"no strength on file matching {strength}",
                "available_strengths": _strengths(med),
                "available_strengths_spoken": [spoken.strength(s) for s in _strengths(med)],
                "guidance": [SAY_IT_AS_WRITTEN,
                             "Read back the strengths we do carry. Do not invent a price "
                             "for a strength that is not on this list.",
                             NO_DOSE_ADVICE]}
    row = pricing[matched]
    # answer with the strength as the catalog spells it, not as the caller said it:
    # a caller who says "one hundred milligrams" of a microgram drug must hear
    # "one hundred micrograms" back, not their own wrong unit repeated.
    display = next((d for d in _strengths(med) if _strength_key(d) == matched), matched)

    # supply normalization: 90 -> 84 (insurance dispensing cycles)
    try:
        supply = int(day_supply)
    except (TypeError, ValueError):
        # silently defaulting to 30 quoted the 30-day price for a 90-day
        # request -- a wrong number said out loud. Make the model re-ask.
        return {"match": med["name"], "strength": display,
                "error": f"could not read a day supply from {day_supply!r}",
                "guidance": "Ask the caller to confirm the day supply, then call check_price again.",
                "day_supplies_available": _spoken_supplies(med)}
    dispensed = dispensed_supply(supply)
    entry = row.get(f"{dispensed}_day")
    if entry is None:
        return {"match": med["name"], "strength": display,
                "error": f"no {dispensed} day supply on file for this medication",
                "day_supplies_available": _spoken_supplies(med),
                "guidance": ["Say what supplies this medication does come in, for example a "
                             "ten day course. Do not offer a supply that is not listed here."]}

    out = {
        "match": med["name"],
        "say_the_name_as": med["name"],
        "strength": display,
        "strength_spoken": spoken.strength(display),
        "requested_day_supply": supply,
        "dispensed_day_supply": dispensed,
        "dispensed_day_supply_spoken": spoken.day_supply(dispensed),
        "insurance_avg_price": entry.get("insurance_avg"),
        "insurance_avg_price_spoken": spoken.money(entry.get("insurance_avg")),
        "cash_price": entry.get("cash"),
        "cash_price_spoken": spoken.money(entry.get("cash")),
        "coupon_savings_up_to": entry.get("coupon_savings_up_to"),
        "coupon_savings_up_to_spoken": spoken.money(entry.get("coupon_savings_up_to")),
        "volume": entry.get("volume"),
        "note": _safe_note(entry.get("note", "")),
        "price_disclaimer": POLICIES["price_disclaimer"],
        # only promise delivery for something we can actually dispense
        "same_day_delivery_eligible": bool(med.get("in_stock", False)),
        "in_stock": bool(med.get("in_stock", False)),
        **_handling(med),
    }
    rules = _speaking_rules(med, _handling(med))
    if not med.get("in_stock", False):
        # The stress-test scenario: "Ask for Wegovy -> says it's unavailable".
        # Without this the agent quoted a cash price and offered same-day.
        out["availability"] = "OUT OF STOCK - do not quote this as purchasable"
        rules.insert(0, "Do not quote this price as if it can be ordered.")
    if entry.get("coupon_savings_up_to") and entry.get("insurance_avg") is not None:
        floor = round(entry["insurance_avg"] - entry["coupon_savings_up_to"], 2)
        out["as_low_as_with_coupon"] = floor
        out["as_low_as_with_coupon_spoken"] = spoken.money(floor)
    if entry.get("insurance_avg") is None:
        out["insurance_coverage"] = "not established for this medication"
        rules.append("Quote the cash price and say coverage varies by plan. Never say a "
                     "price is zero.")
    if dispensed != supply:
        # the caller asked for 90 and will be billed for 84; they have to hear it
        rules.append(f"Say plainly that a {spoken.integer(supply)} day request dispenses as "
                     f"{spoken.day_supply(dispensed)}, and quote the price against the "
                     "dispensed number.")
    # "Lead with the price" was too vague once the payload carried two of them:
    # on a live call the model opened with the nine-hundred-ninety-nine dollar
    # cash price and never mentioned the fifty dollar insurance price at all.
    # Name the field that leads.
    if entry.get("insurance_avg") is not None:
        lead = ("Lead with insurance_avg_price_spoken as the price, then offer "
                "cash_price_spoken as the alternative for paying without insurance. "
                "Never quote the cash price first when an insurance price exists.")
    else:
        lead = "Lead with cash_price_spoken; there is no insurance price on file for this one."
    if out.get("as_low_as_with_coupon_spoken"):
        lead += (" as_low_as_with_coupon_spoken is what they would pay WITH the coupon, "
                 "not the saving; say it as 'as low as ...'.")
    rules.append("Say the estimate disclaimer once per call, never twice.")
    rules.append(FILL_IS_AVAILABLE)
    rules.insert(0, lead)
    rules.insert(0, SAY_IT_AS_WRITTEN)
    rules.insert(1, BE_BRIEF)
    if med.get("controlled_schedule"):
        out["controlled_schedule"] = med["controlled_schedule"]
    out["guidance"] = rules
    return {k: v for k, v in out.items() if v not in (None, "")}
