"""Numbers, money and drug strengths as spoken English.

Everything the model reads aloud should arrive already spoken. Asking it to turn
`50.0` into "fifty dollars" is a conversion on the hot path done by the component
least able to do it reliably: across one 135-call evaluation, 126 assistant turns
went out carrying a dollar sign or a bare decimal, and text-to-speech reads
"$4.99" as "four ninety nine" -- the word "dollars" simply disappears. Worse, a
float and a strength look alike, and "12.5 mg" came back out of the model once as
a coupon "as low as $12.50".

So the server does the conversion and hands over a string. The model's job is to
read the `_spoken` field, not to compute it.
"""

import re

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

# Spoken expansions for every unit that appears in catalog.json, plus the ones a
# caller might say. Singular is only used for exactly 1.
_UNIT_WORDS = {
    "mg": ("milligram", "milligrams"),
    "mcg": ("microgram", "micrograms"),
    "ug": ("microgram", "micrograms"),
    "g": ("gram", "grams"),
    "ml": ("milliliter", "milliliters"),
    "meq": ("milliequivalent", "milliequivalents"),
    "unit": ("unit", "units"),
    "units": ("unit", "units"),
    "%": ("percent", "percent"),
    "percent": ("percent", "percent"),
}


def integer(n: int) -> str:
    """0-999,999 as words. 'and' is omitted, the way a US speaker reads a number."""
    n = int(n)
    if n < 0:
        return "negative " + integer(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {integer(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    return f"{integer(thousands)} thousand" + (f" {integer(rest)}" if rest else "")


def number(value) -> str:
    """A number as words, decimals included: 12.5 -> 'twelve point five'.

    Decimal digits are read one at a time, which is how a strength is said aloud:
    "zero point four milligrams", never "zero point four tenths".
    """
    text = str(value).strip()
    if not text:
        return ""
    negative = text.startswith("-")
    text = text.lstrip("-+")
    if "." not in text:
        try:
            return ("negative " if negative else "") + integer(int(text))
        except ValueError:
            return text
    whole, _, frac = text.partition(".")
    frac = frac.rstrip("0")
    try:
        head = integer(int(whole or 0))
    except ValueError:
        return text
    if not frac:
        return ("negative " if negative else "") + head
    digits = " ".join(_ONES[int(d)] for d in frac if d.isdigit())
    return ("negative " if negative else "") + f"{head} point {digits}"


def money(value) -> str:
    """A price as words: 50.0 -> 'fifty dollars', 4.99 -> 'four dollars and ninety nine cents'."""
    if value is None:
        return ""
    try:
        cents = int(round(float(value) * 100))
    except (TypeError, ValueError):
        return ""
    dollars, rest = divmod(abs(cents), 100)
    parts = []
    if dollars or not rest:
        parts.append(f"{integer(dollars)} dollar" + ("" if dollars == 1 else "s"))
    if rest:
        parts.append(f"{integer(rest)} cent" + ("" if rest == 1 else "s"))
    out = " and ".join(parts)
    return ("negative " + out) if cents < 0 else out


def day_supply(days) -> str:
    """'an eighty four day supply'. The article matters: 'a' before a consonant sound."""
    try:
        n = int(days)
    except (TypeError, ValueError):
        return ""
    words = integer(n)
    # "an" before eight / eighteen / eighty / eleven, "a" before one -- the only
    # number words where the leading letter and the leading sound disagree.
    article = "an" if words.startswith("e") else "a"
    return f"{article} {words} day supply"


_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Homophones speech-to-text produces for a caller reading digits aloud. Safe
    # to accept because the only consumer is verify_otp, which still requires
    # exactly six of them and still has to match the real code.
    "to": "2", "too": "2", "for": "4", "fore": "4", "won": "1", "ate": "8",
    "niner": "9", "tree": "3",
}


def digits_from(text: str) -> str:
    """Pull a digit string out of whatever the caller said.

    verify_otp owns the length check, and the prompt tells the model to pass
    through what it heard rather than pre-judging it -- so the tool has to cope
    with "nine three six seven two two" as readily as "936722". A caller reading
    a code aloud produces words, and the transcript often keeps them that way.

    Digits and digit words are read in the order they appear, so a mixed
    utterance like "it's 9 3 six seven two two" still resolves.
    """
    out = []
    for token in re.findall(r"\d|[a-z]+", str(text or "").lower()):
        if token.isdigit():
            out.append(token)
        elif token in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[token])
    return "".join(out)


_SCALES = {"hundred": 100, "thousand": 1000}
_WORD_VALUES = {w: i for i, w in enumerate(_ONES)}
_WORD_VALUES.update({w: i * 10 for i, w in enumerate(_TENS) if w})


def numeric_from(text: str) -> str:
    """The inverse of strength(): spoken number words back into digits.

    `strength()` exists because the model cannot reliably turn 12.5 into words.
    The same gap runs the other way on the inbound side -- a model is free to
    call check_price with whatever form it please, and gpt-4o-mini sends the
    strength exactly as the caller said it:

        check_price(name="Zepbound", strength="twelve point five milligrams")

    Nothing downstream could read that. `_norm_strength` looks for a digit, found
    none, and the price lookup missed a drug we stock -- the agent then told the
    caller we had no price for a strength it had just listed as available.
    Claude happens to send "12.5 mg", so this never showed up until the think
    model changed. Accepting both forms is cheaper than requiring one.

    Only number words are rewritten; everything else survives untouched, so
    "twelve point five milligrams" -> "12.5 milligrams" and the existing unit
    parsing takes it from there.
    """
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        """Collapse a run of number words into one numeric token."""
        if not run:
            return
        total = current = 0
        frac = ""
        decimal = False
        for w in run:
            if w == "point":
                decimal = True
            elif decimal:
                frac += str(_WORD_VALUES.get(w, 0))
            elif w in _SCALES:
                scale = _SCALES[w]
                if scale == 1000:
                    total += max(current, 1) * 1000
                    current = 0
                else:
                    current = max(current, 1) * 100
            else:
                current += _WORD_VALUES[w]
        value = f"{total + current}"
        out.append(f"{value}.{frac}" if frac else value)
        run.clear()

    for token in re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z\d]", str(text or "")):
        low = token.lower()
        # "point" only continues a run; on its own it is the English word.
        if low in _WORD_VALUES or low in _SCALES or (low == "point" and run):
            run.append(low)
        else:
            flush()
            out.append(token)
    flush()
    return " ".join(out)


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def strength(text: str) -> str:
    """A catalog strength as words: '12.5 mg' -> 'twelve point five milligrams',
    '90 mcg per actuation' -> 'ninety micrograms per actuation',
    '100 units per mL' -> 'one hundred units per milliliter'.

    Unit words are driven off the number immediately before them, so a compound
    strength such as '5 mg and 325 mg' pluralises each half on its own.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    tokens = re.findall(r"\d+(?:\.\d+)?|%|[A-Za-z]+|[^\sA-Za-z\d]", raw)
    out: list[str] = []
    last_number: float | None = None
    for tok in tokens:
        if _NUM_RE.fullmatch(tok):
            last_number = float(tok)
            out.append(number(tok))
            continue
        forms = _UNIT_WORDS.get(tok.lower())
        if forms:
            # "100 units per mL" is per ONE millilitre: a unit with no number in
            # front of it is the denominator and stays singular.
            out.append(forms[1] if last_number not in (None, 1) else forms[0])
            last_number = None
            continue
        if tok == ",":
            out.append(",")
            continue
        out.append(tok)
    return re.sub(r"\s+,", ",", " ".join(out)).strip()
