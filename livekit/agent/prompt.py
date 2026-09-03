"""Builds the live system prompt from prompts/system-prompt.md plus catalog notes.

Ported verbatim from `pharmacy-voice-agent/server/prompt.py`. The prompt is the
agent's brain, and the A/B between the two paths is only meaningful if it is
byte-identical -- so this file, `prompts/system-prompt.md`, and the three
knowledge bases must stay in step across both repos.

Section headings stay in the prompt; only the file title and `>` review notes
are stripped. Drug facts come from catalog.json, not this file. The complete
prompt is passed once as the Agent's `instructions`; this use case needs no
mid-call prompt update.
"""

from pathlib import Path

from . import catalog, knowledge

PROMPT_MD = Path(__file__).resolve().parent.parent / "prompts" / "system-prompt.md"

# Retained from Path A, where it is a hard transport limit rather than a style
# guide: the Voice Agent API silently truncates above ~16,150 characters and
# reports it as a Warning, not an Error -- the call connects, sounds completely
# healthy, and runs with an unknown tail of its own instructions missing.
#
# LiveKit has no such ceiling; the prompt goes to OpenAI or Anthropic directly,
# where 16 KB is unremarkable. The budget is kept anyway, because a prompt that
# grows past it here stops being runnable on Path A, and a prompt the two paths
# do not share makes the comparison worthless. scripts/check_catalog.py enforces
# it in both repos for that reason, not because this transport cares.
PROMPT_LIMIT = 16_000
PRONUNCIATION_BUDGET = 22
PINNED_PRONUNCIATIONS = ("Alprazolam", "Adderall XR")

VERIFY_NOTES = """
## Runtime notes
- Keypad input: when the caller types on their phone keypad, you receive a message like "[keypad] 1234567". Treat it exactly as if the caller spoke those digits.
- The demo account has one open order: Aspirin 81 milligrams, 30 day supply. After successful verification, ask "Are you calling about your Aspirin order?"
"""

PHARMACY_NOTES = """
## What the catalog holds
{count} medications, spanning: {categories}.
That is broad. Assume we carry a medication until lookup_medication tells you otherwise, and never tell a caller we do not stock something without calling it first.
Some names cover more than one product. Semaglutide is Ozempic, Wegovy or Rybelsus; tirzepatide is Zepbound or Mounjaro. When lookup_medication comes back with several suggestions instead of a match, ask the caller which one they mean. Never pick one yourself.

## Pronunciation
{pronunciations}

## Pharmacy policies
{policies}

## Everything else
search_knowledge_base covers these topics: {topics}. get_account_info covers this caller's own orders, prescriptions, refill dates, insurance and payment method.
Between them, almost anything a pharmacy caller asks has an answer. Reach for them before you ever consider saying you cannot help.
"""


def _markdown() -> str:
    """Drop the file title and review quotes from the human-readable prompt."""
    return "\n".join(
        line for line in PROMPT_MD.read_text().split("\n")
        if not line.startswith("# System Prompt") and not line.startswith(">")
    ).strip()


def build() -> str:
    """Build the single prompt handed to the Agent as instructions."""
    return _markdown() + "\n" + VERIFY_NOTES + "\n" + PHARMACY_NOTES.format(
        count=len(catalog.MEDICATIONS),
        categories=", ".join(catalog.categories()),
        pronunciations=" ".join(catalog.pronunciation_lines(
            limit=PRONUNCIATION_BUDGET, pin=PINNED_PRONUNCIATIONS)),
        policies="\n".join(f"- {line}" for line in catalog.policy_lines()),
        topics=", ".join(knowledge.topics()),
    )


# Amazon's compliance block, delivered verbatim. The 911 instruction and the
# recording disclosure ride on this line, which is why session.py speaks it
# itself with interruptions disallowed rather than trusting the model to open
# with it.
GREETING = (
    "Thank you for calling Amazon Pharmacy and PillPack. If this is an emergency, "
    "please hang up and dial nine one one. All calls are recorded for quality and "
    "security."
)

# Asked as its own turn, immediately after the block above. Kept separate so the
# compliance text stays exactly as Amazon wrote it and the verification question
# remains ours to word.
LINE_NUMBER_CHECK = (
    "To get started, I'll need to verify your account. Is the mobile number you "
    "are calling from associated with your account?"
)
