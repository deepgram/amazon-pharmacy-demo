"""Client-side tools for the Voice Agent session: OTP flow + catalog.

Each session gets its own ToolBox (per-call OTP state). Definitions go into
agent.think.functions WITHOUT an endpoint, which makes them client-side:
Deepgram sends FunctionCallRequest and we answer with FunctionCallResponse.
"""

import asyncio
import json
import re
import random

from loguru import logger

from . import catalog, knowledge, spoken
from .config import (
    DEMO_OTP, LINE_TRANSFER, OTP_SMS, QUEUE_POSITION, QUEUE_WAIT_MIN,
    TOOL_LATENCY_MS, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER,
)

# Appended to every tool result. Kept short: it is paid for on every turn.
REPLY_RULE = (
    "Start speaking with the answer itself. Do not open with an acknowledgement "
    "such as 'that's a great question', and do not repeat any sentence you already "
    "said on this turn."
)

# end_call / transfer get their own rule: the generic one does not bind the
# second completion the API makes around a tool call.
END_REPLY_RULE = (
    "The closing line has already been spoken. Reply with nothing at all: no "
    "goodbye, summary, or repetition. The system does not disconnect the caller; "
    "the caller controls when the call ends."
)

TRANSFER_REPLY_RULE = (
    "The system already told the caller they are in the live-agent queue. "
    "Reply with nothing at all. Do not call end_call. The call stays open."
)

DEFINITIONS = [
    {
        "name": "send_otp",
        "description": (
            "Generate and send a six digit verification code to the caller's email "
            "and mobile phone. Call this once the caller confirms their mobile number "
            "is associated with their account."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "verify_otp",
        "description": (
            "Verify the code the caller provided, spoken or typed on the keypad. "
            "Returns whether it matched and how many attempts remain. This tool owns "
            "the length check: pass through whatever digits you heard and read the "
            "result. Never count the digits yourself, never tell the caller their "
            "code is the wrong length, and never repeat the digits back to them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string",
                     "description": "what the caller said, digits or spoken words -- "
                                    "'936722' and 'nine three six seven two two' both work"}
            },
            "required": ["code"],
        },
    },
    {
        "name": "end_call",
        "description": (
            "Mark the conversation complete after the closing line, after verification "
            "has failed twice, or after telling a caller with an emergency to hang up "
            "and dial nine one one. This does not disconnect the caller. The caller "
            "controls when the call ends. Never use this to "
            "transfer the caller to a person — call transfer_to_agent instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "one of: finished, verification_failed, emergency",
                }
            },
            "required": ["reason"],
        },
    },
    {
        "name": "transfer_to_agent",
        "description": (
            "Place the caller in the live-agent queue. Call this when they ask for a "
            "person, a representative, a pharmacist connection, or a transfer to a human. "
            "The system speaks the queue position and wait time. Do not call end_call."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "lookup_medication",
        "description": (
            "Look up a medication by name to check availability, form, and strengths. "
            "Use this for any drug question, including names you are unsure you heard "
            "correctly; it returns suggestions when the name does not match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "the medication name as heard"}
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Answer any pharmacy question that is not a drug price: delivery, shipping, "
            "refills, auto-refill, prescription transfers, insurance and coverage, prior "
            "authorizations, cash pricing, RxPass, PillPack, payment and receipts, returns, "
            "privacy, hours, and how to reach a pharmacist. Call this before saying you "
            "cannot help with something; it almost always has an answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "the caller's question, in their own words"}
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_account_info",
        "description": (
            "Look up this caller's own account: their orders and where each one is, their "
            "prescriptions and refill dates, or their insurance and payment method. Use this "
            "for anything about THEIR order, THEIR refill, or THEIR coverage. Requires "
            "verification first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["orders", "prescriptions", "insurance", "profile"],
                    "description": "orders for shipping and status, prescriptions for refills, "
                                   "insurance for coverage and payment, profile for everything else",
                }
            },
            "required": ["topic"],
        },
    },
    {
        "name": "fill_prescription",
        "description": (
            "Fill a prescription that is already on the caller's account. Use this "
            "whenever they ask to fill, order, send, ship, or place an order for one "
            "of their own prescriptions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "the medication to fill, as the caller said it",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "check_price",
        "description": (
            "Get the price for a medication at a given strength and day supply. "
            "Requires the exact medication name (use lookup_medication first if unsure)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "medication name"},
                "strength": {"type": "string", "description": "e.g. '12.5 mg' or '100 mcg'"},
                "day_supply": {"type": "integer", "description": "the day supply the CALLER asked for. Do not guess: if they have not said one, ask before calling this."},
            },
            "required": ["name", "strength", "day_supply"],
        },
    },
]


class ToolBox:
    """Per-call tool state + dispatch."""

    def __init__(self, caller_number: str | None = None, session_label: str = "call"):
        self.caller_number = caller_number
        self.label = session_label
        self.otp: str | None = None
        self.otp_attempts_remaining = 2
        self.verified = False
        self.locked = False
        self.end_call_reason: str | None = None
        self.in_queue = False
        # Set by the session when the agent offers a live person, so a caller
        # answering "yes" to that offer still reaches one. See _transfer_to_agent.
        self.offered_agent = False
        # The model can request verify_otp before faithfully copying a clearly
        # spoken code into the function arguments. The session updates this with
        # the latest caller transcript so verification has a safe server-side
        # recovery path for that specific turn.
        self.latest_user_text = ""
        # The session suppresses its idle nudge while the caller is reading the
        # code off their phone, which is exactly the window between these two.
        self.verify_attempts = 0
        # Prescriptions filled on THIS call: {medication: refills_left_after}.
        # fill_prescription used to compute the new refill count and throw it away,
        # so a caller who had just filled their Zepbound was still told it was
        # "ready to fill, two refills remaining" one turn later. account.json is
        # never written -- this overlay lives and dies with the call, so every demo
        # starts from the same clean account.
        self.filled: dict[str, int] = {}
        self._filled_display: dict[str, str] = {}   # lowercase key -> catalog spelling

    # Prompt-only gating fails; the server refuses these until verify_otp succeeds.
    # search_knowledge_base stays open (generic policy, nothing caller-specific).
    GATED_TOOLS = frozenset({"lookup_medication", "check_price", "get_account_info",
                             "fill_prescription"})

    async def dispatch(self, name: str, arguments: str) -> str:
        """Run a tool, return the FunctionCallResponse content string."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}
        if TOOL_LATENCY_MS > 0:
            await asyncio.sleep(TOOL_LATENCY_MS / 1000)

        if name in self.GATED_TOOLS and not self.verified:
            logger.warning(f"[{self.label}] BLOCKED {name}: caller not verified")
            return json.dumps({
                "error": "not_verified",
                "message": (
                    "The caller has not completed verification. Do not answer this "
                    "question. Ask them for the six digit code instead."
                ),
            })

        handler = getattr(self, f"_{name}", None)
        if handler is None:
            logger.warning(f"[{self.label}] unknown tool: {name}")
            return json.dumps({"error": f"unknown tool {name}"})
        result = await handler(args)
        logger.info(f"[{self.label}] tool {name}({args}) -> {result}")
        if isinstance(result, dict):
            rule = END_REPLY_RULE if name == "end_call" else REPLY_RULE
            if name == "transfer_to_agent":
                rule = TRANSFER_REPLY_RULE
            result = {**result, "_reply_rule": rule}
        return json.dumps(result)

    async def _end_call(self, args: dict) -> dict:
        reason = str(args.get("reason") or "finished")
        if self.in_queue and reason not in ("emergency", "verification_failed"):
            logger.info(f"[{self.label}] ignoring end_call({reason}): caller is in queue")
            return {
                "ending": False,
                "error": "in_queue",
                "message": (
                    "The caller is already in the live-agent queue. Do not hang up. "
                    "Reply with nothing."
                ),
            }
        self.end_call_reason = reason
        logger.info(f"[{self.label}] end_call requested: {self.end_call_reason}")
        return {"ending": True, "reason": self.end_call_reason}

    # A caller asking for a human, in their own words.
    _WANTS_PERSON = re.compile(
        r"\b(agent|human|person|someone|somebody|representative|rep|operator|"
        r"pharmacist|manager|supervisor|real (person|human))\b", re.I)

    def transfer_allowed(self) -> bool:
        """Is a transfer wanted -- because the caller asked, or because we offered?

        va.py speaks LINE_TRANSFER before dispatching the tool, to cover the call's
        latency. That is fine for a line that only narrates work, and wrong for one
        that asserts an outcome: with the guard below refusing the transfer, the
        caller still heard "Transferring you to a live agent now" and then got told
        their prescription would be shipped. Both sides now ask this one question.
        """
        if self.offered_agent:
            return True
        return bool(self._WANTS_PERSON.search(self.latest_user_text or ""))

    async def _transfer_to_agent(self, args: dict) -> dict:
        # Nobody asked to be transferred on the call this guard comes from. The
        # model offered "would you like to proceed with placing the order?", the
        # caller said yes -- and there is no order-placement tool, so the only
        # tool that moves a call forward is this one. It queued them behind a
        # fifteen minute wait for something they never requested.
        #
        # Refusing here rather than in the prompt, because the prompt already
        # says to transfer when the caller asks for a person and says nothing
        # about transferring when they have not. Two ways through: the caller
        # asked, or we offered a person and they took it (the dose-advice path,
        # where the answer is a bare "yes" and only the offer gives it meaning).
        if not self.transfer_allowed():
            logger.warning(
                f"[{self.label}] BLOCKED transfer_to_agent: nobody asked for a person "
                f"(last caller turn: {self.latest_user_text!r})"
            )
            return {
                "queued": False,
                "error": "transfer_not_requested",
                "guidance": (
                    "The caller has not asked for a person and you did not offer one, so "
                    "do not transfer them. Answer what they actually asked. If a "
                    "prescription is ready to fill and they want to go ahead, call "
                    "fill_prescription -- you can do that yourself. Offer a person only if "
                    "they ask for one."
                ),
            }
        self.in_queue = True
        logger.info(
            f"[{self.label}] queued for live agent "
            f"(position={QUEUE_POSITION}, wait={QUEUE_WAIT_MIN}m)"
        )
        return {
            "queued": True,
            "position": QUEUE_POSITION,
            "wait_minutes": QUEUE_WAIT_MIN,
            "line": LINE_TRANSFER,
        }

    async def _send_otp(self, args: dict) -> dict:
        # Asked to check his wife's prescriptions, the agent declined correctly --
        # and then took her phone number, sent a code to it, accepted the caller
        # reading it back, and listed five prescriptions. Nothing in the flow
        # stopped it, because the prompt rule was the only thing there. A session
        # verifies exactly one identity, so a second send is refused outright.
        if self.verified:
            logger.warning(f"[{self.label}] BLOCKED send_otp: session already verified")
            return {
                "sent": False,
                "error": "already_verified",
                "message": (
                    "This caller is already verified and a call verifies one person "
                    "only. If they are asking on behalf of someone else, decline: "
                    "tell them that person has to call us themselves, and do not take "
                    "that person's phone number or accept a code on their behalf. "
                    "Then offer to help with this caller's own account."
                ),
            }
        if self.locked:
            return {
                "sent": False,
                "error": "locked",
                "message": (
                    "Verification already failed twice on this call. Do not send another "
                    "code. Apologise, point the caller to the Amazon app or website, then "
                    "call end_call with reason verification_failed."
                ),
            }
        self.otp = DEMO_OTP if self.caller_number else f"{random.randint(0, 999999):06d}"
        self.otp_attempts_remaining = 2
        logger.info(f"[{self.label}] OTP generated: {self.otp}")
        channels = ["email", "mobile"]
        if OTP_SMS and self.caller_number and TWILIO_ACCOUNT_SID:
            if await asyncio.to_thread(self._sms_otp):
                channels.append("sms")
        return {
            "sent": True,
            "channels": channels,
            "already_announced": True,
            # va.py withholds this result in the normal flow, so the model does not
            # see this note and has no turn to fill. It is here for the paths that
            # do deliver it, and because a result that says nothing about what to do
            # next is worse than one that does.
            "note": "The caller has already been told the code is coming and how to "
                    "give it back. Wait for the digits. Never say the code yourself.",
        }

    def _sms_otp(self) -> bool:
        try:
            from twilio.rest import Client

            Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
                to=self.caller_number,
                from_=TWILIO_FROM_NUMBER,
                body=f"Your Amazon Pharmacy verification code is {self.otp}",
            )
            return True
        except Exception as e:  # demo must not die on SMS trouble
            logger.warning(f"[{self.label}] OTP SMS failed: {e}")
            return False

    async def _verify_otp(self, args: dict) -> dict:
        supplied = spoken.digits_from(args.get("code", ""))
        # The standing demo code has to work every single time it is spoken. It is
        # what the run sheet hands to anyone calling in, and on a demo the cost of
        # a false reject is the whole call -- a live one burned an attempt and made
        # the caller read a code twice on camera.
        #
        # The exact match below already accepts it, but only at exactly six digits:
        # one stray digit from the recogniser ("0936722", "9367228") failed the
        # length gate before it was ever compared. So look for the demo code inside
        # whatever was heard, BEFORE that gate. This widening applies only to
        # DEMO_OTP, never to the per-session code, which still needs an exact match.
        if DEMO_OTP and DEMO_OTP in supplied and not self.locked:
            self.verify_attempts += 1
            self.verified = True
            if supplied != DEMO_OTP:
                logger.info(
                    f"[{self.label}] demo code found inside {supplied!r}; accepting"
                )
            return {
                "verified": True,
                "guidance": (
                    "Say the code is correct and move straight to what they called "
                    "about. Do not repeat the digits back. This caller is now "
                    "verified for their own account only."
                ),
            }
        if len(supplied) != 6:
            recovered = spoken.digits_from(self.latest_user_text)
            if len(recovered) == 6:
                logger.info(
                    f"[{self.label}] recovered six-digit OTP from latest caller "
                    "transcript after incomplete tool arguments"
                )
                supplied = recovered
        self.verify_attempts += 1
        if self.locked:
            return {"verified": False, "attempts_remaining": 0, "locked": True,
                    "message": "Already locked out. Close the call with end_call."}
        if not self.otp:
            return {
                "verified": False,
                "error": "no_code_sent",
                "message": (
                    "No verification code has been sent yet. Call send_otp first. "
                    "A name or date of birth is not a code. Keep verifying."
                ),
            }
        if len(supplied) != 6:
            # A caller once read six digits cleanly and was told "the number you gave
            # me has more than six digits", twice, because the model counted them
            # itself and got it wrong. This branch is the only place that verdict is
            # allowed to come from, and it never says the length out loud.
            #
            # The three cases are genuinely different, and collapsing them into one
            # "ask again" cost an extra spoken turn on 11% of calls: the common
            # shape is "Sure, the code I received is nine three six seven two two",
            # where the digits land last and a partial transcript arrives while the
            # caller is still talking. Asking them to start over there is wrong --
            # the rest of the code is already on its way.
            partial = 1 <= len(supplied) <= 5
            return {
                "verified": False,
                "error": "not_a_code",
                "attempts_remaining": self.otp_attempts_remaining,
                "digits_heard": len(supplied),
                "message": (
                    (
                        "That is the beginning of a code, not all of it, so the caller is "
                        "probably still speaking. Say NOTHING and wait for the rest of "
                        "the digits, then call verify_otp again with the whole code. Do "
                        "not ask them to repeat it and do not comment on what you heard."
                        if partial else
                        "No code was in that at all -- a name, a date of birth or a phone "
                        "number is not a code. Ask once for the six digit code you sent."
                        if not supplied else
                        "That carried more numbers than a code. Ask them to say just the "
                        "six digits, one at a time."
                    )
                    + " Do not count this as a failed try, do not tell the caller how many "
                      "digits you heard, do not read the digits back, and never announce a "
                      "length verdict yourself -- only this tool counts digits."
                ),
            }
        accepted = {c for c in (self.otp, DEMO_OTP) if c}
        if supplied and supplied in accepted:
            self.verified = True
            if supplied == DEMO_OTP and supplied != self.otp:
                logger.info(f"[{self.label}] verified with the standing demo code")
            return {
                "verified": True,
                "guidance": (
                    "Say the code is correct and move straight to what they called "
                    "about. Do not repeat the digits back. This caller is now verified "
                    "for their own account only."
                ),
            }
        self.otp_attempts_remaining -= 1
        left = max(self.otp_attempts_remaining, 0)
        out = {"verified": False, "attempts_remaining": left}
        if left == 0:
            self.locked = True
            out["locked"] = True
            out["message"] = (
                "That was the second failure. Apologise, tell the caller they can verify "
                "in the Amazon app or on the website, and say they can disconnect whenever "
                "they like. Then call end_call with reason verification_failed. Do not "
                "offer another code."
            )
        return out

    # The caller asking what strengths exist, or naming one themselves.
    _ASKS_STRENGTH = re.compile(
        r"\b(strength|strengths|dose|doses|dosage|milligram|milligrams|mg|microgram|"
        r"micrograms|mcg|how many|which one|what (do|does) (you|it) come in)\b|"
        r"\d+(\.\d+)?\s*(mg|mcg|milligram|microgram)", re.I)

    # The caller saying they already have a prescription, or asking to fill one.
    # These turns must not be answered with "which strength?" -- the strength is
    # on file, and CHECK_ACCOUNT_FIRST says to go and look.
    _HAS_RX = re.compile(
        r"\b(prescription|prescribed|prescriber|refill|refills|fill|filled|on file|"
        r"already (sent|have)|(doctor|physician|endocrinologist|provider) (has |already )?sent)\b",
        re.I)

    async def _lookup_medication(self, args: dict) -> dict:
        res = catalog.lookup(args.get("name", ""))
        # "Do you carry Zepbound?" is a yes/no question, and the answer came back
        # as six strengths, the prescription requirement and the day-supply list --
        # 61 words for a question that needed about eight. BE_BRIEF alone did not
        # beat it, because strengths_spoken is right there in the payload and a
        # model reads what it is given. So say what THIS turn asked for.
        if res.get("match") and isinstance(res.get("guidance"), list):
            spoke = spoken.numeric_from(self.latest_user_text or "")
            if not self._ASKS_STRENGTH.search(spoke) and not self._HAS_RX.search(spoke):
                if res.get("in_stock", True):
                    res["guidance"].insert(1, (
                        "The caller asked whether we carry this, not what strengths it comes "
                        "in. Say we carry it and ask which strength they need. Do not read the "
                        "strength list, the day supplies, or the prescription requirement -- "
                        "they are in this payload for the turn that asks for them."
                    ))
                else:
                    # Said unconditionally, this line answered "do you carry Wegovy?"
                    # with "we do carry Wegovy -- which strength do you need?" and left
                    # the caller to discover it was out of stock by asking. Availability
                    # is the answer to an availability question.
                    res["guidance"].insert(1, (
                        "The caller asked whether we can supply this, and we cannot -- it "
                        "is OUT OF STOCK. Lead with that in your first sentence. Do not ask "
                        "which strength they want and do not read the strength list; there "
                        "is nothing to choose between. Offer a pharmacist follow-up, and "
                        "never suggest a different medication."
                    ))
        return res

    async def _search_knowledge_base(self, args: dict) -> dict:
        return knowledge.search(args.get("question", ""))

    def _apply_fills(self, payload: dict) -> dict:
        """Show what this call has already done, over the top of the stored account.

        Applied to ORDERS as well as prescriptions, and that is not tidiness: asked
        to check a prescription's status, the model called get_account_info with
        topic "orders", found nothing about it there, and reported the Zepbound as
        "waiting on your prescriber" -- which is Lisinopril's status, invented to
        fill the gap. Filling something really does create an order, so putting it
        in both places answers the question whichever topic the model reaches for.
        """
        if not self.filled:
            return payload

        orders = payload.get("orders")
        if orders is not None:
            for med, _left in self.filled.items():
                name = self._filled_display.get(med, med.title())
                orders.insert(0, {
                    "status": "being filled",
                    "placed": "just now, on this call",
                    "arrives": "free two-day delivery once it ships",
                    "items": [name],
                    "delivery_speed": "two day",
                    "tracking": "not yet shipped; tracking follows by text and email",
                    "note": None,
                })
            payload["spoken_summary"] = knowledge.orders_summary(orders)
            payload["guidance"] = list(payload.get("guidance", [])) + [
                "The order marked 'being filled' was placed on this call. If the "
                "caller asks about that medication, this is its status -- do not "
                "describe it as waiting on a prescriber.",
            ]

        rows = payload.get("prescriptions")
        if not rows:
            return payload
        for r in rows:
            left = self.filled.get(r["medication"].lower())
            if left is None:
                continue
            r["status"] = "being filled"
            r["refills_remaining"] = left
            r["next_refill"] = "being filled now, shipping with free two-day delivery"
            r["filled_on_this_call"] = True
        if "refillable_now" in payload:
            payload["refillable_now"] = [
                m for m in payload["refillable_now"] if m.lower() not in self.filled
            ]
            payload["refillable_spoken"] = knowledge.refillable_summary(
                payload["refillable_now"], any_filled=True
            )
        payload["spoken_summary"] = knowledge.prescriptions_summary(rows)
        return payload

    async def _get_account_info(self, args: dict) -> dict:
        if not self.verified:
            return {"error": "caller is not verified yet; complete verification first"}
        return self._apply_fills(knowledge.account_info(args.get("topic", "profile")))

    # A day supply the caller actually said, e.g. "for ninety days" -> 90. Spoken
    # numbers are folded to digits first, so "ninety" and "90" both land.
    _SUPPLY_RE = re.compile(r"(\d{1,3})\s*(?:-|\s)?\s*day", re.I)

    def _requested_supply(self, fallback: int | None) -> int | None:
        """The supply the CALLER asked for, when the model quietly pre-converted it.

        catalog.price dispenses a 90-day request as 84 and returns both numbers, and
        va.py uses the difference to make sure the caller is told. That only works if
        the server is the one doing the conversion. gpt-4o-mini does it itself --
        asked for ninety days it calls check_price with day_supply 84 -- so the
        payload comes back with requested == dispensed, nothing looks converted, and
        the caller is quoted 84 days for the 90 they asked for without ever being
        told why. 3 of 3 runs.

        Narrow on purpose: only when the caller named a supply that this catalog
        would convert INTO exactly the number the model sent. Any other
        disagreement is left alone -- the model may well have better context than a
        regex over one transcript line.
        """
        said = self._SUPPLY_RE.search(spoken.numeric_from(self.latest_user_text or ""))
        if not said or fallback is None:
            return None
        want = int(said.group(1))
        if want == fallback or catalog.dispensed_supply(want) != fallback:
            return None
        return want

    # The caller raising money on the turn that asks for the fill.
    _ASKS_MONEY = re.compile(
        r"\b(coupon|coupons|discount|savings|save|price|cost|costs|how much|"
        r"cheaper|copay|co-pay|insurance)\b", re.I)

    async def _fill_prescription(self, args: dict) -> dict:
        if not self.verified:
            return {"error": "caller is not verified yet; complete verification first"}
        name = args.get("name", "")
        res = knowledge.fill_prescription(name)
        # Guidance alone could not stop this being volunteered: told plainly not to
        # mention price unless asked, the model still answered "please fill my
        # Zepbound" with "a coupon is applied, and it will be as low as twelve
        # dollars and fifty cents". Data in the payload beats an instruction against
        # using it -- the same lesson as the medication list -- so the money only
        # travels on the turns that asked about money.
        if not self._ASKS_MONEY.search(self.latest_user_text or ""):
            for k in ("insurance_avg_price_spoken", "cash_price_spoken",
                      "coupon_savings_up_to_spoken", "as_low_as_with_coupon_spoken",
                      "coupon_available"):
                res.pop(k, None)
        med = (res.get("medication") or "").lower()
        # Asked twice, answer honestly the second time rather than filling again.
        if med and med in self.filled and not res.get("already_filled"):
            return {
                "filled": False,
                "already_filled": True,
                "medication": res.get("medication"),
                "guidance": [
                    f"{res.get('medication')} is already being filled on this call. "
                    "Say so plainly and do not fill it again. Do not treat this as an "
                    "error or apologise for it.",
                ],
            }
        if res.get("filled") and med:
            self.filled[med] = res.get("refills_remaining_after", 0)
            self._filled_display[med] = res.get("medication") or med.title()
            logger.info(
                f"[{self.label}] filled {res.get('medication')} this call; "
                f"{self.filled[med]} refills left (session only, account.json untouched)"
            )
        return res

    # A strength the caller actually said. The unit is REQUIRED, so "sixty days"
    # and "ninety day supply" cannot be mistaken for a dose.
    _STRENGTH_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|milligrams?|micrograms?|g|grams?|ml|"
        r"milliliters?|units?|meq|%|percent)\b", re.I)

    def _requested_strength(self, sent: str) -> str | None:
        """The strength the CALLER said, when the model quietly corrected it.

        Asked for "Zepbound at twelve milligrams", the model called check_price with
        "twelve point five milligrams" -- a strength we do carry, and not the one the
        caller named. The catalog already refuses an unknown strength and reads back
        the real list, but it never saw "12 mg", so the caller was quoted a price for
        a dose they had not asked for and never told. Silent dose substitution is the
        worst answer a pharmacy line can give.

        Same recovery as the day supply above: prefer what the caller said, and let
        the catalog decide whether it exists. Only fires when the two genuinely
        disagree, so the ordinary case where the model simply tidies "twelve point
        five" into "12.5 mg" passes straight through.
        """
        heard = self._STRENGTH_RE.search(spoken.numeric_from(self.latest_user_text or ""))
        if not heard:
            return None
        said = f"{heard.group(1)} {heard.group(2)}"
        if catalog._strength_key(said) == catalog._strength_key(sent or ""):
            return None                      # same dose, differently written
        return said

    async def _check_price(self, args: dict) -> dict:
        supply = args.get("day_supply", 30)
        try:
            asked = self._requested_supply(int(supply))
        except (TypeError, ValueError):
            asked = None
        if asked is not None:
            logger.info(
                f"[{self.label}] check_price: caller asked for {asked} days but the model "
                f"sent {supply}; using {asked} so the conversion is disclosed"
            )
            supply = asked
        strength = args.get("strength", "")
        heard = self._requested_strength(strength)
        if heard is not None:
            logger.info(
                f"[{self.label}] check_price: caller said {heard!r} but the model sent "
                f"{strength!r}; using the caller's so an unknown strength is caught"
            )
            strength = heard
        return catalog.price(args.get("name", ""), strength, supply)
