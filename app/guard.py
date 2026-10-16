"""Task 3 - Intelligent feature: input scope guard + LLM reply validator.

Two cheap, deterministic safety layers around the Task 1 model and the Gemini drafter:

1. ``check_input``  - catches messages the closed-set classifier would otherwise auto-route by mistake
   (third-party products such as "reset my netflix password", non-bank objects such as "my laptop is
   broken", and prompt-injection attempts). It can only make routing MORE conservative: an
   AUTO_ROUTE can be downgraded to HUMAN_REVIEW, never the reverse.
2. ``validate_reply`` - checks a drafted reply against the same rules the system prompt asks for
   (no PIN/CVV requests, no promises, <= 70 words, no leaked system prompt, no PII). A reply that fails
   is replaced by the safe template, and the reasons are reported in ``guard_flags``.
"""
from __future__ import annotations

import re

from .privacy import redact

MAX_REPLY_WORDS = 70
MIN_REPLY_WORDS = 5

# Third-party services / non-bank objects: if the message is about these, a bank intent is a false friend
# ("password" -> passcode_forgotten, "broken" -> card_not_working, "email" -> edit_personal_details).
_THIRD_PARTY = (
    "netflix spotify instagram facebook whatsapp twitter tiktok snapchat youtube gmail outlook yahoo "
    "amazon ebay uber airbnb steam playstation xbox laptop computer phone-case television tv fridge "
    "microwave bicycle recipe pizza weather football cricket homework movie song paypal wifi wi-fi router "
    "tinder dishwasher licence license"
).split()
_THIRD_PARTY_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _THIRD_PARTY) + r")\b", re.I)

# Words that anchor a message in the banking domain; a third-party word next to one of these is allowed
# ("my card was charged by netflix twice" is a real banking complaint).
_BANK_ANCHORS = re.compile(
    r"\b(card|charged|charge|payment|transfer|account|balance|bank|atm|cash|refund|top.?up|statement|"
    r"transaction|direct debit|exchange rate|pin|passcode|iban|swift|withdraw\w*|deposit\w*)\b",
    re.I,
)

_PASSWORD_RE = re.compile(r"\bpass ?words?\b", re.I)

_INJECTION = [
    re.compile(p, re.I)
    for p in (
        r"ignore (all |any |your )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
        r"(reveal|show|print|repeat) (me )?(your |the )?(system )?(prompt|instructions)",
        r"you are now\b",
        r"\b(developer|jailbreak|dan) mode\b",
        r"disregard (your|the) (rules|guidelines|instructions)",
        r"pretend (to be|you are)",
    )
]


def check_input(text: str, result: dict) -> list[str]:
    """Return a list of guard flags. Non-empty => the caller must route to a human."""
    flags: list[str] = []
    if any(p.search(text) for p in _INJECTION):
        flags.append("PROMPT_INJECTION")
    if _THIRD_PARTY_RE.search(text) and not _BANK_ANCHORS.search(text):
        flags.append("OUT_OF_SCOPE_TOPIC")
    # Banks in this domain say "passcode"/"PIN"; a bare "password" almost always means some other service.
    elif _PASSWORD_RE.search(text) and not _BANK_ANCHORS.search(text):
        flags.append("OUT_OF_SCOPE_TOPIC")
    # Very short messages with a confident bank intent are fine ("refund"); a tiny margin means the
    # model is guessing between intents even if the top score looks moderate.
    top = result["top_k"]
    if len(top) > 1 and (top[0]["confidence"] - top[1]["confidence"]) < 0.05 and result["decision"] == "AUTO_ROUTE":
        flags.append("LOW_MARGIN")
    return flags


def apply_input_guard(text: str, result: dict) -> dict:
    """Return ``result`` with ``guard_flags`` added and AUTO_ROUTE downgraded when a flag fired."""
    flags = check_input(text, result)
    out = {**result, "guard_flags": flags}
    if flags and out["decision"] == "AUTO_ROUTE":
        out["decision"] = "HUMAN_REVIEW"
        out["downgraded_by_guard"] = True
    return out


_PIN_ASK = re.compile(
    r"\b(send|share|give|provide|tell|enter|confirm|reply with|type)\b[^.?!]{0,40}"
    r"\b(pin|password|passcode|cvv|cvc|one[- ]time (code|password)|otp|full card number)\b",
    re.I,
)
_PROMISE = re.compile(
    r"\b(we (will|'ll) (refund|credit|reimburse|reverse)|you will (be )?(refunded|reimbursed|receive)|"
    r"guarantee[sd]?|within \d+ (business )?(days?|hours?)|by (tomorrow|monday|tuesday|wednesday|thursday|friday))\b",
    re.I,
)
_LEAK = re.compile(r"(system prompt|customer_message|my instructions|as an ai language model)", re.I)


def validate_reply(reply_text: str) -> list[str]:
    """Return a list of violations in a drafted reply (empty list = safe to show to an agent)."""
    flags: list[str] = []
    if len(reply_text.split()) < MIN_REPLY_WORDS:
        flags.append("TOO_SHORT")
    if len(reply_text.split()) > MAX_REPLY_WORDS:
        flags.append("TOO_LONG")
    if _PIN_ASK.search(reply_text):
        flags.append("ASKS_FOR_SECRET")
    if _PROMISE.search(reply_text):
        flags.append("MAKES_PROMISE")
    if _LEAK.search(reply_text):
        flags.append("PROMPT_LEAK")
    if redact(reply_text)[1]:
        flags.append("CONTAINS_PII")
    return flags
