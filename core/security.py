"""
core/security.py
================
The single choke-point for **input sanitisation**, **prompt-injection detection** and
**rate limiting** in the CAT Smart Operator Assistant.

Threat model (edge node, untrusted operator console / CAN-CSV feed)
-------------------------------------------------------------------
================================ ===============================================
Threat                            Mitigation in this module
================================ ===============================================
Reflected / stored XSS            :func:`sanitize_text` strips tags, control chars and
                                  ``javascript:``-style URIs; :func:`escape_for_display`
                                  HTML-escapes as defence in depth. The UI never calls
                                  ``st.markdown(..., unsafe_allow_html=True)``.
Prompt injection / jailbreak      :func:`detect_prompt_injection` + the Gatekeeper in
                                  :mod:`core.copilot`; the system prompt treats the
                                  operator query as inert data only.
Data-ingestion injection (CSV)    :func:`validate_sensor_value` / :func:`coerce_numeric`
                                  force strict types and bounds on every field.
DoS (huge paste, chat flood)      character caps, :class:`RateLimiter`, row caps.
Log forging / ANSI smuggling      control characters (including ``\\x1b``) are removed and
                                  newlines collapsed in log-bound strings.
Weak crypto / secrets in code     SHA-256 for any fingerprinting; secrets only ever read
                                  from ``st.secrets`` / environment at runtime.
================================ ===============================================

Everything here is pure and dependency-light (stdlib + ``re``) so it can be unit-tested
without Streamlit, pandas or a network.
"""

from __future__ import annotations

import hashlib
import html
import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence

# --------------------------------------------------------------------------------------
# Regular expressions (compiled once, no catastrophic backtracking - all linear)
# --------------------------------------------------------------------------------------

#: C0/C1 control characters except tab/newline (removed from operator text).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

#: Any HTML-ish tag, including malformed/partial ones such as ``<img src=x onerror=...``.
_HTML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>?", re.DOTALL)

#: Dangerous URI schemes that may survive tag stripping (``javascript:alert(1)``).
_DANGEROUS_URI_RE = re.compile(
    r"(?i)\b(?:javascript|vbscript|data|file|blob|jar)\s*:", re.IGNORECASE
)

#: Event-handler fragments (``onerror=``, ``onload=``) that are meaningless without tags.
_EVENT_HANDLER_RE = re.compile(r"(?i)\bon[a-z]{3,15}\s*=")

#: Residual tag opener: a ``<`` immediately followed by optional ``/``/whitespace and a
#: letter. Used *after* tag stripping, because removing an inner tag can re-form an outer
#: one (``<<script>script>`` -> ``<script>``) - the classic stripper-mutation bypass. Only
#: the scaffolding opener is neutralised, so legitimate text such as ``rpm < 1000`` or
#: ``3 > 2`` is preserved.
_RESIDUAL_TAG_RE = re.compile(r"<\s*/?\s*(?=[A-Za-z])")

#: Maximum fixed-point tag-stripping passes (payloads nest a handful of levels at most).
_MAX_TAG_PASSES = 5

#: Collapse 3+ newlines / long runs of whitespace (anti log-flooding).
_WHITESPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
_NEWLINE_RUN_RE = re.compile(r"\n{3,}")

# --- prompt-injection signatures -------------------------------------------------------
#: Each entry is (rule-id, compiled regex). Ordered most-specific first. Detections are
#: *deterministic and explainable* - the audit log records the rule id that fired.
INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "INJ_IGNORE_INSTRUCTIONS",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override|discard)\b[^.\n]{0,40}?"
            r"\b(?:previous|prior|above|earlier|all|any|the)\b[^.\n]{0,20}?"
            r"\b(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions|context)\b"
        ),
    ),
    (
        "INJ_REVEAL_PROMPT",
        re.compile(
            r"(?i)\b(?:reveal|show|print|dump|repeat|expose|display|tell\s+me|what\s+is|whats)\b"
            r"[^.\n]{0,40}?\b(?:system\s*prompt|your\s*rules|initial\s*prompt|hidden\s*prompt|"
            r"instructions\s*above|prompt\s*above|your\s*configuration|developer\s*message)\b"
        ),
    ),
    (
        "INJ_ROLE_SWITCH",
        re.compile(
            r"(?i)\b(?:you\s+are\s+now|from\s+now\s+on\s+you|act\s+as|pretend\s+to\s+be|"
            r"role\s*play\s+as|simulate\s+being|behave\s+like|new\s+persona|"
            r"developer\s*mode|do\s*anything\s*now|\bDAN\b|jailbreak|sudo\s+mode|"
            r"unrestricted\s+mode|no\s+restrictions)\b"
        ),
    ),
    (
        "INJ_SYSTEM_TOKEN",
        re.compile(
            r"(?i)(?:<\|?\s*(?:system|assistant|im_start|im_end)\s*\|?>|\[\s*system\s*\]|"
            r"^\s*system\s*:|###\s*(?:system|instruction)|\{\{\s*system)"
        ),
    ),
    (
        "INJ_ENCODED_PAYLOAD",
        re.compile(r"(?i)\b(?:base64|rot13|hex\s*decode|eval\s*\(|exec\s*\(|\\x[0-9a-f]{2})"),
    ),
    (
        # Pattern A: defeating/tampering with a *safety device* or its sensor.
        "INJ_SECURITY_BYPASS",
        re.compile(
            r"(?i)\b(?:bypass|defeat|disable|circumvent|hotwire|short\s*out|override|trick|"
            r"fool|spoof|silence|mute|neutralis|neutraliz|rewire|shunt|unplug|disconnect|"
            r"remove)\b[^.\n]{0,48}?"
            r"\b(?:interlock|safety|seatbelt|seat\s*belt|alarm|buzzer|guard|limiter|governor|"
            r"hydraulic\s*lock|proximity|kill\s*switch|e-?stop|emergency\s*stop|sensor|"
            r"monitor|immobilis\w*|immobiliz\w*)\b"
        ),
    ),
    (
        # Pattern B: the same "defeat" verbs aimed at the machine itself
        # ("hotwire the machine"). Kept narrow (no remove/disconnect/unplug) so legitimate
        # electrical work - "disconnect the battery for lock out" - stays allowed.
        "INJ_SECURITY_BYPASS",
        re.compile(
            r"(?i)\b(?:bypass|defeat|disable|circumvent|hotwire|short\s*out|override|trick|"
            r"fool|spoof|silence|mute|neutralis|neutraliz)\b[^.\n]{0,40}?"
            r"\b(?:machine|engine|starter|ignition|key)\b"
        ),
    ),
    (
        # Pattern C: skipping the isolation procedure itself. "skip"/"avoid"/"ignore" only
        # appear here, so sentences like "disconnect the battery for lock out tag out" (a
        # correct LOTO step) are not flagged, while "skip the lockout procedure" is.
        "INJ_SECURITY_BYPASS",
        re.compile(
            r"(?i)\b(?:bypass|defeat|disable|circumvent|skip|avoid|ignore|override|get\s*around)"
            r"\b[^.\n]{0,40}?"
            r"\b(?:lock\s*out|tag\s*out|loto|isolation|lockout)\b"
        ),
    ),
)

#: Instructions that ask for content that is out of the machinery domain entirely.
OUT_OF_SCOPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "OOS_CREATIVE",
        re.compile(
            r"(?i)\b(?:write|compose|recite|sing|generate|make\s+up|tell)\b[^.\n]{0,25}?"
            r"\b(?:poem|song|lyrics|rap|haiku|limerick|joke|story|novel|essay|screenplay|"
            r"meme|riddle)\b"
        ),
    ),
    (
        "OOS_GENERAL_KNOWLEDGE",
        re.compile(
            r"(?i)\b(?:weather\s+forecast|stock\s+price|share\s+market|crypto|bitcoin|"
            r"horoscope|astrology|recipe|cook|medication|dosage|diagnose|symptoms|"
            r"visa|passport|flight\s+booking|movie|netflix|cricket\s+score|"
            r"homework|translate\s+this|"
            r"python|javascript|c\+\+|golang|rust|linked[- ]*list|binary[- ]*tree|leetcode|data\s+structure|programming|"
            r"(?:write|give\s+me|generate|provide|create)\b[^.\n]{0,25}?\b(?:code|script|program|algorithm|function)\b)\b"
        ),
    ),
    (
        "OOS_NETWORK_ABUSE",
        re.compile(
            r"(?i)\b(?:sql\s*injection|xss|csrf|ransomware|keylogger|phishing|"
            r"ddos|botnet|malware|exploit\s+kit|reverse\s+shell|nmap|metasploit|"
            r"crack\s+the\s+password|steal\s+(?:credentials|tokens))\b"
        ),
    ),
)


# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SanitizationReport:
    """Audit record describing what a sanitisation pass did to one piece of input."""

    original_length: int
    final_length: int
    truncated: bool
    removed_control_chars: int
    removed_tags: int
    dangerous_uri_hits: int
    event_handler_hits: int
    empty_after_sanitize: bool
    note: str = ""

    @property
    def modified(self) -> bool:
        """True when the sanitiser changed the payload in any way."""
        return (
            self.truncated
            or self.removed_control_chars > 0
            or self.removed_tags > 0
            or self.dangerous_uri_hits > 0
            or self.event_handler_hits > 0
        )

    def describe(self) -> str:
        """Human-readable one-liner for the security audit panel (never echoed raw)."""
        if not self.modified:
            return f"clean ({self.final_length} chars)"
        bits: list[str] = []
        if self.removed_tags:
            bits.append(f"{self.removed_tags} tag(s) stripped")
        if self.dangerous_uri_hits:
            bits.append(f"{self.dangerous_uri_hits} dangerous URI scheme(s)")
        if self.event_handler_hits:
            bits.append(f"{self.event_handler_hits} event attribute(s)")
        if self.removed_control_chars:
            bits.append(f"{self.removed_control_chars} control char(s)")
        if self.truncated:
            bits.append(f"truncated to {self.final_length} chars")
        return ", ".join(bits) if bits else "modified"


@dataclass
class SanitizedInput:
    """Sanitised text plus its audit trail."""

    text: str
    report: SanitizationReport
    allowed_charset: bool = True
    injection_rules: tuple[str, ...] = ()
    out_of_scope_rules: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """True when the payload survived sanitisation untouched."""
        return not self.report.modified

    @property
    def suspicious(self) -> bool:
        """True when any injection or out-of-scope rule fired."""
        return bool(self.injection_rules or self.out_of_scope_rules)


# --------------------------------------------------------------------------------------
# Core string sanitisation
# --------------------------------------------------------------------------------------
def strip_control_chars(text: str) -> tuple[str, int]:
    """
    Remove C0/C1 control characters (keeps ``\\n`` and ``\\t``).

    Blocks ANSI-escape smuggling into terminals and log-forging via ``\\r``.
    """
    cleaned = _CONTROL_CHARS_RE.sub("", text)
    return cleaned, len(text) - len(cleaned)


def normalize_unicode(text: str) -> str:
    """
    NFKC-normalise input.

    Defeats homoglyph / fullwidth-character filter evasion (e.g. ``＜script＞`` or
    ``ｉｇｎｏｒｅ  ｐｒｅｖｉｏｕｓ``) by folding it to canonical ASCII-compatible form.
    """
    return unicodedata.normalize("NFKC", text)


def sanitize_text(
    raw: object,
    max_chars: int = 500,
    allow_newlines: bool = False,
) -> SanitizedInput:
    """
    Sanitise a single untrusted string for storage, display **and** prompt assembly.

    Pipeline: ``NFKC`` -> control-char strip -> HTML/XML tag strip -> dangerous-URI strip
    -> event-attribute strip -> whitespace normalise -> length cap.

    Parameters
    ----------
    raw:
        Any object; coerced with ``str()`` (never ``eval``-ed, never unpickled).
    max_chars:
        Hard character cap (DoS protection). Must be >= 1.
    allow_newlines:
        When ``False`` (default) the payload is flattened onto one line, so a single chat
        turn can never be used to forge multi-line log entries.

    Returns
    -------
    SanitizedInput
        The cleaned text (never ``None``) plus a :class:`SanitizationReport` for the audit
        trail. The function never raises on hostile input - it degrades to ``""``.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    original = "" if raw is None else str(raw)
    original_length = len(original)

    text = normalize_unicode(original)
    text, removed_control = strip_control_chars(text)

    # Tag stripping runs to a fixed point: one pass alone is bypassable by nesting, e.g.
    # ``<<script>script>alert(1)<</script>/script>`` leaves a working ``<script>`` behind.
    n_tags = 0
    for _ in range(_MAX_TAG_PASSES):
        text, removed = _HTML_TAG_RE.subn("", text)
        n_tags += removed
        if removed == 0:
            break

    # No `<` followed by a letter may survive (it could open a tag), while comparisons such
    # as ``rpm < 1000`` keep their meaning.
    text, n_residual = _RESIDUAL_TAG_RE.subn("‹", text)
    n_tags += n_residual

    text, n_uris = _DANGEROUS_URI_RE.subn("", text)
    text, n_handlers = _EVENT_HANDLER_RE.subn("", text)

    if allow_newlines:
        text = _WHITESPACE_RE.sub(" ", text)
        text = _NEWLINE_RUN_RE.sub("\n\n", text)
    else:
        text = _WHITESPACE_RE.sub(" ", text.replace("\r", " ").replace("\n", " "))
    text = text.strip()

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip()

    report = SanitizationReport(
        original_length=original_length,
        final_length=len(text),
        truncated=truncated,
        removed_control_chars=removed_control,
        removed_tags=n_tags,
        dangerous_uri_hits=n_uris,
        event_handler_hits=n_handlers,
        empty_after_sanitize=not text,
    )
    return SanitizedInput(text=text, report=report)


def escape_for_display(text: object) -> str:
    """
    HTML-escape a string before it is handed to any Markdown/HTML renderer.

    Defence in depth: the UI never enables ``unsafe_allow_html``, but if a future
    contributor does, user data is still inert.
    """
    return html.escape("" if text is None else str(text), quote=True)


def flatten_for_log(text: object, max_chars: int = 200) -> str:
    """Flatten + escape a value for inclusion in an audit/log line (anti log-forging)."""
    return escape_for_display(sanitize_text(text, max_chars=max_chars).text)


# --------------------------------------------------------------------------------------
# Signature detection
# --------------------------------------------------------------------------------------
def _match_rules(
    text: str, rules: Sequence[tuple[str, re.Pattern[str]]]
) -> tuple[str, ...]:
    """
    Return the ids of every rule matched by ``text`` (deterministic order, de-duplicated).

    Several patterns may share one rule id (e.g. the two ``INJ_SECURITY_BYPASS`` variants),
    so the id is reported once even when more than one variant fires.
    """
    matched: list[str] = []
    for rule_id, pattern in rules:
        if pattern.search(text) and rule_id not in matched:
            matched.append(rule_id)
    return tuple(matched)


def detect_prompt_injection(text: str) -> tuple[str, ...]:
    """Return the ids of every prompt-injection / jailbreak rule matched by ``text``."""
    return _match_rules(normalize_unicode(text or ""), INJECTION_RULES)


def detect_out_of_scope(text: str) -> tuple[str, ...]:
    """Return the ids of every out-of-domain intent rule matched by ``text``."""
    return _match_rules(normalize_unicode(text or ""), OUT_OF_SCOPE_RULES)


def contains_script_payload(text: str) -> bool:
    """
    Cheap boolean check used by the ingestion path and security tests.

    Detects script tags, event handlers, ``javascript:`` URIs and entity-encoded
    variants *before* sanitisation.
    """
    if not text:
        return False
    probe = normalize_unicode(str(text))
    if _HTML_TAG_RE.search(probe):
        return True
    if _DANGEROUS_URI_RE.search(probe):
        return True
    if _EVENT_HANDLER_RE.search(probe):
        return True
    return bool(
        re.search(r"(?i)&(?:lt|#0*60|#x0*3c)\s*;?\s*script", probe)
        or re.search(r"(?i)<\s*script", probe)
    )


def validate_sensor_value(value: object) -> tuple[bool, object]:
    """
    Validate a raw sensor value read from a CSV/JSON feed.

    Returns ``(is_valid, coerced)`` where ``coerced`` is ``int``/``float``/``bool`` for
    valid input and ``None`` otherwise. Strings are *never* passed through: a payload such
    as ``"1200; DROP TABLE machines;--"`` fails to parse and becomes ``None`` (the caller
    records a sensor fault) - nothing is ever executed or interpolated.
    """
    if value is None:
        return False, None
    if isinstance(value, bool):
        return True, bool(value)
    if isinstance(value, int):
        return True, float(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return False, None
        if value in (float("inf"), float("-inf")):
            return False, None
        return True, value
    if isinstance(value, str):
        probe = value.strip()
        # Allow a single numeric token only - no units, no separators, no expressions.
        if not re.fullmatch(r"[+-]?\d{1,12}(?:\.\d{1,6})?", probe):
            return False, None
        try:
            return True, float(probe)
        except (TypeError, ValueError):  # pragma: no cover - regex already guarantees this
            return False, None
    return False, None


def coerce_numeric(value: object, hard_min: float | None = None,
                   hard_max: float | None = None) -> float | None:
    """
    Coerce + bound-check one numeric field.

    ``None`` is returned for unparsable **or** physically impossible values, which is the
    signal the ingestion layer turns into a ``SENSOR_FAULT`` flag.
    """
    ok, coerced = validate_sensor_value(value)
    if not ok or coerced is None:
        return None
    result = float(coerced)
    if hard_min is not None and result < hard_min:
        return None
    if hard_max is not None and result > hard_max:
        return None
    return result


# --------------------------------------------------------------------------------------
# DoS protection: rate limiting + fingerprinting
# --------------------------------------------------------------------------------------
@dataclass
class RateLimitDecision:
    """Outcome of a :meth:`RateLimiter.check` call."""

    allowed: bool
    remaining: int
    retry_after_s: float
    reason: str = ""


@dataclass
class RateLimiter:
    """
    Sliding-window rate limiter (chat flood protection).

    Time is injected through ``clock`` so tests are fully deterministic - no ``sleep``.
    """

    max_events: int = 20
    window_s: float = 60.0
    clock: object = time.monotonic
    _events: Deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self.max_events = max(1, int(self.max_events))
        self.window_s = max(0.001, float(self.window_s))

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def check(self) -> RateLimitDecision:
        """Register an attempt and report whether it is within budget."""
        now = float(self.clock())  # type: ignore[operator]
        self._evict(now)
        if len(self._events) >= self.max_events:
            retry = max(0.0, self.window_s - (now - self._events[0]))
            return RateLimitDecision(
                allowed=False,
                remaining=0,
                retry_after_s=round(retry, 2),
                reason=f"rate limit: {self.max_events} messages / {self.window_s:.0f}s exceeded",
            )
        self._events.append(now)
        return RateLimitDecision(
            allowed=True, remaining=self.max_events - len(self._events), retry_after_s=0.0
        )

    def reset(self) -> None:
        """Clear the window (called by the session reset button)."""
        self._events.clear()


def fingerprint(value: object, salt: str = "") -> str:
    """
    Non-reversible SHA-256 fingerprint for audit correlation (e.g. operator id).

    SHA-256 is used deliberately: MD5/SHA-1 are *never* used anywhere in this codebase
    (bandit ``B303``/``B324`` clean).
    """
    payload = f"{salt}|{'' if value is None else value}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


def redact_secret(value: object, keep: int = 4) -> str:
    """Redact a credential for display: ``sk-…9f2a`` (never log a full key)."""
    text = "" if value is None else str(value)
    if not text:
        return "<unset>"
    if len(text) <= keep:
        return "*" * len(text)
    return f"{text[:3]}…{text[-keep:]}"


def audit_summary(fields: Iterable[tuple[str, object]]) -> str:
    """Build a flattened ``k=v`` audit line from (already sanitised) key/value pairs."""
    return " ".join(f"{key}={flatten_for_log(value, 120)}" for key, value in fields)


__all__ = [
    "SanitizationReport",
    "SanitizedInput",
    "sanitize_text",
    "escape_for_display",
    "flatten_for_log",
    "normalize_unicode",
    "strip_control_chars",
    "detect_prompt_injection",
    "detect_out_of_scope",
    "contains_script_payload",
    "validate_sensor_value",
    "coerce_numeric",
    "RateLimiter",
    "RateLimitDecision",
    "fingerprint",
    "redact_secret",
    "audit_summary",
    "INJECTION_RULES",
    "OUT_OF_SCOPE_RULES",
]
