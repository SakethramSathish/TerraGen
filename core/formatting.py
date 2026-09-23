"""
core/formatting.py
==================
Pure presentation helpers (zero Streamlit / pandas dependencies).

Keeping formatting out of ``app.py`` means:

* every label the operator sees can be unit-tested;
* the UI layer stays a thin shell (easy to port to a different front-end);
* **no helper ever returns HTML** - only plain text and emoji. Combined with the project
  rule that ``st.markdown(..., unsafe_allow_html=True)`` is never used, this closes the
  XSS surface by construction rather than by escaping discipline.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from core.security import escape_for_display

#: Indicator-level badges. Clean plain-text industrial badges (no emojis, never raw HTML).
LEVEL_BADGES: dict[str, str] = {
    "OK": "[OK]",
    "INFO": "[INFO]",
    "CAUTION": "[CAUTION]",
    "WARNING": "[WARNING]",
    "CRITICAL": "[CRITICAL]",
    "UNKNOWN": "[UNKNOWN]",
    "NO_DATA": "[NO DATA]",
}

#: Hex colours used for chart accents / progress bars (kept here so the palette is one file).
LEVEL_COLORS: dict[str, str] = {
    "OK": "#2E7D32",
    "INFO": "#1565C0",
    "CAUTION": "#F9A825",
    "WARNING": "#EF6C00",
    "CRITICAL": "#C62828",
    "UNKNOWN": "#616161",
}


def badge(level: str) -> str:
    """Return the plain-text badge for a severity/indicator level (no emojis)."""
    return LEVEL_BADGES.get(str(level).upper(), f"[{str(level).upper()}]")


def level_color(level: str) -> str:
    """Hex colour for a level (used for chart lines and progress bars)."""
    return LEVEL_COLORS.get(str(level).upper(), LEVEL_COLORS["UNKNOWN"])


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion that never raises (``None``/NaN -> ``default``)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def fmt_number(value: Any, digits: int = 0, unit: str = "", thousands: bool = True) -> str:
    """Format a number with optional unit - ``"-"`` when the value is unusable."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    text = f"{number:,.{digits}f}" if thousands else f"{number:.{digits}f}"
    return f"{text} {unit}".strip()


def fmt_tons(value: Any, digits: int = 0) -> str:
    """``1830`` -> ``"1,830 t"``."""
    return fmt_number(value, digits=digits, unit="t")


def fmt_tph(value: Any) -> str:
    """Format a tonnage rate (t/h)."""
    return fmt_number(value, digits=1, unit="t/h")


def fmt_litres(value: Any, digits: int = 1) -> str:
    """Format a volume in litres."""
    return fmt_number(value, digits=digits, unit="L")


def fmt_pct(value: Any, digits: int = 0) -> str:
    """Format a percentage (adds the ``%`` sign)."""
    text = fmt_number(value, digits=digits)
    return f"{text} %" if text != "-" else "-"


def fmt_hours(hours: Any) -> str:
    """
    Format a duration in hours as ``"2 h 15 min"``.

    Values are clamped at zero so a clock skew can never render ``"-1 h"``.
    """
    total = max(0.0, _coerce_float(hours))
    whole = int(total)
    minutes = int(round((total - whole) * 60))
    if minutes == 60:
        whole += 1
        minutes = 0
    if whole and minutes:
        return f"{whole} h {minutes:02d} min"
    if whole:
        return f"{whole} h"
    return f"{minutes} min"


def fmt_seconds(seconds: Any) -> str:
    """Format a duration in seconds as ``"1 h 05 min"`` / ``"4 min 20 s"``."""
    total = max(0.0, _coerce_float(seconds))
    if total >= 3600:
        return fmt_hours(total / 3600.0)
    if total >= 60:
        return f"{int(total // 60)} min {int(total % 60):02d} s"
    return f"{total:.0f} s"


def fmt_clock(moment: Any) -> str:
    """Format a datetime/timestamp as ``HH:MM`` (``"-"`` when absent)."""
    if moment is None:
        return "-"
    if isinstance(moment, str):
        try:
            moment = datetime.fromisoformat(moment)
        except ValueError:
            return escape_for_display(moment[:16])
    if isinstance(moment, (datetime, date)):
        return moment.strftime("%H:%M")
    to_pydatetime = getattr(moment, "to_pydatetime", None)
    if callable(to_pydatetime):
        return to_pydatetime().strftime("%H:%M")
    return str(moment)[:16]


def fmt_day(moment: Any) -> str:
    """Format a datetime as ``2026-09-23 14:05`` (``"-"`` when absent)."""
    if moment is None:
        return "-"
    if isinstance(moment, datetime):
        return moment.strftime("%Y-%m-%d %H:%M")
    to_pydatetime = getattr(moment, "to_pydatetime", None)
    if callable(to_pydatetime):
        return to_pydatetime().strftime("%Y-%m-%d %H:%M")
    return str(moment)[:16]


def fmt_delta(value: Any, unit: str = "") -> str:
    """Format a signed delta with an explicit sign (``"+120 t"`` / ``"-45 t"``)."""
    number = _coerce_float(value)
    text = f"{number:+,.0f}"
    return f"{text} {unit}".strip()


def truncate(text: Any, limit: int = 120) -> str:
    """Truncate plain text for compact display (adds an ellipsis)."""
    value = "" if text is None else str(text)
    limit = max(4, int(limit))
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def progress_ratio(part: Any, whole: Any) -> float:
    """
    Safe ratio clamped to ``0.0..1.0`` for ``st.progress``.

    Guards zero/negative/NaN denominators - a fat-fingered target of 0 must not raise.
    """
    numerator = _coerce_float(part)
    denominator = _coerce_float(whole)
    if denominator <= 0:
        return 0.0
    return float(min(1.0, max(0.0, numerator / denominator)))


def status_line(label: str, value: Any, level: str = "OK", note: str = "") -> str:
    """Compose a plain-text status line: ``"Seatbelt · ✅ OK · latched for 4 h"``."""
    parts = [str(label), badge(level)]
    if value not in (None, ""):
        parts.append(str(value))
    if note:
        parts.append(str(note))
    return " · ".join(parts)


def checklist_row(done: bool, text: str) -> str:
    """Render a checkbox-style row as plain text (no HTML)."""
    return f"{'☑' if done else '☐'} {text}"


__all__ = [
    "LEVEL_BADGES",
    "LEVEL_COLORS",
    "badge",
    "level_color",
    "fmt_number",
    "fmt_tons",
    "fmt_tph",
    "fmt_litres",
    "fmt_pct",
    "fmt_hours",
    "fmt_seconds",
    "fmt_clock",
    "fmt_day",
    "fmt_delta",
    "truncate",
    "progress_ratio",
    "status_line",
    "checklist_row",
]
