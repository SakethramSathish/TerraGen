"""
core/icons.py
=============
Curated industrial SVG icons for the CAT Smart Operator Assistant.
Replaces all emojis with crisp, vector-based SVG graphics matching
the Caterpillar industrial design system (Safety Yellow, Charcoal, Slate).
"""

from __future__ import annotations


def svg_icon(svg_markup: str, size: int = 18, color: str = "#FFCD11") -> str:
    """Wrap an inner SVG path with an inline SVG tag of given size and color."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align: middle; display: inline-block; margin-right: 6px;">'
        f'{svg_markup}'
        f'</svg>'
    )


# --- Category and Section Icons ---

# Heavy machinery / excavator
EXCAVATOR_SVG = (
    '<path d="M3 17h18M5 17v-4a2 2 0 0 1 2-2h4l3-4h4a2 2 0 0 1 2 2v8M7 21a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm10 0a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/>'
)

# Safety Guardian Shield
SHIELD_SVG = (
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
)

# Telemetry / Sensor Waves
GAUGE_SVG = (
    '<path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>'
)

# Copilot / AI Assistant
COPILOT_SVG = (
    '<rect x="4" y="4" width="16" height="16" rx="2"/><circle cx="9" cy="9" r="1.5"/><circle cx="15" cy="9" r="1.5"/><path d="M8 15s1.5 2 4 2 4-2 4-2"/>'
)

# Testing Suite / Stream Simulation
TEST_SUITE_SVG = (
    '<path d="M9 3v6l-4 8a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-4-8V3M8 3h8M6 14h12"/>'
)

# Security & Audit Lock
LOCK_SVG = (
    '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>'
)

# Proximity Radar
RADAR_SVG = (
    '<path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm0 0v8M4.93 4.93a10 10 0 0 0 0 14.14M19.07 4.93a10 10 0 0 1 0 14.14"/>'
)

# Seatbelt / Safety Latch
SEATBELT_SVG = (
    '<circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 14.14 14.14M12 8v8M8 12h8"/>'
)

# Hydraulic System / Pressure
HYDRAULIC_SVG = (
    '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3M8 3h8"/>'
)

# Engine System / Thermal
ENGINE_SVG = (
    '<rect x="4" y="8" width="16" height="10" rx="1"/><path d="M2 10v6M22 10v6M8 4h8M9 4v4M15 4v4"/>'
)

# --- Status and Indicator Icons ---

CHECK_SVG = (
    '<path d="M20 6 9 17l-5-5"/>'
)

ALERT_SVG = (
    '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3zM12 9v4M12 17h.01"/>'
)

DANGER_SVG = (
    '<polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
)

INFO_SVG = (
    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
)

# --- Functional Actions ---

CLIPBOARD_SVG = (
    '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'
)

TRASH_SVG = (
    '<path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M10 11v6M14 11v6"/>'
)

PLAY_SVG = (
    '<polygon points="5 3 19 12 5 21 5 3"/>'
)

BOLT_SVG = (
    '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'
)

MOON_SVG = (
    '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>'
)

SUN_SVG = (
    '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
)


def get_status_banner(level: str, headline: str = "") -> str:
    """Generate an accessible, industrial HTML banner with SVG icon."""
    if level == "OK":
        border_col = "#2E7D32"
        bg_col = "rgba(46, 125, 50, 0.15)"
        svg = svg_icon(CHECK_SVG, size=20, color="#4CAF50")
        label = "SYSTEM NORMAL · SAFE TO OPERATE"
        sub = "All sensor telemetry verified within safe thresholds. Zero active interlocks."
    elif level in {"WARNING", "CAUTION"}:
        border_col = "#F9A825"
        bg_col = "rgba(249, 168, 37, 0.15)"
        svg = svg_icon(ALERT_SVG, size=20, color="#FFCD11")
        label = "OPERATIONAL CAUTION"
        sub = headline or "Elevated condition detected. Check proximity zone and maintain situational awareness."
    elif level in {"UNKNOWN", "NO_DATA"}:
        border_col = "#546E7A"
        bg_col = "rgba(84, 110, 122, 0.12)"
        svg = svg_icon(INFO_SVG, size=20, color="#78909C")
        label = "AWAITING TELEMETRY · STANDBY"
        sub = headline or "No sensor data has been loaded yet. Select a data source in the sidebar to begin monitoring."
    else:
        border_col = "#C62828"
        bg_col = "rgba(198, 40, 40, 0.18)"
        svg = svg_icon(DANGER_SVG, size=20, color="#EF5350")
        label = "CRITICAL SAFETY HOLD"
        sub = headline or "Safety interlock engaged or critical fault. Park machine safely and report to supervisor."

    return (
        f'<div style="border-left: 4px solid {border_col}; background: {bg_col}; '
        f'padding: 12px 16px; border-radius: 6px; margin: 8px 0 16px 0;">'
        f'<div style="font-weight: 700; font-size: 14px; letter-spacing: 0.5px; color: {border_col}; margin-bottom: 4px;">'
        f'{svg}{label}</div>'
        f'<div style="font-size: 13px; opacity: 0.9;">{sub}</div>'
        f'</div>'
    )
