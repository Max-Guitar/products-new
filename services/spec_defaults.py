"""Series-based default attribute helpers for spec extraction."""

from __future__ import annotations

import re
from typing import Mapping

SERIES_RULES: list[dict[str, object]] = [
    {
        "when": [
            r"\bFender\b",
            r"\bVintera\s*II\b",
            r"\b(60s|’60s|1960s)\b",
            r"\bP[- ]?Bass\b",
        ],
        "defaults": {
            "scale_mensur": '34"',
            "no_strings": "4",
            "pickup_config": "P",
            "neck_profile": "C Shape",
            "neck_radius": '7.25"',
            "neck_nutwidth": '1.73"',
            "bridge": "Vintage-Style 4-Saddle",
            "tuning_machines": "Vintage-Style Open-Gear",
            "country_of_manufacture": "Mexico",
        },
    },
]


def apply_series_defaults(title: str | None, attrs: dict[str, object]) -> dict[str, object]:
    """Fill missing attributes using known defaults for specific series."""

    text = title or ""
    for rule in SERIES_RULES:
        when_patterns = rule.get("when")
        defaults = rule.get("defaults")
        if not isinstance(when_patterns, list) or not isinstance(defaults, Mapping):
            continue
        if all(re.search(pattern, text, flags=re.IGNORECASE) for pattern in when_patterns):
            for key, value in defaults.items():
                attrs.setdefault(key, value)
    return attrs


def maybe_set_gig_bag(title: str | None, attrs: dict[str, object]) -> dict[str, object]:
    """Set Gig Bag default for Road Worn / Roadworn instruments."""

    lowered = (title or "").lower()
    if "roadworn" in lowered or "road worn" in lowered:
        attrs.setdefault("cases_covers", "Gig Bag")
    return attrs


def enrich_pickups(attrs: dict[str, object]) -> dict[str, object]:
    """Normalize pickup structure for well-known configurations."""

    cfg = (attrs.get("pickup_config") or "").upper()
    if cfg == "P":
        attrs.setdefault("middle_pickup", "Split Single-Coil P")
        attrs.setdefault("neck_pickup", "")
    return attrs
