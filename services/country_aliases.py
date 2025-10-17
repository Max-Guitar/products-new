"""Country alias helpers for Magento attribute normalization."""
from __future__ import annotations

from typing import Dict, Set

try:  # pragma: no cover - optional dependency
    import pycountry  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pycountry = None  # type: ignore


def _load_countries() -> Dict[str, Set[str]]:
    alias_map: Dict[str, Set[str]] = {}
    if pycountry:
        for country in getattr(pycountry, "countries", []):
            aliases: Set[str] = set()
            canonical = str(getattr(country, "name", "")).strip()
            if not canonical:
                continue
            canonical_lower = canonical.lower()
            aliases.add(canonical_lower)
            for attr in ("official_name", "common_name"):
                value = getattr(country, attr, None)
                if isinstance(value, str) and value.strip():
                    aliases.add(value.strip().lower())
            for attr in ("alpha_2", "alpha_3"):
                value = getattr(country, attr, None)
                if isinstance(value, str) and value.strip():
                    aliases.add(value.strip().lower())
            alias_map[canonical_lower] = aliases
    else:
        minimal = {
            "china": {"china", "people's republic of china", "cn", "chn"},
            "united states": {
                "united states",
                "united states of america",
                "usa",
                "us",
            },
            "united kingdom": {"united kingdom", "great britain", "uk", "gbr"},
            "germany": {"germany", "de", "deu", "federal republic of germany"},
            "france": {"france", "fr", "fra"},
            "canada": {"canada", "ca", "can"},
            "japan": {"japan", "jp", "jpn"},
            "australia": {"australia", "au", "aus"},
            "mexico": {"mexico", "mx", "mex"},
        }
        for canonical, aliases in minimal.items():
            alias_map[canonical] = {alias.lower() for alias in aliases}
    return alias_map


_COUNTRY_ALIASES = _load_countries()
_ALIAS_INDEX: Dict[str, str] = {}
for canonical, aliases in _COUNTRY_ALIASES.items():
    for alias in aliases:
        _ALIAS_INDEX[alias] = canonical


def country_aliases(name_or_code: object) -> list[str]:
    """Return a list of lowercase aliases for the given country name or code."""
    if name_or_code is None:
        return []
    if isinstance(name_or_code, str):
        value = name_or_code.strip()
        if not value:
            return []
    else:
        value = str(name_or_code).strip()
        if not value:
            return []
    lower_value = value.lower()
    canonical = _ALIAS_INDEX.get(lower_value)
    if not canonical:
        # if the input looks like a full name, try to use it as canonical
        canonical = lower_value
        if canonical not in _COUNTRY_ALIASES:
            return [lower_value]
    aliases = _COUNTRY_ALIASES.get(canonical, {lower_value})
    return sorted(aliases)
