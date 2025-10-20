import pandas as pd
import pytest

from services.ai_fill import derive_styles_from_texts, enrich_ai_suggestions


CATEGORIES_META = {
    "labels_to_values": {
        "Guitars": "1",
        "Electric Guitar": "2",
        "Fender": "3",
    }
}


@pytest.mark.parametrize(
    "texts, expected",
    [
        (
            [
                "Fender Telecaster Deluxe",
                "Vintage sg Special",
                "Les Paul Custom",
                "Classic Stratocaster",
                "Random text",  # ignored
            ],
            ["T-Style", "Double cut", "Single cut", "S-Style"],
        ),
        (
            ["telecaster", "telecaster", "TELECASTER"],
            ["T-Style"],
        ),
    ],
)
def test_derive_styles_from_texts(texts, expected):
    assert derive_styles_from_texts(texts) == expected


def _get_entry(df: pd.DataFrame, code: str) -> pd.Series:
    matching = df[df["code"] == code]
    assert not matching.empty, f"Entry for {code} not found"
    return matching.iloc[0]


def test_enrich_ai_suggestions_merges_category_and_style_hints():
    base_df = pd.DataFrame(
        [
            {"code": "guitarstylemultiplechoice", "value": ["T-Style"], "reason": "model"},
            {"code": "categories", "value": ["Guitars"], "reason": "ai"},
            {"code": "model", "value": "Stratocaster"},
        ]
    )

    hints = {
        "brand_hint": ["Electric Guitar"],
        "set_hint": "Electric guitar",
        "style_hint": ["S-Style"],
        "model_series_candidates": ["Les Paul Special"],
    }

    enriched = enrich_ai_suggestions(
        base_df,
        hints,
        CATEGORIES_META,
        "Electric guitar",
        12,
    )

    categories_entry = _get_entry(enriched, "categories")
    assert categories_entry["value"] == ["Guitars", "Electric Guitar"]
    assert categories_entry["reason"] == "ai"

    styles_entry = _get_entry(enriched, "guitarstylemultiplechoice")
    assert styles_entry["value"] == [
        "T-Style",
        "S-Style",
        "Single cut",
        "Double cut",
    ]
    assert styles_entry["reason"] == "model"

    meta = enriched.attrs.get("meta", {})
    assert meta.get("brand_hint") == "Electric Guitar"
    assert meta.get("normalized_categories_added") == ["Electric Guitar"]


def test_enrich_ai_suggestions_creates_entries_from_hints():
    base_df = pd.DataFrame(
        [
            {"code": "series", "value": "American Vintage"},
        ]
    )

    hints = {
        "brand_hint": "Fender",
        "style_hint": ["S-Style"],
        "model_series_candidates": ["Telecaster"],
    }

    enriched = enrich_ai_suggestions(
        base_df,
        hints,
        CATEGORIES_META,
        "Electric guitar",
        12,
    )

    categories_entry = _get_entry(enriched, "categories")
    assert categories_entry["value"] == ["Fender", "Electric Guitar"]
    assert categories_entry["reason"] == "enriched_from_hints"

    styles_entry = _get_entry(enriched, "guitarstylemultiplechoice")
    assert styles_entry["value"] == ["S-Style", "T-Style", "Double cut", "Single cut"]
    assert styles_entry["reason"] == "enriched_from_hints"

    # Original entries should still be present
    series_entry = _get_entry(enriched, "series")
    assert series_entry["value"] == "American Vintage"

    meta = enriched.attrs.get("meta", {})
    assert meta.get("brand_hint") == "Fender"
    assert meta.get("normalized_categories_added") == [
        "Fender",
        "Electric Guitar",
    ]
