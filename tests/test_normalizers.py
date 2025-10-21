from dataclasses import dataclass

import pytest

from services.normalizers import _coerce_ai_value, normalize_for_magento, normalize_units


@dataclass
class DummyCache:
    store: dict[str, dict]

    def get(self, code: str):
        return self.store.get(code)


@pytest.fixture
def meta_cache():
    return DummyCache(
        {
            "country_of_manufacture": {
                "attribute_code": "country_of_manufacture",
                "frontend_input": "select",
                "options_map": {"China": "CN"},
                "values_to_labels": {"CN": "China"},
            },
            "description": {
                "attribute_code": "description",
                "frontend_input": "textarea",
            },
            "manufacturer": {
                "attribute_code": "manufacturer",
                "frontend_input": "select",
                "options_map": {"Gibson": 42},
                "values_to_labels": {"42": "Gibson"},
            },
            "product_condition": {
                "attribute_code": "product_condition",
                "frontend_input": "select",
                "options_map": {"New": "new"},
                "values_to_labels": {"new": "New"},
            },
            "no_strings": {
                "attribute_code": "no_strings",
                "frontend_input": "select",
                "options_map": {"4": 4, "6": 6},
                "values_to_labels": {"4": "4 strings", "6": "6 strings"},
            },
        }
    )


def test_country_label_to_code(meta_cache):
    assert (
        normalize_for_magento("country_of_manufacture", "China", meta_cache) == "CN"
    )


def test_country_accepts_code_casefold(meta_cache):
    assert (
        normalize_for_magento("country_of_manufacture", "cn", meta_cache) == "CN"
    )


def test_brand_label_and_code(meta_cache):
    assert normalize_for_magento("manufacturer", "Gibson", meta_cache) == 42
    assert normalize_for_magento("manufacturer", "42", meta_cache) == 42


def test_condition_label_to_value(meta_cache):
    assert normalize_for_magento("product_condition", "New", meta_cache) == "new"


def test_no_strings_blank_returns_none(meta_cache):
    assert normalize_for_magento("no_strings", None, meta_cache) is None
    assert normalize_for_magento("no_strings", "", meta_cache) is None
    assert normalize_for_magento("no_strings", "4", meta_cache) == 4


def test_normalize_units_mm_to_inches():
    assert normalize_units("scale_mensur", "648 mm") == '25.51"'


def test_normalize_units_formats_inches():
    assert normalize_units("neck_radius", "9.5 in") == '9.5"'
    assert normalize_units("neck_nutwidth", 1.6875) == '1.69"'


def test_coerce_text_values_trim_and_preserve():
    meta = {"frontend_input": "text"}
    assert _coerce_ai_value("  Hello  ", meta) == "Hello"
    assert _coerce_ai_value("   ", meta) == ""
    assert _coerce_ai_value(42, meta) == "42"
    assert _coerce_ai_value(None, meta) == ""


def test_normalize_text_attribute_to_string(meta_cache):
    assert normalize_for_magento("description", "  Test  ", meta_cache) == "Test"
    assert normalize_for_magento("description", 123, meta_cache) == "123"
    assert normalize_for_magento("description", "", meta_cache) == ""
    assert normalize_for_magento("description", "   ", meta_cache) == ""
