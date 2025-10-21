from dataclasses import dataclass

import pytest

from services.normalizers import normalize_for_magento


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
