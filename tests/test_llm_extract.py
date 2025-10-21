import json

import pytest

from services.llm_extract import (
    MODEL_STRONG,
    brand_from_title,
    extract_attributes,
    regex_preextract,
    shortlist_allowed_values,
)


class DummyChatCompletions:
    def __init__(self, outer):
        self._outer = outer

    def create(self, *, model, messages, temperature):  # noqa: D401 - test stub
        self._outer.called_models.append(model)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self._outer.response_payload),
                    }
                }
            ]
        }


class DummyClient:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.called_models: list[str] = []
        self.chat = type("Chat", (), {"completions": DummyChatCompletions(self)})()


def test_regex_preextract_handles_vintera_p_bass():
    title = "Fender LTD Vintera II Roadworn 60s P-Bass FRD"
    description = (
        "Road Worn finish with 34\" scale length, 9.5\" radius maple neck, "
        "Vintage-Style 4-Saddle bridge and 20 frets."
    )

    extracted = regex_preextract(title, description)

    assert extracted["finish"] == "Fiesta Red"
    assert extracted["scale_mensur"] == '34"'
    assert extracted["neck_radius"] == '9.5"'
    assert extracted["amount_of_frets"] == "20"
    assert extracted["bridge"] == "Vintage-Style 4-Saddle"
    assert extracted["pickup_config"] == "P"


def test_brand_from_title_detects_brand():
    options = [{"label": "Fender"}, {"label": "Gibson"}]
    guess = brand_from_title("Fender LTD Vintera II Roadworn 60s P-Bass FRD", options)
    assert guess == "Fender"


def test_shortlist_allowed_values_prioritizes_seeds_and_title():
    meta = {
        "options": [
            {"label": "Fiesta Red"},
            {"label": "Black"},
            {"label": "Lake Placid Blue"},
        ]
    }
    shortlist = shortlist_allowed_values(
        "finish",
        meta,
        "Fender Strat Fiesta Red",
        ["Lake Placid Blue", "Shell Pink"],
    )
    assert shortlist[:4] == ["Lake Placid Blue", "Shell Pink", "Fiesta Red", "Black"]


def test_extract_attributes_prefers_preextract_for_numeric():
    title = "Fender LTD Vintera II Roadworn 60s P-Bass FRD"
    description = "Vintage-Style 4-Saddle bridge, 34\" scale length."

    client = DummyClient({
        "scale_mensur": '33"',
        "finish": "Candy Red",
    })

    tasks = [
        {"code": "scale_mensur", "meta": {}},
        {"code": "finish", "meta": {}},
    ]

    result, metadata = extract_attributes(
        client,
        title=title,
        description=description,
        attribute_tasks=tasks,
    )

    assert result["scale_mensur"] == '34"'
    assert result["finish"] == "Candy Red"
    assert metadata["used_model"] == MODEL_STRONG
    assert client.called_models == [MODEL_STRONG]
    assert metadata["preextract"]["finish"] == "Fiesta Red"


if __name__ == "__main__":
    pytest.main([__file__])
