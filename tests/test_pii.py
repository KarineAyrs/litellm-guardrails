from unittest.mock import MagicMock, patch

ENDPOINT = "/pii/beta/litellm_basic_guardrail_api"


def test_person_masked(client):
    # "Меня зовут " = 11 chars → PER spans [11, 22]
    mock_pipeline = MagicMock(return_value=[[
        {"entity_group": "PER", "start": 11, "end": 22, "word": "Иван Петров", "score": 0.99},
    ]])
    with patch("src.utils.models._ner_pipeline", mock_pipeline):
        resp = client.post(ENDPOINT, json={"texts": ["Меня зовут Иван Петров"], "input_type": "request"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "GUARDRAIL_INTERVENED"
    assert data["texts"] == ["Меня зовут [PER]"]


def test_multiple_entity_types(client):
    mock_pipeline = MagicMock(return_value=[[
        {"entity_group": "PER", "start": 0,  "end": 4,  "word": "Иван", "score": 0.99},
        {"entity_group": "LOC", "start": 13, "end": 19, "word": "Москва", "score": 0.98},
        {"entity_group": "ORG", "start": 31, "end": 39, "word": "Газпром", "score": 0.97},
    ]])
    text = "Иван живёт в Москве и работает в Газпроме"
    with patch("src.utils.models._ner_pipeline", mock_pipeline):
        resp = client.post(ENDPOINT, json={"texts": [text], "input_type": "request"})
    assert resp.status_code == 200
    result = resp.json()["texts"][0]
    assert "[PER]" in result
    assert "[LOC]" in result
    assert "[ORG]" in result


def test_no_entities_returns_none(client):
    mock_pipeline = MagicMock(return_value=[[]])
    with patch("src.utils.models._ner_pipeline", mock_pipeline):
        resp = client.post(ENDPOINT, json={"texts": ["обычный текст"], "input_type": "request"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "NONE"


def test_empty_texts(client):
    resp = client.post(ENDPOINT, json={"texts": [], "input_type": "request"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "NONE"


def test_overlapping_spans_deduplicated(client):
    # Two spans that overlap — only the rightmost should be applied
    mock_pipeline = MagicMock(return_value=[[
        {"entity_group": "PER", "start": 0, "end": 10, "word": "Иван Иван", "score": 0.9},
        {"entity_group": "PER", "start": 5, "end": 10, "word": "Иван",      "score": 0.8},
    ]])
    with patch("src.utils.models._ner_pipeline", mock_pipeline):
        resp = client.post(ENDPOINT, json={"texts": ["Иван Иван!"], "input_type": "request"})
    assert resp.status_code == 200
    # Should not crash and should mask something
    assert resp.json()["action"] == "GUARDRAIL_INTERVENED"
