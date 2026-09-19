from unittest.mock import patch

ENDPOINT = "/analyze/beta/litellm_basic_guardrail_api"


def test_blocked_word_in_text(client):
    with patch("src.guardrails.handlers.analyze.BLOCKLIST", {"badword"}):
        resp = client.post(ENDPOINT, json={"texts": ["this is badword here"], "input_type": "request"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "BLOCKED"
    assert "badword" in data["blocked_reason"]


def test_clean_text_passes(client):
    with patch("src.guardrails.handlers.analyze.BLOCKLIST", {"badword"}):
        resp = client.post(ENDPOINT, json={"texts": ["hello world"], "input_type": "request"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "NONE"


def test_case_insensitive(client):
    with patch("src.guardrails.handlers.analyze.BLOCKLIST", {"badword"}):
        resp = client.post(ENDPOINT, json={"texts": ["BADWORD in text"], "input_type": "request"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "BLOCKED"


def test_empty_texts(client):
    resp = client.post(ENDPOINT, json={"texts": [], "input_type": "request"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "NONE"


def test_blocks_on_first_match(client):
    with patch("src.guardrails.handlers.analyze.BLOCKLIST", {"bad", "evil"}):
        resp = client.post(ENDPOINT, json={"texts": ["evil bad text"], "input_type": "request"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "BLOCKED"
