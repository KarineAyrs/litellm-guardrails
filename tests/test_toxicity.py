import torch
from unittest.mock import MagicMock, patch

ENDPOINT = "/toxicity/beta/litellm_basic_guardrail_api"

ID2LABEL = {0: "non-toxic", 1: "insult", 2: "obscenity", 3: "threat", 4: "dangerous"}


def _make_mocks(logits: torch.Tensor):
    mock_tokenizer = MagicMock(return_value={"input_ids": torch.zeros(1, 3, dtype=torch.long)})
    mock_output = MagicMock()
    mock_output.logits = logits
    mock_model = MagicMock(return_value=mock_output)
    mock_model.config.id2label = ID2LABEL
    return mock_tokenizer, mock_model


def test_toxic_text_blocked(client):
    # non-toxic≈0.10, insult≈0.90 → toxicity_score≈0.91 > 0.5
    tok, mdl = _make_mocks(torch.tensor([[-2.2, 2.2, -2.2, -2.2, -2.2]]))
    with patch("src.utils.models._toxicity_tokenizer", tok), \
         patch("src.utils.models._toxicity_model", mdl):
        resp = client.post(ENDPOINT, json={"texts": ["bad text"], "input_type": "request"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "BLOCKED"
    assert "insult" in data["blocked_reason"]


def test_clean_text_passes(client):
    # non-toxic≈0.95, rest≈0.05 → toxicity_score≈0.10 < 0.5
    tok, mdl = _make_mocks(torch.tensor([[2.944, -2.944, -2.944, -2.944, -2.944]]))
    with patch("src.utils.models._toxicity_tokenizer", tok), \
         patch("src.utils.models._toxicity_model", mdl):
        resp = client.post(ENDPOINT, json={"texts": ["hello world"], "input_type": "request"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "NONE"


def test_worst_offender_reported(client):
    # Two texts: first clean, second toxic
    logits = torch.tensor([
        [2.944, -2.944, -2.944, -2.944, -2.944],   # non-toxic≈0.95
        [-2.2,  -2.2,   -2.2,  2.2,   -2.2],        # threat≈0.90
    ])
    tok, mdl = _make_mocks(logits)
    with patch("src.utils.models._toxicity_tokenizer", tok), \
         patch("src.utils.models._toxicity_model", mdl):
        resp = client.post(ENDPOINT, json={"texts": ["hi", "bad"], "input_type": "request"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "BLOCKED"
    assert "threat" in data["blocked_reason"]


def test_empty_texts(client):
    resp = client.post(ENDPOINT, json={"texts": [], "input_type": "request"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "NONE"
