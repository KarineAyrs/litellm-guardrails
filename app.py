# app.py
import os
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import torch
from fastapi import FastAPI, Header
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

# ── Config ───────────────────────────────────────────────────────────────────

BLOCKLIST = [w.strip().lower() for w in os.getenv("BLOCKLIST", "secret,password").split(",") if w.strip()]
TOXICITY_MODEL_PATH = os.getenv("TOXICITY_MODEL_PATH", "./models/rubert-tiny-toxicity")
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.5"))
NER_MODEL_PATH = os.getenv("NER_MODEL_PATH", "./models/bert-multilingual-ner")

_NER_MASK: dict[str, str] = {"PER": "[PER]", "LOC": "[LOC]", "ORG": "[ORG]"}

# ── Model handles ─────────────────────────────────────────────────────────────

_toxicity_tokenizer: Optional[AutoTokenizer] = None
_toxicity_model: Optional[AutoModelForSequenceClassification] = None
_ner_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _toxicity_tokenizer, _toxicity_model, _ner_pipeline

    _toxicity_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_PATH)
    _toxicity_model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_PATH)
    _toxicity_model.eval()

    _ner_pipeline = hf_pipeline(
        "ner",
        model=AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH),
        tokenizer=AutoTokenizer.from_pretrained(NER_MODEL_PATH),
        aggregation_strategy="simple",
        device=-1,
    )

    yield


app = FastAPI(title="litellm-generic-guardrail", lifespan=lifespan)


# ── Shared LiteLLM schema ─────────────────────────────────────────────────────

class GuardrailRequest(BaseModel):
    texts: Optional[list[str]] = None
    images: Optional[list[str]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    structured_messages: Optional[list[dict[str, Any]]] = None
    request_data: Optional[dict[str, Any]] = None
    request_headers: Optional[dict[str, str]] = None
    litellm_version: Optional[str] = None
    input_type: Optional[Literal["request", "response"]] = None
    litellm_call_id: Optional[str] = None
    litellm_trace_id: Optional[str] = None
    additional_provider_specific_params: Optional[dict[str, Any]] = None


class GuardrailResponse(BaseModel):
    action: Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"] = "NONE"
    blocked_reason: Optional[str] = None
    texts: Optional[list[str]] = None
    images: Optional[list[str]] = None
    structured_messages: Optional[list[dict[str, Any]]] = None


# ── Toxicity internals ────────────────────────────────────────────────────────

class ToxicityResult(BaseModel):
    text: str
    scores: dict[str, float]
    toxicity_score: float
    is_toxic: bool


def _classify(texts: list[str]) -> list[ToxicityResult]:
    with torch.no_grad():
        enc = _toxicity_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        proba = torch.sigmoid(_toxicity_model(**enc).logits).cpu().numpy()  # (N, 5)

    labels: list[str] = [_toxicity_model.config.id2label[i] for i in range(proba.shape[1])]

    results = []
    for text, row in zip(texts, proba):
        scores = {label: float(score) for label, score in zip(labels, row)}
        non_toxic = scores.get("non-toxic", scores.get("non_toxic", 0.0))
        dangerous = scores.get("dangerous", 0.0)
        toxicity_score = float(1 - non_toxic * (1 - dangerous))
        results.append(ToxicityResult(
            text=text,
            scores=scores,
            toxicity_score=round(toxicity_score, 4),
            is_toxic=toxicity_score > TOXICITY_THRESHOLD,
        ))
    return results


# ── PII internals ─────────────────────────────────────────────────────────────

def _anonymize(text: str) -> str:
    """Replace NER entities (PER/LOC/ORG) with typed mask tokens."""
    spans: list[tuple[int, int, str]] = []

    for ent in _ner_pipeline(text):
        mask = _NER_MASK.get(ent["entity_group"])
        if mask:
            spans.append((ent["start"], ent["end"], mask))

    if not spans:
        return text

    # Apply right-to-left so earlier offsets remain valid; drop overlapping spans.
    spans.sort(key=lambda s: s[0], reverse=True)
    safe: list[tuple[int, int, str]] = []
    right_bound = len(text)
    for start, end, mask in spans:
        if end <= right_bound:
            safe.append((start, end, mask))
            right_bound = start

    for start, end, mask in safe:
        text = text[:start] + mask + text[end:]

    return text


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/analyze/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse, response_model_exclude_none=True)
async def guardrail(
    body: GuardrailRequest,
    authorization: Optional[str] = Header(default=None),
) -> GuardrailResponse:
    texts = body.texts or []

    for text in texts:
        lowered = text.lower()
        for word in BLOCKLIST:
            if word in lowered:
                return GuardrailResponse(
                    action="BLOCKED",
                    blocked_reason=f"policy violation: '{word}' ({body.input_type})",
                )

    masked = [t.replace("@example.com", "@[REDACTED]") for t in texts]
    if masked != texts:
        return GuardrailResponse(action="GUARDRAIL_INTERVENED", texts=masked)

    return GuardrailResponse(action="NONE")


@app.post("/toxicity/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse, response_model_exclude_none=True)
async def toxicity_guardrail(
    body: GuardrailRequest,
    authorization: Optional[str] = Header(default=None),
) -> GuardrailResponse:
    texts = body.texts or []
    if not texts:
        return GuardrailResponse(action="NONE")

    results = _classify(texts)

    toxic = [r for r in results if r.is_toxic]
    if toxic:
        worst = max(toxic, key=lambda r: r.toxicity_score)
        top_label = max(
            (k for k in worst.scores if k != "non-toxic"),
            key=lambda k: worst.scores[k],
        )
        return GuardrailResponse(
            action="BLOCKED",
            blocked_reason=(
                f"toxicity detected: {top_label} "
                f"({worst.scores[top_label]:.0%}), "
                f"aggregate score {worst.toxicity_score:.2f}"
            ),
        )

    return GuardrailResponse(action="NONE")


@app.post("/pii/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse, response_model_exclude_none=True)
async def pii_guardrail(
    body: GuardrailRequest,
    authorization: Optional[str] = Header(default=None),
) -> GuardrailResponse:
    texts = body.texts or []
    if not texts:
        return GuardrailResponse(action="NONE")

    masked = [_anonymize(t) for t in texts]

    if masked != texts:
        return GuardrailResponse(action="GUARDRAIL_INTERVENED", texts=masked)

    return GuardrailResponse(action="NONE")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
