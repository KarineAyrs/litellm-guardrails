from dataclasses import dataclass
from typing import Optional

import torch
from fastapi import APIRouter, Header

from src.config import Config
from src.guardrails.schemas import GuardrailRequest, GuardrailResponse
from src.utils import models as model_store
from src.utils.logging import getLogger, trace_id

logger = getLogger(__name__)
router = APIRouter()


@dataclass
class ToxicityResult:
    text: str
    scores: dict[str, float]
    toxicity_score: float
    is_toxic: bool


def _classify(texts: list[str]) -> list[ToxicityResult]:
    with torch.no_grad():
        enc = model_store._toxicity_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        proba = torch.sigmoid(model_store._toxicity_model(**enc).logits).cpu().numpy()  # (N, 5)

    labels: list[str] = [
        model_store._toxicity_model.config.id2label[i] for i in range(proba.shape[1])
    ]

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
            is_toxic=toxicity_score > Config.ToxicityConfig.THRESHOLD,
        ))
    return results


@router.post("/toxicity/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse, response_model_exclude_none=True)
async def toxicity_guardrail(
    body: GuardrailRequest,
    authorization: Optional[str] = Header(default=None),
) -> GuardrailResponse:
    trace_id.set(body.litellm_trace_id or "-")
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
        logger.warning(
            "blocked: toxicity detected",
            extra={
                "label": top_label,
                "label_score": f"{worst.scores[top_label]:.2f}",
                "toxicity_score": worst.toxicity_score,
                "input_type": body.input_type,
            },
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
