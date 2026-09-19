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
class InjectionResult:
    text: str
    score: float
    is_injection: bool


def _detect(texts: list[str]) -> list[InjectionResult]:
    with torch.no_grad():
        enc = model_store._prompt_injection_tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True, max_length=256
        )
        probs = torch.softmax(model_store._prompt_injection_model(**enc).logits, dim=-1)
        scores = probs[:, 1].cpu().tolist()  # class 1 = prompt_injection

    threshold = Config.PromptInjectionConfig.THRESHOLD
    return [
        InjectionResult(text=text, score=round(score, 4), is_injection=score > threshold)
        for text, score in zip(texts, scores)
    ]


@router.post("/prompt-injection/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse, response_model_exclude_none=True)
async def prompt_injection_guardrail(
    body: GuardrailRequest,
    authorization: Optional[str] = Header(default=None),
) -> GuardrailResponse:
    trace_id.set(body.litellm_trace_id or "-")
    texts = body.texts or []
    if not texts:
        return GuardrailResponse(action="NONE")

    results = _detect(texts)

    injections = [r for r in results if r.is_injection]
    if injections:
        worst = max(injections, key=lambda r: r.score)
        logger.warning(
            "blocked: prompt injection detected",
            extra={"score": worst.score, "input_type": body.input_type},
        )
        return GuardrailResponse(
            action="BLOCKED",
            blocked_reason=f"prompt injection detected (score {worst.score:.2f})",
        )

    return GuardrailResponse(action="NONE")
