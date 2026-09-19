from typing import Optional

from fastapi import APIRouter, Header

from src.config import Config
from src.guardrails.schemas import GuardrailRequest, GuardrailResponse
from src.utils import models as model_store
from src.utils.logging import getLogger, trace_id

logger = getLogger(__name__)
router = APIRouter()


def _apply_masks(text: str, entities: list[dict]) -> tuple[str, list[str]]:
    spans: list[tuple[int, int, str]] = []

    for ent in entities:
        mask = Config.NerConfig.MASK.get(ent["entity_group"])
        if mask:
            spans.append((ent["start"], ent["end"], mask))

    if not spans:
        return text, []

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

    return text, [mask for _, _, mask in safe]


@router.post("/pii/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse, response_model_exclude_none=True)
async def pii_guardrail(
    body: GuardrailRequest,
    authorization: Optional[str] = Header(default=None),
) -> GuardrailResponse:
    trace_id.set(body.litellm_trace_id or "-")
    texts = body.texts or []
    if not texts:
        return GuardrailResponse(action="NONE")

    # Single batched inference call for all texts.
    batch_entities: list[list[dict]] = model_store._ner_pipeline(texts)

    results = [_apply_masks(text, entities) for text, entities in zip(texts, batch_entities)]
    masked = [text for text, _ in results]
    found = [label for _, labels in results for label in labels]

    if masked != texts:
        logger.info(
            "PII masked",
            extra={"entities": found, "input_type": body.input_type},
        )
        return GuardrailResponse(action="GUARDRAIL_INTERVENED", texts=masked)

    return GuardrailResponse(action="NONE")
