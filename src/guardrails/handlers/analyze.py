from typing import Optional

from fastapi import APIRouter, Header

from src.config import BLOCKLIST
from src.guardrails.schemas import GuardrailRequest, GuardrailResponse
from src.utils.logging import getLogger, trace_id

logger = getLogger(__name__)
router = APIRouter()


@router.post("/analyze/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse, response_model_exclude_none=True)
async def guardrail(
    body: GuardrailRequest,
    authorization: Optional[str] = Header(default=None),
) -> GuardrailResponse:
    trace_id.set(body.litellm_trace_id or "-")
    texts = body.texts or []

    for text in texts:
        for word in text.lower().split():
            if word in BLOCKLIST:
                logger.warning(
                    "blocked by blocklist",
                    extra={"word": word, "input_type": body.input_type},
                )
                return GuardrailResponse(
                    action="BLOCKED",
                    blocked_reason=f"policy violation: '{word}' ({body.input_type})",
                )

    return GuardrailResponse(action="NONE")
