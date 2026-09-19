# app.py
import os
from typing import Any, Literal, Optional

from fastapi import FastAPI, Header
from pydantic import BaseModel


BLOCKLIST = [w.strip().lower() for w in os.getenv("BLOCKLIST", "secret,password").split(",") if w.strip()]

app = FastAPI(title="litellm-generic-guardrail")


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


@app.post("/beta/litellm_basic_guardrail_api", response_model=GuardrailResponse, response_model_exclude_none=True)
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}