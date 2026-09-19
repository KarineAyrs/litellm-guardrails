from typing import Any, Literal, Optional

from pydantic import BaseModel


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
