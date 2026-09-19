from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.guardrails.handlers import analyze, pii, prompt_injection, toxicity
from src.utils.logging import getLogger
from src.utils.logging.logging import setup_logging
from src.utils.models import load_all_models

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("starting up, loading models")
    load_all_models()
    logger.info("models loaded, service ready")
    yield


app = FastAPI(title="litellm-generic-guardrail", lifespan=lifespan)

app.include_router(analyze.router)
app.include_router(toxicity.router)
app.include_router(pii.router)
app.include_router(prompt_injection.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
