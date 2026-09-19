from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

from src.config import Config
from src.utils.logging import getLogger

logger = getLogger(__name__)

_toxicity_tokenizer: Optional[AutoTokenizer] = None
_toxicity_model: Optional[AutoModelForSequenceClassification] = None
_ner_pipeline = None
_prompt_injection_tokenizer: Optional[AutoTokenizer] = None
_prompt_injection_model: Optional[AutoModelForSequenceClassification] = None


def load_all_models() -> None:
    global _toxicity_tokenizer, _toxicity_model, _ner_pipeline, \
           _prompt_injection_tokenizer, _prompt_injection_model

    logger.info("loading toxicity model", extra={"path": Config.ToxicityConfig.MODEL_PATH})
    _toxicity_tokenizer = AutoTokenizer.from_pretrained(Config.ToxicityConfig.MODEL_PATH)
    _toxicity_model = AutoModelForSequenceClassification.from_pretrained(Config.ToxicityConfig.MODEL_PATH)
    _toxicity_model.eval()

    logger.info("loading NER model", extra={"path": Config.NerConfig.MODEL_PATH})
    _ner_pipeline = hf_pipeline(
        "ner",
        model=AutoModelForTokenClassification.from_pretrained(Config.NerConfig.MODEL_PATH),
        tokenizer=AutoTokenizer.from_pretrained(Config.NerConfig.MODEL_PATH),
        aggregation_strategy="simple",
    )

    logger.info("loading prompt injection model", extra={"path": Config.PromptInjectionConfig.MODEL_PATH})
    _prompt_injection_tokenizer = AutoTokenizer.from_pretrained(Config.PromptInjectionConfig.MODEL_PATH)
    _prompt_injection_model = AutoModelForSequenceClassification.from_pretrained(Config.PromptInjectionConfig.MODEL_PATH)
    _prompt_injection_model.eval()
