from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.parent
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class BlocklistSettings(BaseSettings):
    PATH: str = Field("./blocklist.yml", alias="BLOCKLIST_PATH")


class ToxicitySettings(BaseSettings):
    MODEL_PATH: str = Field("./models/rubert-tiny-toxicity", alias="TOXICITY_MODEL_PATH")
    THRESHOLD: float = Field(0.5, alias="TOXICITY_THRESHOLD")


class NerSettings(BaseSettings):
    MODEL_PATH: str = Field("./models/bert-multilingual-ner", alias="NER_MODEL_PATH")
    MASK: dict[str, str] = {"PER": "[PER]", "LOC": "[LOC]", "ORG": "[ORG]"}


class PromptInjectionSettings(BaseSettings):
    MODEL_PATH: str = Field("./models/mdeberta-prompt-injection", alias="PROMPT_INJECTION_MODEL_PATH")
    THRESHOLD: float = Field(0.5, alias="PROMPT_INJECTION_THRESHOLD")


class AppSettings(BaseSettings):
    BlocklistConfig: BlocklistSettings = BlocklistSettings()
    ToxicityConfig: ToxicitySettings = ToxicitySettings()
    NerConfig: NerSettings = NerSettings()
    PromptInjectionConfig: PromptInjectionSettings = PromptInjectionSettings()


Config = AppSettings()


def load_blocklist() -> set[str]:
    with open(Config.BlocklistConfig.PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {w.strip().lower() for w in data.get("words", []) if w.strip()}


BLOCKLIST: set[str] = load_blocklist()
