import logging
import logging.config
from typing import Optional
from pythonjsonlogger.json import JsonFormatter

from .adapters import ServiceLoggerAdapter

SERVICE_NAME = "litellm-generic-guardrail"


def setup_logging(level: str = "INFO") -> None:
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter,
                "fmt": "%(asctime)s %(levelname)s %(name)s %(funcName)s %(message)s",
                "json_ensure_ascii": False,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
                "level": level,
            }
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
    }

    logging.config.dictConfig(config)


def getLogger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger = ServiceLoggerAdapter(logger, {"service": SERVICE_NAME})
    return logger
