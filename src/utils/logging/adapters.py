import logging

from .tracing import trace_id


class ServiceLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        kwargs.setdefault("extra", {})
        kwargs["extra"]["service"] = self.extra["service"]
        kwargs["extra"]["trace_id"] = trace_id.get()
        return msg, kwargs
