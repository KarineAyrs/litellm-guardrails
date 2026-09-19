from contextvars import ContextVar

trace_id = ContextVar("trace_id", default="-")
