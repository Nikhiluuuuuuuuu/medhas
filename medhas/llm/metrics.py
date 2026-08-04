"""Tiny latency instrumentation (no external dependency).

Mirrors the `measure_latency` helper the rest of the codebase uses, kept local so the
`medhas.llm` package is self-contained.
"""
import asyncio
import functools
import logging
import time

_log = logging.getLogger("medhas.llm")


def measure_latency(name: str):
    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _log.info("[LATENCY] %s -> %.2f ms", name, (time.perf_counter() - start) * 1000)
            return awrapper

        @functools.wraps(fn)
        def swrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _log.info("[LATENCY] %s -> %.2f ms", name, (time.perf_counter() - start) * 1000)
        return swrapper
    return decorator
