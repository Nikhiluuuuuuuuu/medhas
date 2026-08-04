"""Production Structured & Color-Coded Logging Engine using Rich."""

import sys
import time
import logging
from typing import Any
from contextlib import asynccontextmanager
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler

# UTF-8 Encoding enforcement for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Custom Theme for Color-Coded Log Identification
custom_theme = Theme({
    "session": "bold cyan",
    "working": "bold magenta",
    "atomic": "bold yellow",
    "graph": "bold blue",
    "latency": "bold green",
    "tool": "bold purple",
    "error": "bold red",
    "warning": "bold orange3",
})

console = Console(theme=custom_theme, force_terminal=True)

# Suppress noisy third-party loggers
for quiet_logger in ["httpx", "groq", "huggingface_hub", "fastembed", "asyncpg", "urllib3"]:
    logging.getLogger(quiet_logger).setLevel(logging.ERROR)


def setup_logger(name: str = "unified_memory") -> logging.Logger:
    """Configure structured logger with clean Rich formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

# Helper Functions for Explicit Color-Coded Log Categorization

def log_session(msg: str) -> None:
    """Log session layer message (Cyan)."""
    console.print(f"[session][LAYER 1: SESSION][/session] {msg}")

def log_working(msg: str) -> None:
    """Log working memory block RAM message (Magenta)."""
    console.print(f"[working][LAYER 2: WORKING RAM][/working] {msg}")

def log_atomic(msg: str) -> None:
    """Log atomic vector fact message (Yellow)."""
    console.print(f"[atomic][LAYER 3: ATOMIC FACT][/atomic] {msg}")

def log_graph(msg: str) -> None:
    """Log bi-temporal knowledge graph message (Blue)."""
    console.print(f"[graph][LAYER 4: TEMPORAL GRAPH][/graph] {msg}")

def log_tool(tool_name: str, result: str) -> None:
    """Log tool call execution (Purple)."""
    console.print(f"[tool][TOOL CALL: {tool_name}][/tool] {result}")

def log_error(msg: str) -> None:
    """Log error message neatly without messy raw tracebacks (Red)."""
    console.print(f"[error][SYSTEM ERROR][/error] {msg}")


@asynccontextmanager
async def measure_latency(operation_name: str):
    """Async context manager measuring sub-millisecond execution latency."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        console.print(f"[latency][LATENCY][/latency] [bold white]{operation_name}[/bold white] -> [green]{elapsed_ms:.2f} ms[/green]")
