"""Production logging configuration loader.

Complements the Rich-based ``medhas.utils.logger`` by allowing ops to switch the engine's
root logging to structured JSON (for log aggregation) or keep human-readable text, and to
route to a rotating file. Driven entirely by environment variables — no code changes needed
to reconfigure in different environments.

Env:
  MEDHAS_LOG_LEVEL   -> DEBUG|INFO|WARNING|ERROR  (default INFO)
  MEDHAS_LOG_FORMAT -> json|text              (default text)
  MEDHAS_LOG_FILE   -> path|"" (empty disables file) (default logs/medhas.log when json)
"""
from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

from medhas.utils import logger as _rich_logger

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"
# Dev (editable install) keeps config/ at the repo root, not inside the package.
_REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "logging.yaml"

def _resolve_config() -> Path | None:
    for cand in (_CONFIG_PATH, _REPO_CONFIG):
        if cand.exists():
            return cand
    return None


def configure_logging(
    level: str | None = None,
    fmt: str | None = None,
    log_file: str | None = None,
) -> None:
    """Apply logging config from ``config/logging.yaml`` with env overrides."""
    import yaml

    level = (level or os.getenv("MEDHAS_LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.getenv("MEDHAS_LOG_FORMAT", "text")).lower()
    log_file = log_file if log_file is not None else os.getenv("MEDHAS_LOG_FILE", "")

    cfg_path = _resolve_config()
    if cfg_path is None:
        # Fall back to the Rich logger if the YAML is somehow missing.
        _rich_logger.warning("logging.yaml not found (%s / %s); using Rich logger only", _CONFIG_PATH, _REPO_CONFIG)
        return

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Env-driven overrides.
    cfg.setdefault("loggers", {}).setdefault("medhas", {})["level"] = level
    for name in ("medhas.llm", "medhas.storage"):
        cfg["loggers"].setdefault(name, {})["level"] = level

    handlers: dict = cfg.setdefault("handlers", {})
    if fmt == "json":
        handlers["console"]["formatter"] = "json"
    # Ensure the rotating file handler's directory exists before dictConfig validates it.
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    elif "json_file" in handlers:
        Path(handlers["json_file"]["filename"]).parent.mkdir(parents=True, exist_ok=True)
    if log_file:
        handlers["json_file"]["filename"] = log_file
        cfg["root"]["handlers"] = ["console", "json_file"]
        for lg in cfg["loggers"].values():
            lg["handlers"] = ["console", "json_file"]

    logging.config.dictConfig(cfg)
    logging.getLogger("medhas").info(
        "logging configured: level=%s format=%s file=%s", level, fmt, log_file or "disabled"
    )


if __name__ == "__main__":
    configure_logging()
