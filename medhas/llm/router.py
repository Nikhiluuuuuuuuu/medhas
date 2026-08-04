"""Task-based LLM router (graphiti ``LLMClient`` / cognee ``LLMGateway`` style).

Different memory subsystems need different models (cheap/fast for extraction, strong for
reasoning). The router holds one primary LLM plus optional per-task overrides, all built
from medhas.config. The engine never imports a specific provider.
"""
from typing import Dict, Optional

from .base import BaseLLM
from .config import LLMConfig
from .errors import LLMConfigError
from .factory import create_llm

# Logical task names used across the engine.
EXTRACTION = "extraction"      # graph/entity/date extraction
RESOLUTION = "resolution"      # anaphora / coreference
REASONING = "reasoning"        # cognitive reasoning + dreaming
SYNTHESIS = "synthesis"        # summarisation / reflection

DEFAULT_TASKS = (EXTRACTION, RESOLUTION, REASONING, SYNTHESIS)


class LLMRouter:
    def __init__(self, primary_config: LLMConfig,
                 task_configs: Optional[Dict[str, LLMConfig]] = None) -> None:
        self._primary = create_llm(primary_config)
        self._tasks: Dict[str, BaseLLM] = {}
        for name, cfg in (task_configs or {}).items():
            self._tasks[name] = create_llm(cfg)
        self._default_config = primary_config

    @classmethod
    def from_settings(cls, settings) -> "LLMRouter":
        """Build from an app ``settings`` object exposing the LLM_*/GROQ_* fields."""
        primary = LLMConfig(
            provider=getattr(settings, "LLM_PROVIDER", "groq"),
            model=getattr(settings, "LLM_MODEL", getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")),
            api_key=getattr(settings, "LLM_API_KEY", getattr(settings, "GROQ_API_KEY", None)),
            base_url=getattr(settings, "LLM_BASE_URL", getattr(settings, "GROQ_BASE_URL", None)) or None,
            temperature=getattr(settings, "LLM_TEMPERATURE", getattr(settings, "GROQ_TEMPERATURE", 0.1)),
            max_tokens=getattr(settings, "LLM_MAX_TOKENS", getattr(settings, "GROQ_MAX_TOKENS", 2048)),
            timeout=getattr(settings, "LLM_TIMEOUT", 60.0),
        )
        # Optional fast model for extraction/resolution.
        fast_cfg = None
        fast_model = getattr(settings, "LLM_FAST_MODEL", getattr(settings, "GROQ_FAST_MODEL", None))
        if fast_model:
            fast_cfg = primary.merged(model=fast_model)
        tasks = {}
        if fast_cfg:
            tasks[EXTRACTION] = fast_cfg
            tasks[RESOLUTION] = fast_cfg
        router = cls(primary, tasks)
        return router

    def for_task(self, task: str) -> BaseLLM:
        return self._tasks.get(task, self._primary)

    @property
    def primary(self) -> BaseLLM:
        return self._primary

    async def complete(self, task: str, messages, **kwargs):
        return await self.for_task(task).acompletion(messages, **kwargs)
