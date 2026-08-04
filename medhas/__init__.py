"""Medhas — production-grade unified AI agent memory engine.

Restructured 2026-08-04 into a deep, hierarchical package:
  - medhas.llm        -> provider-agnostic LLM integration (factory + providers + router)
  - medhas.embeddings -> provider-agnostic embedding integration
  - medhas.memory     -> 6-layer memory engine (session/working/atomic/graph/ingestion/associative)
  - medhas.cognition  -> cognitive loop (perception/reasoning/generalization/embodiment)
  - medhas.engine     -> top-level orchestration (MemoryEngine)

Design follows the integration patterns of the six reference engines
(mem0, cognee, LightRAG, HippoRAG, graphiti, Letta): an abstract provider base,
concrete provider classes, a config-driven factory/registry, task-based routing,
and a structured-output layer. No provider (including Groq) is hard-coded.
"""
__version__ = "0.1.0"
