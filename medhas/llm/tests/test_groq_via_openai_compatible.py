"""Online smoke test for the new provider-agnostic LLM subsystem.

Proves Groq works through the new OpenAI-compatible provider with ZERO Groq-specific code
(no groq SDK, no hard-coded GROQ_MODEL). Run with a real key in env (GROQ_API_KEY).
"""
import os
import asyncio
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from medhas.llm import LLMConfig, create_llm


async def main():
    # Groq reached purely as an OpenAI-compatible endpoint.
    cfg = LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.0,
    )
    llm = create_llm(cfg)
    print("Built provider:", llm)
    out = await llm.acompletion([
        {"role": "system", "content": "Reply with a single word: PONG"},
        {"role": "user", "content": "ping"},
    ])
    print("RESPONSE:", repr(out["content"]))
    assert out["content"].strip().lower() in ("pong", "pong."), out["content"]
    print("OK — provider-agnostic Groq call succeeded with no groq-specific code.")


if __name__ == "__main__":
    asyncio.run(main())
