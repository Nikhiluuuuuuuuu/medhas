"""Production General-Purpose FastAPI REST Server for 6-in-1 Unified Memory Engine."""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from uuid import UUID
import os

from medhas.storage import DatabasePool, initialize_schema
from medhas.memory.session import create_session, get_transcript
from medhas.memory.working import get_blocks, update_block
from medhas.memory.atomic import search_facts, get_all_active_facts, run_dream_cycle, purge_user_memories
from medhas.memory.graph import query_point_in_time, query_subgraph, run_spreading_activation
from medhas.memory.graph.export_graph import export_knowledge_graph
from medhas.memory.procedural import store_skill_playbook
from medhas.pipeline import UnifiedMemoryEngine

# Roadmap AGI-memory endpoints (E1–E37), additive router — does not touch existing routes.
from medhas.platform.api import router as agi_router

app = FastAPI(
    title="AGI Unified Memory Engine API",
    description="General-purpose production memory engine supporting any arbitrary facts, domains, entities, and multi-turn conversations.",
    version="1.0.0"
)

engine: Optional[UnifiedMemoryEngine] = None

class TurnRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

class StorePlaybookRequest(BaseModel):
    user_id: str
    task_pattern: str
    steps: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize DB connection pool and schemas on startup."""
    global engine
    await DatabasePool.initialize()
    await initialize_schema()
    engine = UnifiedMemoryEngine()

@app.on_event("shutdown")
async def shutdown_event():
    """Close DB pool on shutdown."""
    await DatabasePool.close()

@app.post("/session/create")
async def api_create_session(user_id: str):
    """Create a new conversational session."""
    session = await create_session(user_id)
    return {"status": "success", "session_id": str(session.id), "user_id": user_id}

@app.get("/session/transcript/{session_id}")
async def api_get_transcript(session_id: str, limit: int = 50):
    """Get conversation transcript history for a session."""
    try:
        uuid_val = UUID(session_id)
        transcript = await get_transcript(uuid_val, limit=limit)
        return {"status": "success", "session_id": session_id, "messages": [m.__dict__ for m in transcript]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/turn/execute")
async def api_execute_turn(req: TurnRequest):
    """Execute a general-purpose conversation turn with arbitrary user text."""
    if not engine:
        raise HTTPException(status_code=500, detail="Memory engine not initialized")
    response_text = await engine.execute_turn(req.user_id, UUID(req.session_id), req.message)
    return {"status": "success", "response": response_text}

@app.get("/working/blocks")
async def api_get_working_blocks(user_id: str):
    """Query Layer 2 Letta working memory prompt RAM blocks."""
    blocks = await get_blocks(user_id)
    return {"user_id": user_id, "blocks": blocks}

@app.get("/memory/facts")
async def api_get_facts(user_id: str, query: Optional[str] = None):
    """Query long-term vector facts for any user ID."""
    if query:
        results = await search_facts(user_id, query)
        return {"user_id": user_id, "query": query, "facts": [r.__dict__ for r in results]}
    else:
        facts = await get_all_active_facts(user_id)
        return {"user_id": user_id, "facts": facts}

@app.get("/memory/graph")
async def api_get_graph(user_id: str):
    """Export Layer 4 knowledge graph for visualizer."""
    graph_data = await export_knowledge_graph(user_id)
    return {"status": "success", "graph": graph_data}

@app.post("/memory/dream")
async def api_run_dream(user_id: str):
    """Trigger dream cycle memory consolidation for any user ID."""
    result = await run_dream_cycle(user_id)
    return {"status": "success", "user_id": user_id, "result": result}

@app.post("/procedural/playbook")
async def api_store_playbook(req: StorePlaybookRequest):
    """Store a procedural skill playbook."""
    playbook = await store_skill_playbook(req.user_id, req.task_pattern, req.steps)
    return {"status": "success", "playbook": playbook}

# Mount the additive AGI roadmap router (E1–E37) without disturbing existing routes.
app.include_router(agi_router)

# Mount static files for web application frontend (preserve original behaviour).
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

