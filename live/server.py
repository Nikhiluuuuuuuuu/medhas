"""FastAPI Backend Server for Medhas 6-in-1 Memory Engine Web Interface."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from infrastructure.db import DatabasePool, initialize_schema
from memory.session import create_session, get_transcript
from memory.working import get_blocks, update_block, create_memory_block, delete_memory_block, audit_memory_doctor, auto_archive_context_window
from memory.atomic import insert_fact, search_facts, search_facts_dual_level, get_all_active_facts, run_dream_cycle
from memory.graph import upsert_node, update_edge, query_subgraph, query_point_in_time, run_spreading_activation, export_knowledge_graph
from memory.procedural import get_skill_playbook, store_skill_playbook
from pipeline import UnifiedMemoryEngine

app = FastAPI(title="Medhas Memory Engine Live Control Center", version="1.0.0")

engine = UnifiedMemoryEngine()

# Input Pydantic Schemas
class ChatRequest(BaseModel):
    user_id: str = "production_demo_user"
    session_id: str
    message: str

class CreateBlockRequest(BaseModel):
    user_id: str = "production_demo_user"
    label: str
    description: str
    value: str = ""
    limit_tokens: int = 1000

class DeleteBlockRequest(BaseModel):
    user_id: str = "production_demo_user"
    label: str

class SearchFactsRequest(BaseModel):
    user_id: str = "production_demo_user"
    query: str
    limit: int = 5
    session_id: Optional[str] = None

class InsertFactRequest(BaseModel):
    user_id: str = "production_demo_user"
    fact_text: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None

class PPRRequest(BaseModel):
    user_id: str = "production_demo_user"
    seed_nodes: List[str]

@app.on_event("startup")
async def startup_event():
    await DatabasePool.initialize()
    await initialize_schema()

@app.on_event("shutdown")
async def shutdown_event():
    await DatabasePool.close()

# API Endpoints
@app.get("/api/health")
async def get_health():
    return {"status": "online", "engine": "Medhas 6-in-1 Unified Memory System", "version": "1.0.0"}

@app.post("/api/session/create")
async def api_create_session(user_id: str = Query("production_demo_user")):
    sess = await create_session(user_id)
    return {"session_id": str(sess.id), "user_id": user_id, "created_at": sess.created_at.isoformat()}

@app.get("/api/session/transcript")
async def api_get_transcript(session_id: str):
    try:
        sid = UUID(session_id)
        transcript = await get_transcript(sid, limit=20)
        return {"session_id": session_id, "messages": [m.model_dump() for m in transcript]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    try:
        sid = UUID(req.session_id)
        start_ts = asyncio.get_event_loop().time()
        response_text = await engine.execute_turn(req.user_id, sid, req.message)
        latency_ms = round((asyncio.get_event_loop().time() - start_ts) * 1000, 2)
        return {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "response": response_text,
            "latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/working-memory")
async def api_get_working_memory(user_id: str = Query("production_demo_user")):
    record = await get_blocks(user_id)
    return {"user_id": user_id, "blocks": record.blocks.model_dump()}

@app.post("/api/working-memory/block")
async def api_create_block(req: CreateBlockRequest):
    res = await create_memory_block(req.user_id, req.label, req.description, req.value, req.limit_tokens)
    return res

@app.delete("/api/working-memory/block")
async def api_delete_block(user_id: str, label: str):
    res = await delete_memory_block(user_id, label)
    return res

@app.get("/api/working-memory/doctor")
async def api_memory_doctor(user_id: str = Query("production_demo_user")):
    res = await audit_memory_doctor(user_id)
    return res

@app.post("/api/atomic-memory/search")
async def api_search_facts(req: SearchFactsRequest):
    sid = UUID(req.session_id) if req.session_id else None
    results = await search_facts(req.user_id, req.query, limit=req.limit, session_id=sid)
    return {"query": req.query, "results": [r.model_dump() for r in results]}

@app.post("/api/atomic-memory/dual-search")
async def api_dual_search(req: SearchFactsRequest):
    sid = UUID(req.session_id) if req.session_id else None
    results = await search_facts_dual_level(req.user_id, req.query, limit=req.limit, session_id=sid)
    return results

@app.post("/api/atomic-memory/fact")
async def api_insert_fact(req: InsertFactRequest):
    sid = UUID(req.session_id) if req.session_id else None
    fact = await insert_fact(req.user_id, req.fact_text, session_id=sid, agent_id=req.agent_id)
    return fact.model_dump()

@app.get("/api/atomic-memory/facts")
async def api_get_all_facts(user_id: str = Query("production_demo_user")):
    facts = await get_all_active_facts(user_id)
    return {"user_id": user_id, "total": len(facts), "facts": facts}

@app.post("/api/atomic-memory/dream-cycle")
async def api_run_dream_cycle(user_id: str = Query("production_demo_user")):
    res = await run_dream_cycle(user_id)
    return res

@app.get("/api/graph")
async def api_get_graph(user_id: str = Query("production_demo_user")):
    res = await export_knowledge_graph(user_id)
    return res

@app.post("/api/graph/ppr")
async def api_run_ppr(req: PPRRequest):
    edges = await run_spreading_activation(req.user_id, req.seed_nodes)
    return {"seed_nodes": req.seed_nodes, "activated_edges": edges}

@app.get("/api/procedural")
async def api_get_procedural(user_id: str = Query("production_demo_user"), task: str = Query("deploy rust microservice")):
    pb = await get_skill_playbook(user_id, task)
    return {"user_id": user_id, "task": task, "playbook": pb}

# Serve Frontend HTML
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("live.server:app", host="127.0.0.1", port=8000, reload=True)
