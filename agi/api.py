"""Roadmap AGI-memory HTTP endpoints (E1–E37).

Additive FastAPI router. Imported once from server.py so the original route handlers
are untouched. Exposes the new AGI capabilities over REST: remember/recall with the
full pipeline, prospective memory, consolidation, forgetting/security, export, and
admin metrics. E22/E23 auth + rate-limit are applied as dependency middleware here.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel

from infrastructure.db import DatabasePool
from agi import (
    engine, add_intention, check_cues, list_intentions, complete_intention,
    run_consolidation, run_forgetting_sweep, protect_core_memories,
    forget as agi_forget, export_user_memory, export_to_file, import_user_memory,
    build_user_model, get_user_model, timeline, what_changed, why_chain,
    interference_matrix, resolve_interference, evict_working_memory,
    knowledge_map, known_unknowns, partition_report,
    authenticate, authorize, rate_limiter,
)

router = APIRouter(prefix="/agi", tags=["agi-memory-roadmap"])


# ----- request / response models --------------------------------------------

class RememberReq(BaseModel):
    user_id: str
    text: str
    session_id: Optional[str] = None
    memory_type: str = "semantic"
    source: str = "user"
    affect: Optional[Dict[str, float]] = None
    force_admit: bool = False


class RecallReq(BaseModel):
    user_id: str
    query: str
    limit: int = 5
    use_tools: bool = False
    enforce_abstention: bool = True


class IntentionReq(BaseModel):
    user_id: str
    intent: str
    cue_text: Optional[str] = None
    trigger_at: Optional[str] = None   # ISO timestamp


# ----- auth + rate-limit dependency -----------------------------------------

async def _auth(api_key: Optional[str], action: str) -> str:
    rec = await authenticate(api_key or "")
    if not authorize(rec, action):
        raise HTTPException(status_code=401, detail="unauthorized or missing scope")
    key = api_key or (rec["user_id"] if rec else "anon")
    if not rate_limiter.allow(key, action):
        raise HTTPException(status_code=429, detail="rate limit exceeded; slow down")
    return rec["user_id"] if rec else "default"


# ----- remember / recall -----------------------------------------------------

@router.post("/remember")
async def api_remember(req: RememberReq, x_api_key: Optional[str] = Header(None)):
    uid = await _auth(x_api_key, "write")
    if req.user_id != uid and uid != "default":
        # tenant/user scoping: a key may only write its own user unless admin
        raise HTTPException(status_code=403, detail="key not scoped to this user")
    sess = UUID(req.session_id) if req.session_id else None
    result = await engine.remember(
        req.user_id, req.text, session_id=sess, memory_type=req.memory_type,
        source=req.source, affect=req.affect, force_admit=req.force_admit,
    )
    return {"status": "ok", **result}


@router.post("/recall")
async def api_recall(req: RecallReq, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "read")
    import datetime
    result = await engine.recall(
        req.user_id, req.query, limit=req.limit, use_tools=req.use_tools,
        enforce_abstention=req.enforce_abstention,
    )
    return result


# ----- prospective memory ----------------------------------------------------

@router.post("/intentions")
async def api_add_intention(req: IntentionReq, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "write")
    from datetime import datetime, timezone
    trig = None
    if req.trigger_at:
        trig = datetime.fromisoformat(req.trigger_at.replace("Z", "+00:00"))
    iid = await add_intention(req.user_id, req.intent, cue_text=req.cue_text, trigger_at=trig)
    return {"status": "ok", "intention_id": str(iid)}


@router.post("/intentions/fire")
async def api_fire_intentions(user_id: str, context: str = "", x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "read")
    fired = await check_cues(user_id, current_context=context)
    return {"status": "ok", "fired": fired}


@router.get("/intentions")
async def api_list_intentions(user_id: str, include_done: bool = False, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "read")
    return {"status": "ok", "intentions": await list_intentions(user_id, include_done)}


# ----- consolidation / forgetting / security --------------------------------

@router.post("/consolidate")
async def api_consolidate(user_id: str, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "write")
    return {"status": "ok", **await run_consolidation(user_id)}


@router.post("/forget")
async def api_forget(user_id: str, scope: Optional[str] = None, hard: bool = True,
                     x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "write")
    return {"status": "ok", **await agi_forget(user_id, scope, hard=hard)}


@router.get("/quarantine")
async def api_quarantine(user_id: str, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "admin")
    return {"status": "ok", "quarantined": await _list_quar(user_id)}


async def _list_quar(user_id: str):
    from agi.security import list_quarantined
    return await list_quarantined(user_id)


# ----- user model / temporal / causal ----------------------------------------

@router.post("/profile/build")
async def api_build_profile(user_id: str, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "write")
    return {"status": "ok", **await build_user_model(user_id)}


@router.get("/profile")
async def api_get_profile(user_id: str, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "read")
    return {"status": "ok", **await get_user_model(user_id)}


@router.get("/timeline")
async def api_timeline(user_id: str, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "read")
    return {"status": "ok", "timeline": await timeline(user_id)}


@router.get("/metamemory")
async def api_metamemory(user_id: str, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "read")
    return {"status": "ok", "known_unknowns": await known_unknowns(user_id),
            "knowledge_map": await knowledge_map(user_id)}


# ----- interference / working memory ----------------------------------------

@router.get("/interference")
async def api_interference(user_id: str, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "read")
    return {"status": "ok", "matrix": await interference_matrix(user_id)}


@router.post("/working/evict")
async def api_evict_wm(user_id: str, query: str = "", x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "write")
    return {"status": "ok", **await evict_working_memory(user_id, query)}


# ----- export / scaling -----------------------------------------------------

@router.get("/export")
async def api_export(user_id: str, x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "read")
    bundle = await export_user_memory(user_id)
    return {"status": "ok", "rows": sum(len(v) for v in bundle.get("tables", {}).values()),
            "bundle": bundle}


@router.get("/scaling/report")
async def api_scaling(x_api_key: Optional[str] = Header(None)):
    await _auth(x_api_key, "admin")
    return {"status": "ok", **await partition_report()}
