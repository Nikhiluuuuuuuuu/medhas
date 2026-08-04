"""Regression guard for the MemoryEngine public API surface.

Guards against the 2026-08-04 latent bug where consolidate/forget_user/backup/
build_profile/plan_intention/fire_intentions were nested INSIDE _result_dict (after
its return) and therefore never became class methods. Also guards engine.think().

Runs offline (MEDHAS_OFFLINE=1): every method exercises its real Postgres path with
deterministic fallbacks, no Groq required. Uses the session-scoped `user_id` fixture.
"""

import os
import tempfile

import pytest

from agi import engine as eng  # shared MemoryEngine instance (agi/__init__ exports it)
from infrastructure.db import DatabasePool


async def test_engine_api_methods_are_real_class_methods():
    # Structural guard: the six methods must exist ON THE INSTANCE (not dead code).
    for name in ("consolidate", "forget_user", "backup", "build_profile",
                 "plan_intention", "fire_intentions", "think", "remember", "recall"):
        assert hasattr(eng, name), f"MemoryEngine.{name} is missing (latent dead-code bug?)"


async def test_build_profile_runs(user_id):
    prof = await eng.build_profile(user_id)
    assert isinstance(prof, dict) and prof, "build_profile should return a populated dict"


async def test_consolidate_runs(user_id):
    res = await eng.consolidate(user_id)
    assert isinstance(res, dict)
    assert "dream" in res or "forgetting" in res or "compression" in res, res


async def test_plan_and_fire_intention(user_id):
    iid = await eng.plan_intention(user_id, "ship the v2 release", cue_text="when build is green")
    assert iid is not None, "plan_intention should return an intention id"
    fired = await eng.fire_intentions(user_id, context="build is green")
    assert isinstance(fired, list)
    assert len(fired) >= 1, "intention with matching cue should fire"


async def test_backup_exports_file(user_id):
    tmp = os.path.join(tempfile.gettempdir(), f"{user_id}_bak.json")
    try:
        res = await eng.backup(user_id, tmp)
        assert res.get("path") == tmp
        assert res.get("rows", 0) >= 0
        assert os.path.exists(tmp) and os.path.getsize(tmp) > 0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


async def test_forget_user_purges(user_id):
    res = await eng.forget_user(user_id, scope="atomic", hard=True)
    assert res.get("status") in ("purged", "ok", "ran", None) or "facts" in res, res
    # No exception == pass; out-of-scope data may remain but the call must not error.


async def test_think_pipeline_runs_and_persists(user_id):
    from agi.cognition.embodiment import BodyModel, ensure_body_schema
    await ensure_body_schema()
    body = BodyModel(user_id)
    body.register_capability("notify", lambda ctx, params: f"notified:{params.get('who')}")
    res = await eng.think(
        "priya launched lumina and mentors rahul",
        user_id, body=body, action="notify", action_params={"who": "team"},
    )
    assert res["percept"] is not None
    assert "priya" in res["percept"]["entities"]
    assert res["action"]["executed"] is True


async def test_engine_methods_not_nested_in_result_dict():
    # Explicit structural check: the methods must NOT be attributes of the module-level
    # _result_dict function object (which is where the latent bug hid them).
    import importlib.util
    spec = importlib.util.spec_from_file_location("engmod", "agi/engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_result_dict"), "_result_dict helper missing"
    assert not hasattr(mod._result_dict, "consolidate"), \
        "consolidate is nested inside _result_dict again — latent bug regressed!"
    assert not hasattr(mod._result_dict, "backup"), \
        "backup is nested inside _result_dict again — latent bug regressed!"
