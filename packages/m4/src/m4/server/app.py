"""Optional FastAPI HTTP server wrapper around M4Memory (Production Mode)."""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

try:
    from fastapi import FastAPI, HTTPException, Body
except ImportError:
    # Allow importing m4 package even if fastapi is not installed
    FastAPI = None


class SessionPayload(BaseModel):
    id: str
    turns: List[Dict[str, str]]
    timestamp: Optional[datetime] = None


class QueryPayload(BaseModel):
    query: str
    dialogue_context: Optional[str] = None


class MaintainPayload(BaseModel):
    now: Optional[datetime] = None


def create_app(memory_engine) -> Any:
    """Create a FastAPI application delegating to the provided M4Memory instance."""
    if FastAPI is None:
        raise RuntimeError("fastapi is required to run server/app.py. Install with: pip install fastapi uvicorn")

    app = FastAPI(title="M4 Long-Term Memory API", version="1.0.0")

    @app.post("/v1/{ns}/sessions")
    def ingest_session(ns: str, payload: SessionPayload):
        from m4.types import Session, Turn
        session_obj = Session(
            id=payload.id,
            turns=[Turn(speaker=t["speaker"], text=t["text"]) for t in payload.turns],
            timestamp=payload.timestamp,
        )
        report = memory_engine.ingest_session(ns, session_obj)
        return report.model_dump()

    @app.post("/v1/{ns}/retrieve")
    def retrieve_memory(ns: str, payload: QueryPayload):
        result = memory_engine.retrieve(ns, payload.query, dialogue_context=payload.dialogue_context)
        return result.model_dump()

    @app.post("/v1/{ns}/respond")
    def generate_response(ns: str, payload: QueryPayload):
        resp = memory_engine.respond(ns, payload.query, dialogue_context=payload.dialogue_context)
        return resp.model_dump()

    @app.post("/v1/{ns}/maintain")
    def run_maintenance(ns: str, payload: Optional[MaintainPayload] = None):
        target_time = payload.now if (payload and payload.now) else datetime.now(timezone.utc)
        count = memory_engine.maintain(ns, target_time)
        return {"compressed_nodes": count, "timestamp": target_time.isoformat()}

    @app.delete("/v1/{ns}")
    def reset_namespace(ns: str):
        memory_engine.reset(ns)
        return {"status": "cleared", "namespace": ns}

    @app.get("/v1/{ns}/stats")
    def get_stats(ns: str):
        stats = memory_engine.stats(ns)
        return stats.model_dump()

    return app
