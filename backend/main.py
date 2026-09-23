"""
ResearchMind — FastAPI Application
All HTTP endpoints + SSE stream for the agent pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Annotated, Optional

import aiofiles
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from core.config          import get_settings
from core.request_context import set_request_api_key
from core.models  import (
    AgentStatusUpdate, ChatRequest, DiagnosticRequest,
    ExplainSectionRequest, Mode, QuizSubmitRequest,
    SessionState, StartSessionRequest, StartSessionResponse,
)
from agents.pipeline      import stream_mode2_pipeline
from agents.explainer_agent import explain_section, explain_section_stream, answer_question, ingest_pdf
from agents.quiz_agent    import grade_quiz, analyse_weak_spots
from agents.export_agent  import export_session_pdf
from agents.discovery_agent import run_discovery_agent

log = logging.getLogger(__name__)
cfg = get_settings()


# ── Per-request API key injection ─────────────────────────────────────────────

async def inject_api_key(
    request: Request,
    x_openai_key: Optional[str] = Header(default=None),
) -> None:
    # Accept key from header (regular requests) or query param (SSE EventSource)
    key = x_openai_key or request.query_params.get("api_key") or None
    set_request_api_key(key)


# ── In-memory session store (swap for Redis in production) ────────────────────
_SESSIONS: dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    if session_id not in _SESSIONS:
        raise HTTPException(404, f"Session {session_id} not found")
    return _SESSIONS[session_id]


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ResearchMind API", version="1.0.0", dependencies=[Depends(inject_api_key)])


@app.on_event("startup")
async def _startup():
    from core.database import init_db, migrate_from_cache
    await asyncio.to_thread(init_db)
    await asyncio.to_thread(migrate_from_cache)

from fastapi.responses import JSONResponse
from core.clients import MissingAPIKeyError


@app.exception_handler(MissingAPIKeyError)
async def _missing_key_handler(request: Request, exc: MissingAPIKeyError):
    return JSONResponse(status_code=401, content={"detail": str(exc)})


_extra_origins = [o.strip() for o in __import__("os").getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[cfg.frontend_url, "http://localhost:5173", "http://localhost:3000", *_extra_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Session lifecycle ─────────────────────────────────────────────────────────

@app.post("/sessions", response_model=StartSessionResponse)
async def create_session(req: StartSessionRequest):
    """Create a new session and return its ID."""
    session = SessionState(
        mode=req.mode,
        query=req.query if req.mode == Mode.SINGLE_PAPER else "",
        topic=req.query if req.mode == Mode.TOPIC_MASTERY else "",
    )
    _SESSIONS[session.session_id] = session
    log.info("Created session %s (mode=%s)", session.session_id, req.mode)
    return StartSessionResponse(session_id=session.session_id)


@app.get("/sessions/{session_id}")
async def get_session_state(session_id: str):
    """Return the full current state of a session."""
    session = get_session(session_id)
    return session.model_dump()


# ── Diagnostic ────────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/diagnostic")
async def submit_diagnostic(session_id: str, req: DiagnosticRequest):
    """
    Accept diagnostic answers, build user profile, and kick off the pipeline
    via Server-Sent Events stream.
    """
    session = get_session(session_id)
    session.diagnostic_ans = req.answers
    session.topic = req.topic or session.topic
    _SESSIONS[session_id] = session
    return {"ok": True}


# ── Mode 2 pipeline SSE stream ────────────────────────────────────────────────

@app.get("/sessions/{session_id}/pipeline/stream")
async def pipeline_stream(session_id: str):
    """
    Server-Sent Events stream that runs the full Mode 2 pipeline
    and emits progress updates after each agent.
    """
    session = get_session(session_id)

    async def event_generator():
        async def on_update(agent_id: str, status, data: dict):
            payload = json.dumps({
                "agent_id": agent_id,
                "status":   status.value if hasattr(status, "value") else status,
                "data":     {k: v for k, v in (data or {}).items() if v is not None},
                "overall_pct": session.overall_pct,
            })
            yield f"data: {payload}\n\n"

        # We can't yield from a callback easily, so we use a queue
        q: asyncio.Queue = asyncio.Queue()

        async def on_update_q(agent_id, status, data):
            await q.put({"agent_id": agent_id, "status": status, "data": data or {}})

        async def run():
            updated = await stream_mode2_pipeline(session, on_update_q)
            _SESSIONS[session_id] = updated
            await q.put(None)  # sentinel

        task = asyncio.create_task(run())

        while True:
            item = await q.get()
            if item is None:
                yield f"data: {json.dumps({'event': 'done', 'overall_pct': _SESSIONS[session_id].overall_pct})}\n\n"
                break
            yield f"data: {json.dumps({'agent_id': item['agent_id'], 'status': item['status'].value if hasattr(item['status'],'value') else item['status'], 'data': item['data']})}\n\n"

        await task

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── PDF Upload ────────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    """Upload a PDF for RAG ingestion."""
    session = get_session(session_id)
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    save_path = cfg.upload_dir / f"{session_id}_{file.filename}"
    async with aiofiles.open(save_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    try:
        n_chunks = await ingest_pdf(session_id, save_path)
        # Update first paper's PDF path
        if session.papers:
            session.papers[0].pdf_path = str(save_path)
            _SESSIONS[session_id] = session
        return {"ok": True, "chunks_indexed": n_chunks, "filename": file.filename}
    except Exception as e:
        log.error("PDF ingestion failed: %s", e)
        raise HTTPException(500, f"PDF ingestion failed: {e}")


# ── Section explanation ───────────────────────────────────────────────────────

_SECTION_ORDER = ["abstract", "intro", "arch", "attention", "training", "results", "conclusion"]


def _unlock_next(session: SessionState, section: str, session_id: str) -> None:
    idx = _SECTION_ORDER.index(section) if section in _SECTION_ORDER else -1
    if 0 <= idx < len(_SECTION_ORDER) - 1:
        nxt = _SECTION_ORDER[idx + 1]
        if nxt not in session.unlocked_sections:
            session.unlocked_sections.append(nxt)
            _SESSIONS[session_id] = session


@app.get("/sessions/{session_id}/explain/stream")
async def explain_stream_endpoint(
    session_id: str,
    paper_id: str = "",
    section: str = "abstract",
    level_override: str = "",
):
    """SSE stream that emits explanation tokens one chunk at a time."""
    session = get_session(session_id)

    async def gen():
        collected: list[str] = []
        try:
            async for chunk in explain_section_stream(session, paper_id, section, level_override):
                collected.append(chunk)
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Persist completed explanation so PDF export can include it
            if collected:
                pid = paper_id or (session.papers[0].id if session.papers else "no_paper")
                if pid not in session.section_explanations:
                    session.section_explanations[pid] = {}
                session.section_explanations[pid][section] = "".join(collected)
                _SESSIONS[session_id] = session

            _unlock_next(session, section, session_id)
            yield f"data: {json.dumps({'done': True, 'unlocked_sections': session.unlocked_sections})}\n\n"
        except Exception as e:
            log.error("Explain stream failed: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/sessions/{session_id}/explain")
async def explain(session_id: str, req: ExplainSectionRequest):
    """Generate a section explanation for a paper (non-streaming)."""
    session = get_session(session_id)
    try:
        text = await explain_section(session, req.paper_id, req.section, req.level_override)

        # Persist for PDF export
        pid = req.paper_id or (session.papers[0].id if session.papers else "no_paper")
        if pid not in session.section_explanations:
            session.section_explanations[pid] = {}
        session.section_explanations[pid][req.section] = text
        _SESSIONS[session_id] = session

        _unlock_next(session, req.section, session_id)
        return {"explanation": text, "unlocked_sections": session.unlocked_sections}
    except Exception as e:
        log.error("Explainer failed: %s", e)
        raise HTTPException(401 if isinstance(e, MissingAPIKeyError) else 500, str(e))


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest):
    """Answer a follow-up question (Standard / Socratic / Devil's Advocate)."""
    session = get_session(session_id)

    # For Mode 1, use the query as paper title; for Mode 2 pick first paper
    paper_id = session.papers[0].id if session.papers else ""
    section  = req.context or "general"

    try:
        reply = await answer_question(
            session, paper_id, section, req.message, req.chat_mode.value
        )
        # Store turn in history
        session.chat_history.append({"role": "user",      "content": req.message})
        session.chat_history.append({"role": "assistant",  "content": reply})
        session.chat_history = session.chat_history[-40:]   # keep last 20 turns
        _SESSIONS[session_id] = session
        return {"reply": reply}
    except Exception as e:
        log.error("Chat failed: %s", e)
        raise HTTPException(401 if isinstance(e, MissingAPIKeyError) else 500, str(e))


# ── Quiz ──────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/quiz")
async def get_quiz(session_id: str):
    """Return quiz questions for this session."""
    session = get_session(session_id)
    if not session.quiz_questions:
        raise HTTPException(404, "Quiz not yet generated — run the pipeline first")
    return {"questions": [q.model_dump() for q in session.quiz_questions]}


@app.post("/sessions/{session_id}/quiz/submit")
async def submit_quiz(session_id: str, req: QuizSubmitRequest):
    """Grade a submitted quiz."""
    session = get_session(session_id)
    if not session.quiz_questions:
        raise HTTPException(400, "No questions available")

    result      = grade_quiz(session.quiz_questions, req.answers)
    weak_spots  = await analyse_weak_spots(session.quiz_questions, req.answers, session)

    session.quiz_result  = result
    session.overall_pct  = min(session.overall_pct + 10, 95) if result.passed else session.overall_pct
    _SESSIONS[session_id] = session

    return {
        "result":    result.model_dump(),
        "weak_spots":weak_spots,
        "passed":    result.passed,
    }


# ── Mode 1 — quick explain ────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/quick-explain/stream")
async def quick_explain_stream(session_id: str, level_override: str = "expert"):
    """SSE stream for Mode 1: discovers paper then streams explanation tokens."""
    session = get_session(session_id)

    async def gen():
        nonlocal session
        try:
            if not session.papers:
                yield f"data: {json.dumps({'status': 'discovering'})}\n\n"
                session = await run_discovery_agent(session)
                if not session.papers:
                    from core.models import Paper
                    session.papers = [Paper(
                        title=session.query, authors="", year=2024, abstract="", order=1
                    )]
                _SESSIONS[session_id] = session

            yield f"data: {json.dumps({'status': 'explaining'})}\n\n"
            collected: list[str] = []
            async for chunk in explain_section_stream(session, session.papers[0].id, "full", level_override):
                collected.append(chunk)
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Persist for PDF export
            if collected and session.papers:
                pid = session.papers[0].id
                if pid not in session.section_explanations:
                    session.section_explanations[pid] = {}
                session.section_explanations[pid]["full"] = "".join(collected)
                _SESSIONS[session_id] = session

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            log.error("Quick explain stream failed: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/sessions/{session_id}/quick-explain")
async def quick_explain(session_id: str):
    """
    For Mode 1: generate a full paper explanation using world-knowledge (non-streaming).
    """
    session = get_session(session_id)
    if not session.papers:
        session = await run_discovery_agent(session)
        if not session.papers:
            from core.models import Paper
            session.papers = [Paper(
                title=session.query, authors="", year=2024, abstract="", order=1
            )]
        _SESSIONS[session_id] = session

    text = await explain_section(session, session.papers[0].id, "full")
    return {"explanation": text}


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/history")
async def list_history():
    """Return all previously analysed papers / topics from the cache index."""
    from core.cache import get_history
    return {"entries": get_history()}


@app.delete("/history/{entry_id}")
async def delete_history_entry(entry_id: str):
    """Remove a history entry and its cached data so it can be re-analysed fresh."""
    from core.cache import remove_history_entry
    remove_history_entry(entry_id)
    return {"ok": True}


@app.delete("/history")
async def clear_all_history():
    """Wipe every history entry and its associated cached data."""
    from core.cache import clear_history
    clear_history()
    return {"ok": True}


class RestoreRequest(BaseModel):
    entry_id: str


@app.post("/sessions/{session_id}/restore")
async def restore_session_from_cache(session_id: str, req: RestoreRequest):
    """
    Populate a session from a cached pipeline result so the user can jump
    straight to the overview / explainer without re-running the agents.
    """
    from core.cache import get_history, get_pipeline_cache
    from core.models import Paper, KnowledgeGraph, QuizQuestion, AgentStatus

    session = get_session(session_id)

    entries  = get_history()
    entry    = next((e for e in entries if e.get("id") == req.entry_id), None)
    if not entry:
        raise HTTPException(404, "History entry not found")

    if entry.get("mode") == "topic_mastery":
        cached = get_pipeline_cache(entry["query"])
        if not cached:
            raise HTTPException(410, "Cache expired — please re-analyse this topic")
        session.papers         = [Paper(**p)        for p in cached.get("papers", [])]
        session.graph          = KnowledgeGraph(**cached.get("graph", {"nodes": []}))
        session.quiz_questions = [QuizQuestion(**q) for q in cached.get("quiz_questions", [])]
        session.overall_pct    = 88
        session.topic          = entry["query"]
        for aid in ["diagnostic", "discovery", "graph_builder", "scraper", "quiz_gen"]:
            session.agent_statuses[aid] = AgentStatus.DONE
        _SESSIONS[session_id] = session

    return session.model_dump()


# ── Paper database (persistent history) ──────────────────────────────────────

@app.get("/papers")
async def list_papers(mode: str = ""):
    """Return all stored papers from the SQLite database."""
    from core.database import get_records
    records = await asyncio.to_thread(get_records, mode or None)
    return {"papers": records}


@app.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Return a single paper record with its stored explanation."""
    from core.database import get_record
    record = await asyncio.to_thread(get_record, paper_id)
    if not record:
        raise HTTPException(404, "Paper not found in database")
    return record


@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """
    Permanently delete a paper from the database, remove its cached data,
    and delete any generated PDF file.
    """
    from core.database import delete_record
    from core.cache    import remove_history_entry

    pdf_path = await asyncio.to_thread(delete_record, paper_id)
    await asyncio.to_thread(remove_history_entry, paper_id)

    if pdf_path:
        p = Path(pdf_path)
        if p.exists():
            p.unlink(missing_ok=True)

    return {"ok": True}


def _session_from_record(record: dict) -> SessionState:
    """Reconstruct a minimal SessionState from a DB record for PDF generation."""
    from core.models import Paper, KnowledgeGraph

    mode = Mode.SINGLE_PAPER if record.get("mode") == "single_paper" else Mode.TOPIC_MASTERY
    session = SessionState(
        mode=mode,
        query=record.get("title", "") if mode == Mode.SINGLE_PAPER else "",
        topic=record.get("topic", record.get("title", "")) if mode == Mode.TOPIC_MASTERY else "",
        overall_pct=88,
        user_profile={
            "level":             record.get("reader_level", ""),
            "mastered_concepts": record.get("tags", []),
        },
    )

    raw_papers = record.get("papers") or []
    if raw_papers:
        built = []
        for p in raw_papers:
            try:
                built.append(Paper(**p) if isinstance(p, dict) else p)
            except Exception:
                pass
        session.papers = built
    elif mode == Mode.SINGLE_PAPER:
        session.papers = [Paper(
            title=record.get("title", ""),
            authors=record.get("authors", ""),
            year=record.get("year", 0),
            abstract=record.get("abstract", ""),
            arxiv_id=record.get("arxiv_id", ""),
            citations=record.get("citations", 0),
            url=record.get("url", ""),
            tag=record.get("tag", ""),
            order=1,
        )]

    # Embed stored explanation as the first Q&A turn so it appears in the PDF
    explanation = record.get("explanation", "")
    if explanation:
        session.chat_history = [
            {"role": "user",      "content": f"Explain this paper: {record.get('title', '')}"},
            {"role": "assistant", "content": explanation},
        ]

    return session


@app.get("/papers/{paper_id}/export")
async def export_paper_pdf(paper_id: str):
    """
    Generate (or return cached) PDF for a stored paper.
    If a PDF was previously generated for this record it is returned immediately;
    otherwise a new one is generated from the stored data.
    """
    from core.database import get_record, update_pdf_path

    record = await asyncio.to_thread(get_record, paper_id)
    if not record:
        raise HTTPException(404, "Paper not found in database")

    # Return cached PDF if it still exists on disk
    cached_path = record.get("pdf_path", "")
    if cached_path and Path(cached_path).exists():
        return FileResponse(
            cached_path,
            media_type="application/pdf",
            filename=Path(cached_path).name,
        )

    # Generate a fresh PDF from stored data
    try:
        session  = _session_from_record(record)
        pdf_path = export_session_pdf(session)
        await asyncio.to_thread(update_pdf_path, paper_id, str(pdf_path))
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename=pdf_path.name,
        )
    except Exception as e:
        log.error("PDF export from DB failed: %s", e)
        raise HTTPException(401 if isinstance(e, MissingAPIKeyError) else 500, str(e))


# ── Export (session-based) ────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/export")
async def export_pdf(session_id: str):
    """Generate and return the session PDF."""
    session = get_session(session_id)
    try:
        path = export_session_pdf(session)
        return FileResponse(
            str(path),
            media_type="application/pdf",
            filename=path.name,
        )
    except Exception as e:
        log.error("PDF export failed: %s", e)
        raise HTTPException(401 if isinstance(e, MissingAPIKeyError) else 500, str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
