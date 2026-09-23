"""
ResearchMind — single-container entrypoint.

Serves the built React app at "/" and the FastAPI backend under "/api",
so one process (and one port) runs the whole app. Used by the root
Dockerfile for platforms like Render / Railway / Fly.io.

Local dev and docker-compose keep using `main:app` directly.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from main import app as api_app

STATIC_DIR = Path(os.getenv("STATIC_DIR", Path(__file__).parent / "static"))

app = FastAPI(title="ResearchMind", docs_url=None, redoc_url=None)
app.mount("/api", api_app)


@app.on_event("startup")
async def _run_api_startup():
    # Starlette doesn't run a mounted sub-app's startup hooks, so run them here
    for handler in api_app.router.on_startup:
        await handler()


@app.get("/health")
async def health():
    return {"status": "ok"}


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        f = STATIC_DIR / full_path
        if full_path and f.is_file():
            return FileResponse(f)
        return FileResponse(STATIC_DIR / "index.html")
