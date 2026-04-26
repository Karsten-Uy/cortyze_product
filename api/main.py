"""FastAPI app entry point."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import analyze, health


def create_app() -> FastAPI:
    app = FastAPI(title="Cortyze BrainScore", version="0.0.1")

    origins = [
        o.strip()
        for o in os.environ.get("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(analyze.router)
    return app


app = create_app()
