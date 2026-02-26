from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app.api.routes import router
from app.core.config import ensure_dirs, settings
from app.db.session import engine, init_db
from app.services.jobs import job_runner
from app.services.report_retention import enforce_report_retention
from app.services.runtime import log_runtime_event

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()
    init_db()
    if settings.report_retention_enabled:
        try:
            with Session(engine) as session:
                enforce_report_retention(
                    session,
                    keep_latest=max(1, int(settings.report_retention_limit)),
                )
        except Exception as exc:
            log_runtime_event("report_retention_failed", {"error": str(exc)})
    job_runner.start()
    log_runtime_event("backend_startup", {"app_name": settings.app_name})


@app.on_event("shutdown")
def shutdown() -> None:
    job_runner.stop()
    log_runtime_event("backend_shutdown", {"app_name": settings.app_name})


app.include_router(router)
