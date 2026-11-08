"""
ForceCRM Archimedes2026 Web Server (`src/server.py`)
===================================================
High-performance Starlette ASGI server serving the modern JavaScript SPA and REST API.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

# Dynamically locate virtual environment site-packages if present
_venv_base = Path(__file__).resolve().parent.parent / ".venv"
if _venv_base.exists():
    _py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _venv_site = _venv_base / "lib" / _py_ver / "site-packages"
    if _venv_site.exists() and str(_venv_site) not in sys.path:
        sys.path.insert(0, str(_venv_site))

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.service import EngineService, Superseded

_SERVICE_INSTANCE: EngineService | None = None
STATIC_DIR = Path(__file__).resolve().parent / "static"
if not (STATIC_DIR / "index.html").exists():
    STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def get_service() -> EngineService:
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = EngineService()
    return _SERVICE_INSTANCE


# -------------------------------------------------------------------------
# Request Handlers
# -------------------------------------------------------------------------
async def index_handler(request: Request) -> Response:
    """Serve the single-page application entry point."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>ForceCRM Archimedes2026 frontend loading...</h1>", status_code=200)
    return FileResponse(index_file, media_type="text/html")


async def bootstrap_handler(request: Request) -> JSONResponse:
    """Return initial state, parameter schemas, benchmarks and Ollama status."""
    service = get_service()
    data = await asyncio.to_thread(service.bootstrap)
    return JSONResponse(data)


async def run_handler(request: Request) -> JSONResponse:
    """Run interactive calculation with given parameters (instant re-score or fast clustering)."""
    service = get_service()
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    gen = service.next_generation()
    try:
        params = await asyncio.to_thread(service.parse_params, body.get("params", body))
        payload = await asyncio.to_thread(service.run, params, gen)
        return JSONResponse({"status": "ok", "generation": gen, "data": payload})
    except Superseded:
        return JSONResponse({"status": "superseded", "generation": gen}, status_code=409)
    except ValueError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"status": "error", "message": f"Server error: {exc}"}, status_code=500)


async def upload_handler(request: Request) -> JSONResponse:
    """Handle custom client Excel (.xlsx) upload."""
    service = get_service()
    form = await request.form()
    uploaded_file = form.get("file")
    if uploaded_file is None:
        return JSONResponse({"status": "error", "message": "Δεν επιλέχθηκε αρχείο."}, status_code=400)
    
    filename = getattr(uploaded_file, "filename", "clients.xlsx")
    raw_bytes = await uploaded_file.read()
    try:
        meta = await asyncio.to_thread(service.upload_clients, raw_bytes, filename)
        return JSONResponse({"status": "ok", "client_set": meta})
    except ValueError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"status": "error", "message": f"Σφάλμα επεξεργασίας: {exc}"}, status_code=500)


async def company_handler(request: Request) -> JSONResponse:
    """Return comprehensive registry & KAD activity record for a single company."""
    service = get_service()
    company_id_str = request.path_params.get("company_id")
    try:
        cid = int(company_id_str)
    except (ValueError, TypeError):
        return JSONResponse({"status": "error", "message": "Invalid company_id"}, status_code=400)
    
    data = await asyncio.to_thread(service.company, cid)
    if data is None:
        return JSONResponse({"status": "not_found", "message": "Η επιχείρηση δεν βρέθηκε."}, status_code=404)
    return JSONResponse(data)


async def llm_prompt_handler(request: Request) -> JSONResponse:
    """Return the grounded system prompt and user statistical JSON payload for an ICP."""
    service = get_service()
    profile_id = request.path_params.get("profile_id", "")
    try:
        data = await asyncio.to_thread(service.llm_prompt, profile_id)
        return JSONResponse(data)
    except KeyError:
        return JSONResponse({"status": "not_found", "message": "Το προφίλ δεν βρέθηκε."}, status_code=404)


async def llm_generate_handler(request: Request) -> JSONResponse:
    """Trigger live local Ollama generation with temperature=0 and Article 50 Groundedness check."""
    service = get_service()
    profile_id = request.path_params.get("profile_id", "")
    try:
        body = await request.json()
    except Exception:
        body = {}
    model = body.get("model")
    try:
        res = await asyncio.to_thread(service.llm_generate, profile_id, model=model)
        return JSONResponse(res)
    except KeyError:
        return JSONResponse({"status": "not_found", "message": "Το προφίλ δεν βρέθηκε."}, status_code=404)
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)


async def export_handler(request: Request) -> Response:
    """Download official Section 10 deliverable files (profiles.json, assignments.csv, lookalikes.csv)."""
    service = get_service()
    filename = request.path_params.get("filename", "")
    try:
        file_bytes, file_path = await asyncio.to_thread(service.export, filename)
        media_type = "application/json" if filename.endswith(".json") else "text/csv"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
        return Response(file_bytes, media_type=media_type, headers=headers)
    except ValueError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"status": "error", "message": f"Export error: {exc}"}, status_code=500)


async def api_health_handler(request: Request) -> Response:
    """Dedicated standard health check endpoint."""
    return JSONResponse({"status": "ok"})


async def st_health(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def st_host_config(request: Request) -> Response:
    return JSONResponse({"allowedOrigins": ["*"], "useExternalAuthToken": False})


async def exclude_company_handler(request: Request) -> Response:
    try:
        company_id = int(request.path_params["company_id"])
        data = {}
        try:
            data = await request.json()
        except Exception:
            pass
        reason = data.get("reason", "")
        service = get_service()
        res = await asyncio.to_thread(service.exclude_company, company_id, reason=reason)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)


async def unexclude_company_handler(request: Request) -> Response:
    try:
        company_id = int(request.path_params["company_id"])
        service = get_service()
        res = await asyncio.to_thread(service.unexclude_company, company_id)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)


async def get_excluded_handler(request: Request) -> Response:
    service = get_service()
    excluded = await asyncio.to_thread(service.get_excluded_companies)
    return JSONResponse({"excluded": excluded})


async def verify_company_handler(request: Request) -> Response:
    try:
        company_id = int(request.path_params["company_id"])
        service = get_service()
        res = await asyncio.to_thread(service.verify_company, company_id, force=True)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)


async def verify_top_leads_handler(request: Request) -> Response:
    try:
        data = {}
        try:
            data = await request.json()
        except Exception:
            pass
        limit = int(data.get("limit", 25))
        service = get_service()
        res = await asyncio.to_thread(service.verify_top_leads, count=limit)
        return JSONResponse(res)
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)


# -------------------------------------------------------------------------
# Application Factory
# -------------------------------------------------------------------------
def create_app() -> Starlette:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    
    routes = [
        Route("/", endpoint=index_handler, methods=["GET"]),
        Route("/api/health", endpoint=api_health_handler, methods=["GET"]),
        Route("/_stcore/health", endpoint=st_health, methods=["GET"]),
        Route("/_stcore/host-config", endpoint=st_host_config, methods=["GET"]),
        Route("/api/bootstrap", endpoint=bootstrap_handler, methods=["GET"]),
        Route("/api/run", endpoint=run_handler, methods=["POST"]),
        Route("/api/upload", endpoint=upload_handler, methods=["POST"]),
        Route("/api/company/{company_id:int}", endpoint=company_handler, methods=["GET"]),
        Route("/api/verify-company/{company_id:int}", endpoint=verify_company_handler, methods=["GET", "POST"]),
        Route("/api/verify-top-leads", endpoint=verify_top_leads_handler, methods=["POST"]),
        Route("/api/exclude-company/{company_id:int}", endpoint=exclude_company_handler, methods=["POST"]),
        Route("/api/unexclude-company/{company_id:int}", endpoint=unexclude_company_handler, methods=["POST"]),
        Route("/api/excluded-companies", endpoint=get_excluded_handler, methods=["GET"]),
        Route("/api/llm/prompt/{profile_id:str}", endpoint=llm_prompt_handler, methods=["GET"]),
        Route("/api/llm/generate/{profile_id:str}", endpoint=llm_generate_handler, methods=["POST"]),
        Route("/api/export/{filename:str}", endpoint=export_handler, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ]
    
    middleware = [
        Middleware(GZipMiddleware, minimum_size=1000),
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
    
    return Starlette(routes=routes, middleware=middleware)


# Module-level ASGI application instance for uvicorn/gunicorn/test clients
app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8501) -> None:
    import uvicorn
    print("=" * 76)
    print("🚀 Archimedes2026 Executive B2B Intelligence UI")
    print(f"👉 Local Web App Running at: http://{host}:{port}")
    print("=" * 76)
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()

