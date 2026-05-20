# backend/app/main.py
# =============================================================================
# FastAPI entry point.
# =============================================================================

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import cameras
from app.core.config import settings
from app.core.database import connect_db, disconnect_db
from app.core.redis import connect_redis, disconnect_redis
from fastapi.responses import RedirectResponse
from logging.handlers import RotatingFileHandler
from fastapi import WebSocket, WebSocketDisconnect
import os

# Cấu hình logging ghi ra cả Console (màn hình) và File (app.log)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

file_handler = RotatingFileHandler(
    "/tmp/app.log", maxBytes=5 * 1024 * 1024, backupCount=5
)
file_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=settings.log_level.upper(), handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng (Startup & Shutdown)."""
    logger.info("Starting up FastAPI application...")

    # Kết nối DB & Redis
    if settings.service_role == "api":
        await connect_db()
        await connect_redis()

    yield  # Ứng dụng hoạt động tại đây

    logger.info("Shutting down FastAPI application...")
    if settings.service_role == "api":
        await disconnect_db()
        await disconnect_redis()


app = FastAPI(
    title="AI Camera System API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên giới hạn origin của Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Static Files ────────────────────────────────────────────────────────
# Phục vụ thư mục chứa ảnh snapshot tạm thời để UI Next.js có thể hiển thị

os.makedirs(settings.raw_snapshot_dir, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=settings.raw_snapshot_dir), name="snapshots")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(cameras.router, tags=["Cameras"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Endpoint dùng cho Docker Healthcheck."""
    return {"status": "ok", "role": settings.service_role}


@app.get("/", include_in_schema=False)
async def root():
    """Tự động chuyển hướng người dùng từ trang chủ sang giao diện quản lý API (Swagger UI)."""
    return RedirectResponse(url="/docs")
