# backend/app/tasks/celery_app.py
# =============================================================================
# Celery application instance and global configuration.
#
# This module is the single source of truth for the Celery app object.
# Import it in task modules and in the celery CLI command:
#   celery -A app.tasks.celery_app worker ...
#
# Design decisions:
#   - Prefork pool is required for CPU-bound ONNX inference (no GIL sharing).
#   - One task queue `face_processing` keeps routing explicit and observable.
#   - Thread-limit env vars are set at OS level in docker-compose; they are
#     also re-applied here as a safety net for bare-metal / local runs.
#   - Worker initializer (`worker_process_init`) loads the InsightFace model
#     ONCE per worker process into a module-level singleton so it is not
#     re-loaded for every task (expensive: ~2-4 s per cold load).
# =============================================================================

from __future__ import annotations
import logging
import os
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Force CPU thread limits as early as possible in the worker process.
# These must be set BEFORE numpy / onnxruntime are imported.
# ---------------------------------------------------------------------------
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _k, _v in _THREAD_ENV.items():
    os.environ.setdefault(_k, _v)


# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------
celery_app = Celery(
    "acs",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.process_event",
        "app.tasks.cleanup",
    ],
)

celery_app.conf.update(
    # ── Serialization ─────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # ── Timezone ──────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,
    # ── Queue routing ─────────────────────────────────────────────────────────
    task_default_queue="face_processing",
    task_queues={
        "face_processing": {
            "exchange": "face_processing",
            "routing_key": "face_processing",
        },
    },
    # ── Execution limits ──────────────────────────────────────────────────────
    # Soft limit: raises SoftTimeLimitExceeded (can be caught for cleanup).
    # Hard limit: SIGKILL after this many seconds regardless.
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_time_limit=settings.celery_task_time_limit,
    # ── Worker pool ───────────────────────────────────────────────────────────
    # prefork = separate OS processes, required for CPU-bound ONNX tasks.
    # concurrency is set via CLI flag (--concurrency) from docker-compose env.
    worker_pool="prefork",
    worker_prefetch_multiplier=1,  # fetch only 1 task per worker process at a time
    # prevents a fast worker from hoarding tasks
    # while a slow worker sits idle
    # ── Result backend ────────────────────────────────────────────────────────
    result_expires=300,  # TTL for task results in Redis (seconds)
    # ── Reliability ───────────────────────────────────────────────────────────
    task_acks_late=True,  # Acknowledge AFTER the task completes,
    # not when it is picked up. Prevents task
    # loss if a worker crashes mid-execution.
    task_reject_on_worker_lost=True,  # Re-queue task if the worker process dies
    # ── Redis broker transport options ───────────────────────────────────────
    broker_transport_options={
        "visibility_timeout": 3600,  # 1 h; must be > task_time_limit
        "socket_keepalive": True,
        "retry_on_timeout": True,
    },
    # ── Logging ───────────────────────────────────────────────────────────────
    worker_hijack_root_logger=False,  # Let our structlog config manage logging
    worker_log_color=False,
    beat_schedule={
        "cleanup-snapshots-every-5-minutes": {
            "task": "app.tasks.cleanup.cleanup_old_snapshots",
            "schedule": 300.0,
        },
        "clear-stale-camera-customers-every-5-minutes": {
            "task": "app.tasks.cleanup.clear_stale_camera_customers",
            "schedule": 300.0,
        },
        # 👉 TASK 10: CHỐT SỔ THỐNG KÊ DASHBOARD MỖI 10 GIÂY
        "update-dashboard-stats-every-10s": {
            "task": "app.tasks.celery_app.periodic_dashboard_sync",
            "schedule": 10.0,
        },
    },
)


# ---------------------------------------------------------------------------
# Worker process lifecycle hooks
# ---------------------------------------------------------------------------


@worker_process_init.connect
def init_worker_process(**kwargs: object) -> None:
    """
    Called once per worker PROCESS after fork.

    We load the InsightFace model here (not at module import time) so that:
      1. The model is only loaded in worker processes, not in the Beat scheduler
         or the FastAPI process.
      2. Each forked process gets its own model instance (no shared memory race).
      3. The model is reused across all tasks handled by the same process.
    """
    logger.info("Worker process initialising — loading InsightFace model…")

    # Lazy import here so the model loader is not executed on every module import
    from app.services.face import FaceAnalyzer  # noqa: PLC0415

    FaceAnalyzer.initialize()

    logger.info("Worker process ready.")


@worker_process_shutdown.connect
def shutdown_worker_process(**kwargs: object) -> None:
    """Called once per worker PROCESS on graceful shutdown."""
    logger.info("Worker process shutting down.")

    from app.services.face import FaceAnalyzer  # noqa: PLC0415

    FaceAnalyzer.teardown()


@celery_app.task(name="app.tasks.celery_app.periodic_dashboard_sync")
def periodic_dashboard_sync():
    """Chạy ngầm định kỳ để cập nhật tổng số lượng khách hàng thay vì gọi mỗi khi Webhook kích hoạt"""
    import asyncio
    from app.services.stats_service import update_dashboard_stats

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(update_dashboard_stats())
