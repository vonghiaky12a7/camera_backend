# backend/app/services/stats_service.py
# =============================================================================
# Dashboard Stats Service
#
# Responsibilities:
#   1. Compute all 4 dashboard counters by querying Odoo's PostgreSQL directly:
#        - Total cameras (per floor & global)
#        - Online cameras (per floor & global)
#        - Customer count   = recognized partners seen TODAY (per floor & global)
#        - Waiting count    = unknown faces detected TODAY   (per floor & global)
#
#   2. Write the computed values DIRECTLY into camera_floor rows via SQL UPDATE.
#      The AI backend is the sole writer for these counter columns.
#
#   3. Notify Odoo via a lightweight HTTP call so Odoo can broadcast an Odoo Bus
#      message → the OWL dashboard refreshes without polling.
#
# Design decisions:
#   - All heavy SQL runs in one async session to minimize round-trips.
#   - We use explicit UPDATE ... WHERE id = :id so we never touch columns
#     managed by Odoo (write_date, __last_update, mail thread fields, etc.).
#   - customer_count and waiting_customer_count are DAILY counters reset each
#     time this function runs (they reflect TODAY's activity, not all-time).
#   - This service is called by the Celery task after every processed event,
#     so the dashboard stays near-real-time without a separate cron job.
# =============================================================================

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import TypedDict

import httpx
from sqlalchemy import func, select, text, update

from app.core.config import settings
from app.core.database import get_db_session
from app.models.camera import CameraCamera, CameraFloor, CameraZone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class FloorStats(TypedDict):
    floor_id: int
    camera_count: int
    camera_online_count: int
    camera_offline_count: int
    customer_count: int
    waiting_customer_count: int


class GlobalStats(TypedDict):
    total_cameras: int
    online_cameras: int
    offline_cameras: int
    total_customers: int
    total_waiting: int
    floors: list[FloorStats]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


async def _compute_stats(db) -> GlobalStats:
    """
    Run all aggregation queries inside a single DB session.

    Returns a GlobalStats dict with both per-floor and global totals.
    """

    # ── 1. Camera counts grouped by floor ────────────────────────────────────
    # Join camera_camera → camera_zone → camera_floor to group by floor_id.
    # camera_camera.floor_id is a denormalized stored related field in Odoo,
    # so we can query it directly without going through zone.

    # Use raw SQL for clarity and reliability with boolean aggregation
    camera_sql = text("""
        SELECT
            floor_id,
            COUNT(*)                                    AS total,
            COUNT(*) FILTER (WHERE status = 'online')   AS online_count,
            COUNT(*) FILTER (WHERE status = 'offline')  AS offline_count
        FROM camera_camera
        WHERE active = TRUE
          AND floor_id IS NOT NULL
        GROUP BY floor_id
    """)

    camera_rows = (await db.execute(camera_sql)).mappings().all()

    # Build dict: floor_id → {total, online, offline}
    camera_by_floor: dict[int, dict] = {}
    for row in camera_rows:
        camera_by_floor[row["floor_id"]] = {
            "total": row["total"] or 0,
            "online": row["online_count"] or 0,
            "offline": row["offline_count"] or 0,
        }

    # ── 2. Face recognition event counts TODAY grouped by floor ──────────────
    # We join camera_events_log → camera_camera (via camera_id = code/mac)
    # to map events → floor.
    #
    # customer_count   = FACE_MATCH events today (recognized partners)
    # waiting_count    = UNKNOWN_FACE events today (unregistered visitors)
    #
    # "Today" = from midnight UTC of the current day.

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    recognition_sql = text("""
        SELECT
            cc.floor_id,
            COUNT(*) FILTER (
                WHERE cel.event_type = 'face_match'
                  AND cel.processing_status = 'success'
            )  AS customer_count,
            COUNT(*) FILTER (
                WHERE cel.event_type = 'unknown_face'
                  AND cel.processing_status = 'success'
            )  AS waiting_count
        FROM camera_events_log cel
        JOIN camera_camera cc
          ON cc.code = cel.camera_id   -- camera_id in log = camera code (MAC-derived)
        WHERE cel.occurred_at >= :today_start
          AND cc.active = TRUE
          AND cc.floor_id IS NOT NULL
        GROUP BY cc.floor_id
    """)

    recognition_rows = (
        (await db.execute(recognition_sql, {"today_start": today_start}))
        .mappings()
        .all()
    )

    recognition_by_floor: dict[int, dict] = {}
    for row in recognition_rows:
        recognition_by_floor[row["floor_id"]] = {
            "customers": row["customer_count"] or 0,
            "waiting": row["waiting_count"] or 0,
        }

    # ── 3. Load all active floors ─────────────────────────────────────────────
    floor_rows = (
        (await db.execute(select(CameraFloor.id).where(CameraFloor.active == True)))
        .scalars()
        .all()
    )

    # ── 4. Assemble per-floor stats ───────────────────────────────────────────
    floors: list[FloorStats] = []
    for floor_id in floor_rows:
        cam = camera_by_floor.get(floor_id, {})
        rec = recognition_by_floor.get(floor_id, {})

        floors.append(
            FloorStats(
                floor_id=floor_id,
                camera_count=cam.get("total", 0),
                camera_online_count=cam.get("online", 0),
                camera_offline_count=cam.get("offline", 0),
                customer_count=rec.get("customers", 0),
                waiting_customer_count=rec.get("waiting", 0),
            )
        )

    # ── 5. Global totals ──────────────────────────────────────────────────────
    global_stats = GlobalStats(
        total_cameras=sum(f["camera_count"] for f in floors),
        online_cameras=sum(f["camera_online_count"] for f in floors),
        offline_cameras=sum(f["camera_offline_count"] for f in floors),
        total_customers=sum(f["customer_count"] for f in floors),
        total_waiting=sum(f["waiting_customer_count"] for f in floors),
        floors=floors,
    )

    logger.info(
        "Stats computed — cameras: %d (online: %d) | customers: %d | waiting: %d",
        global_stats["total_cameras"],
        global_stats["online_cameras"],
        global_stats["total_customers"],
        global_stats["total_waiting"],
    )

    return global_stats


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


async def _write_floor_stats(db, floors: list[FloorStats]) -> None:
    """
    Write computed stats directly into camera_floor rows.

    Uses an explicit parameterised UPDATE for each floor so we never
    accidentally overwrite Odoo-managed columns.
    """
    if not floors:
        logger.debug("No floors to update.")
        return

    update_sql = text("""
        UPDATE camera_floor
        SET
            camera_count          = :camera_count,
            camera_online_count   = :camera_online_count,
            camera_offline_count  = :camera_offline_count,
            customer_count        = :customer_count,
            waiting_customer_count = :waiting_customer_count
        WHERE id = :floor_id
    """)

    for floor in floors:
        await db.execute(
            update_sql,
            {
                "floor_id": floor["floor_id"],
                "camera_count": floor["camera_count"],
                "camera_online_count": floor["camera_online_count"],
                "camera_offline_count": floor["camera_offline_count"],
                "customer_count": floor["customer_count"],
                "waiting_customer_count": floor["waiting_customer_count"],
            },
        )

    logger.debug("camera_floor updated for %d floor(s).", len(floors))


# ---------------------------------------------------------------------------
# Odoo Bus notification (lightweight — Odoo only needs to broadcast)
# ---------------------------------------------------------------------------


async def _notify_odoo(global_stats: GlobalStats) -> None:
    """
    POST a compact stats payload to the Odoo webhook endpoint.
    Odoo's controller will:
      1. Read nothing from this payload (data is already in DB).
      2. Simply call bus.bus._sendone() to push a notification to the OWL
         dashboard so it re-fetches from the DB.

    We fire-and-forget: a failure here does NOT roll back the DB write.
    """
    url = f"{settings.odoo_base_url}/api/v1/cameras/stats-updated"

    payload = {
        "secret_key": settings.hik_webhook_secret,
        "total_cameras": global_stats["total_cameras"],
        "online_cameras": global_stats["online_cameras"],
        "total_customers": global_stats["total_customers"],
        "total_waiting": global_stats["total_waiting"],
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                logger.warning(
                    "Odoo stats-updated webhook returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
            else:
                logger.debug("Odoo notified of stats update.")
    except Exception as exc:
        # Non-fatal: the DB is already updated; frontend will catch up on
        # next event or on next manual refresh.
        logger.warning("Failed to notify Odoo of stats update: %s", exc)


# ---------------------------------------------------------------------------
# Public entry point — called by Celery tasks
# ---------------------------------------------------------------------------


async def update_dashboard_stats() -> GlobalStats:
    """
    Full pipeline:
      1. Compute all stats from DB.
      2. Write per-floor counters directly to camera_floor table.
      3. Notify Odoo to broadcast a bus message → OWL dashboard refreshes.

    Returns the computed GlobalStats for optional logging / chaining.

    Usage inside a Celery task::

        from app.services.stats_service import update_dashboard_stats

        stats = run_async(update_dashboard_stats())
    """
    async with get_db_session() as db:
        global_stats = await _compute_stats(db)
        await _write_floor_stats(db, global_stats["floors"])
        # Session commits automatically on context manager exit

    # Notify Odoo AFTER commit so data is visible in DB before Odoo reads it
    await _notify_odoo(global_stats)

    return global_stats
