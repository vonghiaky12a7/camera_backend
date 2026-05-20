# backend/app/services/stats_service.py
# =============================================================================
# Dashboard Stats Service
#
# Customer count logic (v2):
#   - "Customers inside"  → COUNT res_partner WHERE current_floor_id IS NOT NULL
#   - "Waiting / unknown" → COUNT distinct unknown faces in
#     camera_scan_history_result with scanned_at in last UNKNOWN_WINDOW_MIN minutes
#
# Tối ưu hóa: Bulk/Batch Execute khi Update DB
# =============================================================================

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import TypedDict

from sqlalchemy import text

from app.core.database import get_db_session
from app.core.redis import publish_event

logger = logging.getLogger(__name__)

UNKNOWN_WINDOW_MIN = 30


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


async def _compute_stats(db) -> GlobalStats:
    camera_sql = text("""
        SELECT
            floor_id,
            COUNT(*)                                   AS total,
            COUNT(*) FILTER (WHERE status = 'online')  AS online_count,
            COUNT(*) FILTER (WHERE status = 'offline') AS offline_count
        FROM camera_camera
        WHERE active = TRUE
          AND floor_id IS NOT NULL
        GROUP BY floor_id
    """)
    camera_rows = (await db.execute(camera_sql)).mappings().all()

    camera_by_floor: dict[int, dict] = {
        row["floor_id"]: {
            "total": int(row["total"] or 0),
            "online": int(row["online_count"] or 0),
            "offline": int(row["offline_count"] or 0),
        }
        for row in camera_rows
    }

    customer_sql = text("""
        SELECT
            current_floor_id AS floor_id,
            COUNT(*)         AS customer_count
        FROM res_partner
        WHERE current_floor_id IS NOT NULL
          AND active = TRUE
        GROUP BY current_floor_id
    """)
    customer_rows = (await db.execute(customer_sql)).mappings().all()

    customer_by_floor: dict[int, int] = {
        row["floor_id"]: int(row["customer_count"] or 0) for row in customer_rows
    }

    window_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        minutes=UNKNOWN_WINDOW_MIN
    )

    unknown_sql = text("""
        SELECT
            floor_id,
            COUNT(*) AS waiting_count
        FROM camera_scan_history_result
        WHERE is_matched = FALSE
          AND scanned_at >= :window_start
          AND floor_id IS NOT NULL
        GROUP BY floor_id
    """)
    unknown_rows = (
        (await db.execute(unknown_sql, {"window_start": window_start})).mappings().all()
    )

    unknown_by_floor: dict[int, int] = {
        row["floor_id"]: int(row["waiting_count"] or 0) for row in unknown_rows
    }

    floor_ids = (
        (await db.execute(text("SELECT id FROM camera_floor WHERE active = TRUE")))
        .scalars()
        .all()
    )

    floors: list[FloorStats] = []
    for floor_id in floor_ids:
        cam = camera_by_floor.get(floor_id, {})
        floors.append(
            FloorStats(
                floor_id=floor_id,
                camera_count=cam.get("total", 0),
                camera_online_count=cam.get("online", 0),
                camera_offline_count=cam.get("offline", 0),
                customer_count=customer_by_floor.get(floor_id, 0),
                waiting_customer_count=unknown_by_floor.get(floor_id, 0),
            )
        )

    global_stats = GlobalStats(
        total_cameras=sum(f["camera_count"] for f in floors),
        online_cameras=sum(f["camera_online_count"] for f in floors),
        offline_cameras=sum(f["camera_offline_count"] for f in floors),
        total_customers=sum(f["customer_count"] for f in floors),
        total_waiting=sum(f["waiting_customer_count"] for f in floors),
        floors=floors,
    )
    return global_stats


async def _write_floor_stats(db, floors: list[FloorStats]) -> None:
    if not floors:
        return

    # CHỈ update 2 cột customer_count và waiting_customer_count
    update_sql = text("""
        UPDATE camera_floor
        SET
            customer_count         = :customer_count,
            waiting_customer_count = :waiting_customer_count
        WHERE id = :floor_id
    """)

    bind_params = [
        {
            "floor_id": floor["floor_id"],
            "customer_count": floor["customer_count"],
            "waiting_customer_count": floor["waiting_customer_count"],
        }
        for floor in floors
    ]

    await db.execute(update_sql, bind_params)
    logger.debug("camera_floor counters bulk-updated for %d floor(s).", len(floors))


async def update_dashboard_stats() -> GlobalStats:
    async with get_db_session() as db:
        global_stats = await _compute_stats(db)
        await _write_floor_stats(db, global_stats["floors"])
        await db.commit()  # Add commit after executing the updates

    try:
        await publish_event(
            {
                "type": "dashboard_stats_updated",
                "data": global_stats,
            }
        )
    except Exception as exc:
        logger.warning("Failed to publish stats to Redis: %s", exc)

    return global_stats
