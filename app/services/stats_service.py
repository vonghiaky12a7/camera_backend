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
from datetime import datetime, time, timezone
from typing import TypedDict
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.core.database import get_db_session
from app.core.redis import publish_event

logger = logging.getLogger(__name__)

BUSINESS_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


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


async def _build_non_employee_filter(db) -> str:
    res_partner_columns = set(
        (
            await db.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'res_partner'
                    """
                )
            )
        )
        .scalars()
        .all()
    )
    hr_employee_columns = set(
        (
            await db.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'hr_employee'
                    """
                )
            )
        )
        .scalars()
        .all()
    )

    clauses = [
        """
        NOT EXISTS (
            SELECT 1
            FROM res_users u
            WHERE u.partner_id = p.id
              AND u.active = TRUE
        )
        """
    ]

    if "employees_count" in res_partner_columns:
        clauses.append("COALESCE(p.employees_count, 0) = 0")

    if "employee" in res_partner_columns:
        clauses.append("COALESCE(p.employee, FALSE) = FALSE")

    employee_links = []
    if "work_contact_id" in hr_employee_columns:
        employee_links.append("e.work_contact_id = p.id")
    if "address_home_id" in hr_employee_columns:
        employee_links.append("e.address_home_id = p.id")
    if "user_id" in hr_employee_columns:
        employee_links.append(
            """
            EXISTS (
                SELECT 1
                FROM res_users eu
                WHERE eu.id = e.user_id
                  AND eu.partner_id = p.id
            )
            """
        )

    if employee_links:
        active_clause = "e.active IS NOT FALSE AND " if "active" in hr_employee_columns else ""
        clauses.append(
            f"""
            NOT EXISTS (
                SELECT 1
                FROM hr_employee e
                WHERE {active_clause}({ ' OR '.join(employee_links) })
            )
            """
        )

    return " AND ".join(f"({clause})" for clause in clauses)


async def _compute_stats(db) -> GlobalStats:
    today = datetime.now(BUSINESS_TZ).date()
    day_start = datetime.combine(today, time.min, tzinfo=BUSINESS_TZ)
    day_end = datetime.combine(today, time.max, tzinfo=BUSINESS_TZ)
    day_start_utc = day_start.astimezone(timezone.utc).replace(tzinfo=None)
    day_end_utc = day_end.astimezone(timezone.utc).replace(tzinfo=None)
    non_employee_filter = await _build_non_employee_filter(db)

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

    customer_sql = text(f"""
        WITH inside_customers AS (
            SELECT
                p.id,
                p.current_floor_id AS floor_id
            FROM res_partner p
            WHERE p.current_floor_id IS NOT NULL
              AND p.active = TRUE
              AND {non_employee_filter}
        ),
        handled_today AS (
            SELECT DISTINCT m.partner_id
            FROM medical_record m
            JOIN inside_customers ic ON ic.id = m.partner_id
            WHERE m.date_planned >= :day_start
              AND m.date_planned <= :day_end
              AND m.state IN ('in_progress', 'completed')
        )
        SELECT
            ic.floor_id,
            COUNT(*) AS customer_count,
            COUNT(*) FILTER (WHERE ht.partner_id IS NULL) AS waiting_count
        FROM inside_customers ic
        LEFT JOIN handled_today ht ON ht.partner_id = ic.id
        GROUP BY ic.floor_id
    """)
    customer_rows = (
        await db.execute(
            customer_sql,
            {"day_start": day_start_utc, "day_end": day_end_utc},
        )
    ).mappings().all()

    customer_by_floor: dict[int, int] = {}
    waiting_by_floor: dict[int, int] = {}
    for row in customer_rows:
        floor_id = row["floor_id"]
        customer_by_floor[floor_id] = int(row["customer_count"] or 0)
        waiting_by_floor[floor_id] = max(int(row["waiting_count"] or 0), 0)

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
                waiting_customer_count=waiting_by_floor.get(floor_id, 0),
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
