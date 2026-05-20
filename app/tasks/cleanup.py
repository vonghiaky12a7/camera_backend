# backend/app/tasks/cleanup.py
import logging
import os
import time
import asyncio
from datetime import datetime, timedelta, timezone
from celery import shared_task
from app.core.config import settings  # Sử dụng settings đã cấu hình
from sqlalchemy import text
from app.core.database import get_db_session
from app.services.stats_service import update_dashboard_stats
logger = logging.getLogger(__name__)

STALE_CUSTOMER_TTL_SECONDS = 3600


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed loop")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@shared_task(name="app.tasks.cleanup.cleanup_old_snapshots")
def cleanup_old_snapshots():
    """
    Quét thư mục chứa raw snapshots và xóa các file cũ dựa trên SNAPSHOT_TTL_SECONDS.
    """
    # Lấy giá trị TTL từ file .env thông qua settings
    ttl_seconds = settings.snapshot_ttl_seconds

    target_dir = settings.raw_snapshot_dir
    now = time.time()
    deleted_count = 0

    if not os.path.exists(target_dir):
        logger.warning(f"Thư mục {target_dir} không tồn tại để dọn dẹp.")
        return deleted_count

    for filename in os.listdir(target_dir):
        filepath = os.path.join(target_dir, filename)

        if os.path.isfile(filepath):
            file_age = now - os.path.getmtime(filepath)

            # Xóa nếu tuổi thọ file lớn hơn giá trị cấu hình
            if file_age > ttl_seconds:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Lỗi khi xóa file {filepath}: {e}")

    if deleted_count > 0:
        logger.info(
            f"Đã dọn dẹp {deleted_count} ảnh snapshots cũ (TTL: {ttl_seconds}s)."
        )

    return deleted_count


async def _clear_stale_customers_async() -> int:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        seconds=STALE_CUSTOMER_TTL_SECONDS
    )

    async with get_db_session() as db:
        result = await db.execute(
            text(
                """
                UPDATE res_partner
                SET current_floor_id = NULL,
                    current_zone_id = NULL,
                    last_seen_time = NULL,
                    last_seen_camera_id = NULL
                WHERE current_floor_id IS NOT NULL
                  AND active = TRUE
                  AND (
                      last_seen_time IS NULL
                      OR last_seen_time < :cutoff
                  )
                """
            ),
            {"cutoff": cutoff},
        )
        cleared_count = result.rowcount or 0

    if cleared_count:
        logger.info(
            "Cleared %d stale camera customer(s) older than %d seconds.",
            cleared_count,
            STALE_CUSTOMER_TTL_SECONDS,
        )
        await update_dashboard_stats()

    return cleared_count


@shared_task(name="app.tasks.cleanup.clear_stale_camera_customers")
def clear_stale_camera_customers():
    return _run_async(_clear_stale_customers_async())
