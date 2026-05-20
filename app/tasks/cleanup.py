# backend/app/tasks/cleanup.py
import logging
import os
import time
from celery import shared_task
from app.core.config import settings  # Sử dụng settings đã cấu hình
logger = logging.getLogger(__name__)


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
