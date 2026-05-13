# backend/app/tasks/process_event.py
# =============================================================================
# Celery task: process a single camera event
#
# After every recognition result (match / unknown / no_face), this task:
#   1. Runs AI inference (face detection + embedding search)
#   2. Writes the event log to camera_events_log
#   3. Calls update_dashboard_stats() which:
#        a. Computes all 4 counters directly from PostgreSQL
#        b. Writes them into camera_floor rows
#        c. Notifies Odoo to broadcast a bus message → OWL dashboard refreshes
# =============================================================================

import asyncio
import logging
import os
import time
import json
from datetime import datetime

import httpx
from celery import shared_task
from sqlalchemy import select

from app.core.config import settings
from app.core.database import get_db_session
from app.core.redis import publish_event
from app.models.camera_event_log import CameraEventLog, EventType
from app.models.res_partner import ResPartner
from app.models.face_embedding import FaceEmbedding
from app.services.face import FaceAnalyzer, load_image_from_path
from app.services.stats_service import update_dashboard_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async helper for Celery sync context
# ---------------------------------------------------------------------------


def run_async(coro):
    """Hàm bổ trợ để chạy code async trong môi trường sync của Celery."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------


@shared_task(
    bind=True,
    name="app.tasks.process_event.process_camera_event",
    max_retries=3,
)
def process_camera_event(
    self,
    camera_id: str,
    snapshot_filename: str,
    occurred_at_iso: str,
    event_type: str = "unknown",
):
    task_id = self.request.id
    start_time = time.perf_counter()
    snapshot_url = f"/snapshots/{snapshot_filename}"

    logger.info(
        f"[{camera_id}] 🟢 BẮT ĐẦU xử lý Event: {event_type} | Task: {task_id[:8]}"
    )

    base_event_payload = {
        "camera_id": camera_id,
        "timestamp": occurred_at_iso,
        "image_url": snapshot_url,
        "camera_trigger": event_type,
        "event_result": "processing",
    }

    try:
        t_detect = time.perf_counter()
        analyzer = FaceAnalyzer.get()
        image = load_image_from_path(
            os.path.join(settings.snapshot_dir, snapshot_filename)
        )

        faces = analyzer.extract(image)
        detect_time = (time.perf_counter() - t_detect) * 1000

        valid_faces = [
            f for f in faces if f.det_score >= settings.insightface_det_thresh
        ]

        if not valid_faces:
            logger.info(
                f"[{camera_id}] 🟡 KHÔNG tìm thấy mặt nào đạt chuẩn (Thời gian: {detect_time:.1f}ms)"
            )
            base_event_payload["event_result"] = "no_face"
            run_async(
                _handle_unknown(
                    camera_id,
                    snapshot_filename,
                    snapshot_url,
                    occurred_at_iso,
                    task_id,
                    start_time,
                    base_event_payload,
                    is_no_face=True,
                )
            )
            return {"status": "no_face"}

        logger.info(
            f"[{camera_id}] 👥 TÌM THẤY {len(valid_faces)} KHUÔN MẶT ĐẠT CHUẨN trong {detect_time:.1f}ms"
        )

        # =====================================================================
        # 💾 TÍNH NĂNG MỚI: LƯU EMBEDDING RA FILE JSON
        # =====================================================================
        json_data = []
        for idx, face in enumerate(valid_faces):
            json_data.append(
                {
                    "face_index": idx + 1,
                    "det_score": round(float(face.det_score), 4),
                    "bbox": face.bbox.tolist(),
                    "embedding": face.to_list(),
                }
            )

        # Tạo file JSON cùng tên với file ảnh (đổi đuôi .jpg thành .json)
        json_filename = snapshot_filename.rsplit(".", 1)[0] + ".json"
        json_filepath = os.path.join(settings.snapshot_dir, json_filename)

        try:
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[{camera_id}] 💾 Đã lưu dữ liệu {len(valid_faces)} khuôn mặt vào JSON: {json_filename}"
            )
        except Exception as e:
            logger.error(f"[{camera_id}] Lỗi khi lưu file JSON: {e}")
        # =====================================================================

        matched_count = 0
        unknown_count = 0

        for idx, face in enumerate(valid_faces):
            t_search = time.perf_counter()
            match_result = run_async(_vector_search_odoo(face.to_list()))
            search_time = (time.perf_counter() - t_search) * 1000

            event_payload = base_event_payload.copy()
            event_payload["bbox"] = face.bbox.tolist()

            if match_result:
                partner, distance = match_result
                confidence = max(0.0, 1.0 - distance)

                logger.info(
                    f"[{camera_id}] ✅ MẶT #{idx+1}: {partner.name} (Conf: {confidence:.2%}, Dist: {distance:.3f}) - Search: {search_time:.1f}ms"
                )

                event_payload.update(
                    {
                        "event_result": "face_match",
                        "partner_id": partner.id,
                        "partner_name": partner.name,
                        "confidence": round(confidence, 4),
                        "distance_score": distance,
                    }
                )

                run_async(
                    _handle_match(
                        camera_id=camera_id,
                        partner=partner,
                        distance=distance,
                        confidence=confidence,
                        snapshot_filename=snapshot_filename,
                        snapshot_url=snapshot_url,
                        occurred_at=occurred_at_iso,
                        task_id=task_id,
                        start_time=start_time,
                        event_payload=event_payload,
                        camera_event_type=event_type,
                    )
                )
                matched_count += 1

            else:
                logger.info(
                    f"[{camera_id}] ❓ MẶT #{idx+1}: KHÁCH LẠ (Search: {search_time:.1f}ms)"
                )
                event_payload["event_result"] = "unknown_face"

                run_async(
                    _handle_unknown(
                        camera_id,
                        snapshot_filename,
                        snapshot_url,
                        occurred_at_iso,
                        task_id,
                        start_time,
                        event_payload,
                        is_no_face=False,
                    )
                )
                unknown_count += 1

        return {
            "status": "processed_multiple",
            "total": len(valid_faces),
            "matched": matched_count,
            "unknown": unknown_count,
        }

    except Exception as exc:
        logger.error(f"[{camera_id}] ❌ LỖI TASK: {str(exc)}", exc_info=True)
        raise self.retry(exc=exc, countdown=5)
    finally:
        try:
            run_async(update_dashboard_stats())
        except Exception as e:
            logger.warning(f"[{camera_id}] Cập nhật stats thất bại: {e}")


async def _handle_unknown(
    camera_id: str,
    snapshot_filename: str,
    snapshot_url: str,
    occurred_at: str,
    task_id: str,
    start_time: float,
    event_payload: dict,
    is_no_face: bool = False,
):
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    await publish_event(event_payload)

    try:
        dt_occurred = datetime.fromisoformat(occurred_at)
        async with get_db_session() as db:
            if is_no_face:
                log_entry = CameraEventLog.for_no_face(
                    camera_id=camera_id,
                    snapshot_filename=snapshot_filename,
                    snapshot_url=snapshot_url,
                    occurred_at=dt_occurred,
                    celery_task_id=task_id,
                    processing_time_ms=elapsed_ms,
                )

            else:
                log_entry = CameraEventLog.for_unknown(
                    camera_id=camera_id,
                    snapshot_filename=snapshot_filename,
                    snapshot_url=snapshot_url,
                    occurred_at=dt_occurred,
                    celery_task_id=task_id,
                    processing_time_ms=elapsed_ms,
                )

            db.add(log_entry)
            await db.commit()
            logger.debug(
                f"[{camera_id}] Đã lưu Log DB ({'No Face' if is_no_face else 'Unknown'})"
            )
    except Exception as exc:
        logger.error(f"[{camera_id}] Ghi DB (Unknown) thất bại: {exc}")


async def _handle_match(
    camera_id: str,
    partner,
    distance: float,
    confidence: float,
    snapshot_filename: str,
    snapshot_url: str,
    occurred_at: str,
    task_id: str,
    start_time: float,
    event_payload: dict,
    camera_event_type: str,
):
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    await publish_event(event_payload)

    try:
        dt_occurred = datetime.fromisoformat(occurred_at)
        async with get_db_session() as db:
            log_entry = CameraEventLog.for_match(
                camera_id=camera_id,
                partner_id=partner.id,
                partner_name=partner.name,
                distance=distance,
                confidence=confidence,
                snapshot_filename=snapshot_filename,
                snapshot_url=snapshot_url,
                occurred_at=dt_occurred,
                celery_task_id=task_id,
                processing_time_ms=elapsed_ms,
            )

            db.add(log_entry)
            await db.commit()
            logger.debug(f"[{camera_id}] Đã lưu Log DB (Face Match)")
    except Exception as exc:
        logger.error(f"[{camera_id}] Ghi DB thất bại: {exc}")

    # Webhook Odoo
    odoo_url = f"{settings.odoo_base_url}/api/v1/cameras/recognition-event"
    webhook_data = {
        "camera_id": camera_id,
        "partner_id": partner.id,
        "confidence": round(confidence, 4),
        "image_url": snapshot_url,
        "occurred_at": occurred_at,
        "event_type": "face_match",
        "secret_key": settings.hik_webhook_secret,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(odoo_url, json=webhook_data)
    except Exception as exc:
        logger.error(f"[{camera_id}] Gửi Odoo thất bại: {exc}")


# ---------------------------------------------------------------------------
# Internal async helpers
# ---------------------------------------------------------------------------


async def _finish_event(event_payload: dict) -> None:
    """Publish the event to Redis Pub/Sub (no DB match to log)."""
    await publish_event(event_payload)


async def _vector_search_odoo(embedding: list[float]):
    """Search the face_embedding table for the closest vector match."""
    async with get_db_session() as db:
        distance_expr = FaceEmbedding.embedding.l2_distance(embedding)

        stmt = (
            select(ResPartner, distance_expr.label("distance"))
            .select_from(FaceEmbedding)
            .join(ResPartner, FaceEmbedding.partner_id == ResPartner.id)
            .where(FaceEmbedding.active == True)
            .where(ResPartner.active == True)
            .where(FaceEmbedding.embedding != None)
            .where(distance_expr <= settings.vector_match_threshold)
            .order_by(distance_expr)
            .limit(1)
        )

        result = await db.execute(stmt)
        return result.first()
