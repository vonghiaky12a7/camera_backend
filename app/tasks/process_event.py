# backend/app/tasks/process_event.py
# =============================================================================
# Celery task â€“ processes each Hikvision camera event end-to-end.
#
# Changes in this version:
#   1. ÄÃ£ gom táº¥t cáº£ cÃ¡c Database & Redis operations vÃ o Má»˜T Láº¦N gá»i run_async
#   2. Boto3 (AWS) cháº¡y á»Ÿ cháº¿ Ä‘á»™ Äá»’NG Bá»˜ hoÃ n toÃ n trÃ¡nh khÃ³a Event loop
#   3. Khá»Ÿi táº¡o Event Loop chuáº©n xÃ¡c giáº£m thiá»ƒu overhead
# =============================================================================

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np
from celery import shared_task
from sqlalchemy import select, text

from app.core.config import settings
from app.core.database import get_db_session
from app.core.redis import get_redis, publish_event
from app.models.camera_event_log import CameraEventLog
from app.models.camera_scan_history import CameraScanHistory, CameraScanHistoryResult
from app.models.face_embedding import FaceEmbedding
from app.models.res_partner import ResPartner
from app.services.aws_rekognition import config_summary, detect_face_crops, search_face_by_image
from app.services.face import FaceAnalyzer, load_image_from_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_sys_config_cache: dict[str, str] = {}
_sys_config_cache_time: float = 0.0
_SYS_CONFIG_TTL = 120


def run_async(coro):
    """Run a coroutine synchronously inside a Celery prefork worker."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed loop")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def get_system_config() -> dict[str, str]:
    global _sys_config_cache, _sys_config_cache_time
    if time.time() - _sys_config_cache_time < _SYS_CONFIG_TTL:
        return _sys_config_cache

    async with get_db_session() as db:
        rows = (
            await db.execute(
                text(
                    "SELECT key, value FROM ir_config_parameter WHERE key LIKE 'camera.%'"
                )
            )
        ).all()
        _sys_config_cache = {row.key: row.value for row in rows}
        _sys_config_cache_time = time.time()
    return _sys_config_cache


async def _resolve_camera_info(camera_str_id: str) -> dict[str, Any]:
    async with get_db_session() as db:
        res = await db.execute(
            text("""
                SELECT
                    c.id, c.name, c.floor_id, c.zone_id, c.direction,
                    f.name AS floor_name, z.name AS zone_name
                FROM camera_camera c
                LEFT JOIN camera_floor f ON c.floor_id = f.id
                LEFT JOIN camera_zone  z ON c.zone_id  = z.id
                WHERE UPPER(c.mac_address) = UPPER(:val) OR c.ip_address = :val
                ORDER BY CASE WHEN UPPER(c.mac_address) = UPPER(:val) THEN 1 ELSE 2 END
                LIMIT 1
            """),
            {"val": camera_str_id},
        )
        row = res.first()

    if row:
        return {
            "id": row.id,
            "name": row.name or camera_str_id,
            "floor_id": row.floor_id,
            "zone_id": row.zone_id,
            "direction": row.direction or "internal",
            "floor_name": row.floor_name or "",
            "zone_name": row.zone_name or "",
        }

    return {
        "id": None,
        "name": camera_str_id,
        "floor_id": None,
        "zone_id": None,
        "direction": "internal",
        "floor_name": "",
        "zone_name": "",
    }


async def _is_debounced(camera_id: str, entity_id: str, ttl: int = 30) -> bool:
    key = f"debounce:{camera_id}:{entity_id}"
    try:
        async with get_redis() as r:
            if await r.exists(key):
                return True
            await r.set(key, 1, ex=ttl)
            return False
    except Exception:
        return False


def _laplacian_variance(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def _is_blurry(img_bgr: np.ndarray, threshold: float) -> tuple[bool, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    var = _laplacian_variance(gray)
    return var < threshold, var


FULL_FRAME_BLUR_THRESHOLD = 80.0
FACE_CROP_BLUR_THRESHOLD = 50.0

def _is_truthy(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}



async def _vector_search_odoo(db, embedding: list[float]) -> Optional[tuple]:
    if not embedding:
        logger.warning("[pgvector] skipped because embedding is empty")
        return None

    distance_expr = FaceEmbedding.embedding.cosine_distance(embedding)
    similarity_expr = (1.0 - distance_expr).label("similarity")

    is_user_expr = (
        select(text("1"))
        .select_from(text("res_users"))
        .where(
            text("res_users.partner_id = res_partner.id"),
            text("res_users.active = TRUE"),
        )
        .exists()
        .label("is_user")
    )

    # Bá»Ž Ä‘iá»u kiá»‡n .where(similarity_expr >= settings.vector_match_threshold)
    # Ä‘á»ƒ láº¥y ra ngÆ°á»i giá»‘ng nháº¥t dÃ¹ Ä‘iá»ƒm tháº¥p
    stmt = (
        select(ResPartner, similarity_expr, is_user_expr)
        .select_from(FaceEmbedding)
        .join(ResPartner, FaceEmbedding.partner_id == ResPartner.id)
        .where(FaceEmbedding.active == True)
        .where(ResPartner.active == True)
        .where(FaceEmbedding.embedding != None)
        .order_by(similarity_expr.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.first()

    if not row:
        return None

    if row.similarity is None:
        logger.warning(
            "[pgvector] top candidate '%s' returned NULL similarity; skipping vector match",
            row.ResPartner.name,
        )
        return None

    logger.info(
        "[pgvector] Top match: '%s' similarity=%.4f threshold=%.2f",
        row.ResPartner.name,
        row.similarity,
        settings.vector_match_threshold,
    )

    if row.similarity >= settings.vector_match_threshold:
        return (row.ResPartner, float(row.similarity), row.is_user)

    return None



async def _resolve_partner_staff_state(db, partner_id: int) -> dict[str, Any]:
    """Return Odoo user/employee markers for a matched partner using raw SQL."""
    user_id = (
        await db.execute(
            text("SELECT id FROM res_users WHERE partner_id = :pid AND active = TRUE LIMIT 1"),
            {"pid": partner_id},
        )
    ).scalar()

    is_employee = False

    has_count_col = (
        await db.execute(
            text(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'res_partner'
                  AND column_name = 'employees_count'
                LIMIT 1
                """
            )
        )
    ).scalar() is not None

    if has_count_col:
        employees_count = (
            await db.execute(
                text("SELECT COALESCE(employees_count, 0) FROM res_partner WHERE id = :pid"),
                {"pid": partner_id},
            )
        ).scalar()
        is_employee = bool(employees_count and employees_count > 0)

    has_employee_col = (
        await db.execute(
            text(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'res_partner'
                  AND column_name = 'employee'
                LIMIT 1
                """
            )
        )
    ).scalar() is not None

    if has_employee_col:
        employee_flag = (
            await db.execute(
                text("SELECT COALESCE(employee, FALSE) FROM res_partner WHERE id = :pid"),
                {"pid": partner_id},
            )
        ).scalar()
        is_employee = is_employee or bool(employee_flag)

    employee_columns = (
        await db.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'hr_employee'
                """
            )
        )
    ).scalars().all()

    employee_columns = set(employee_columns)
    if employee_columns:
        conditions = []
        params = {"pid": partner_id, "uid": user_id}
        if "work_contact_id" in employee_columns:
            conditions.append("work_contact_id = :pid")
        if "address_home_id" in employee_columns:
            conditions.append("address_home_id = :pid")
        if user_id and "user_id" in employee_columns:
            conditions.append("user_id = :uid")

        if conditions:
            active_clause = "active IS NOT FALSE AND " if "active" in employee_columns else ""
            employee_id = (
                await db.execute(
                    text(
                        f"SELECT id FROM hr_employee WHERE {active_clause}({ ' OR '.join(conditions) }) LIMIT 1"
                    ),
                    params,
                )
            ).scalar()
            is_employee = is_employee or bool(employee_id)

    is_employee = is_employee or bool(user_id)
    return {"user_id": user_id, "is_employee": is_employee}

async def _handle_match(
    db,
    camera_id,
    camera_info,
    partner,
    similarity,
    confidence,
    snapshot_filename,
    snapshot_url,
    occurred_at,
    task_id,
    elapsed_ms,
):
    cam_int_id = camera_info.get("id")
    floor_id = camera_info.get("floor_id")
    zone_id = camera_info.get("zone_id")
    direction = camera_info.get("direction", "internal")

    log_entry = CameraEventLog.for_match(
        camera_id=camera_id,
        partner_id=partner.id,
        partner_name=partner.name,
        similarity=similarity,
        confidence=confidence,
        snapshot_filename=snapshot_filename,
        snapshot_url=snapshot_url,
        occurred_at=occurred_at,
        celery_task_id=task_id,
        processing_time_ms=elapsed_ms,
    )
    db.add(log_entry)

    if direction == "out":
        await db.execute(
            text(
                """UPDATE res_partner SET current_floor_id = NULL, current_zone_id = NULL, last_seen_time = :now, last_seen_camera_id = :cam_id WHERE id = :pid"""
            ),
            {"now": occurred_at, "cam_id": cam_int_id, "pid": partner.id},
        )
    else:
        await db.execute(
            text(
                """UPDATE res_partner SET current_floor_id = :floor_id, current_zone_id = :zone_id, last_seen_time = :now, last_seen_camera_id = :cam_id WHERE id = :pid"""
            ),
            {
                "floor_id": floor_id,
                "zone_id": zone_id,
                "now": occurred_at,
                "cam_id": cam_int_id,
                "pid": partner.id,
            },
        )


async def _handle_unknown(
    db,
    camera_id,
    snapshot_filename,
    snapshot_url,
    occurred_at,
    task_id,
    elapsed_ms,
    is_no_face=False,
):
    if is_no_face:
        log_entry = CameraEventLog.for_no_face(
            camera_id=camera_id,
            snapshot_filename=snapshot_filename,
            snapshot_url=snapshot_url,
            occurred_at=occurred_at,
            celery_task_id=task_id,
            processing_time_ms=elapsed_ms,
        )
    else:
        log_entry = CameraEventLog.for_unknown(
            camera_id=camera_id,
            snapshot_filename=snapshot_filename,
            snapshot_url=snapshot_url,
            occurred_at=occurred_at,
            celery_task_id=task_id,
            processing_time_ms=elapsed_ms,
        )
    db.add(log_entry)


async def _save_scan_history(db, camera_info, occurred_at, faces_data, full_image_b64):
    cam_int_id = camera_info.get("id")
    cam_name = camera_info.get("name", "")
    floor_id = camera_info.get("floor_id")
    floor_name = camera_info.get("floor_name", "")
    zone_id = camera_info.get("zone_id")
    zone_name = camera_info.get("zone_name", "")

    history = CameraScanHistory(
        camera_id=cam_int_id,
        floor_id=floor_id,
        zone_id=zone_id,
        scanned_at=occurred_at,
        face_count=len(faces_data),
        matched_count=sum(1 for f in faces_data if f.get("is_matched")),
        status="success" if faces_data else "no_face",
        full_image=full_image_b64.encode() if full_image_b64 else None,
    )
    db.add(history)
    await db.flush()

    result_objects = []
    for face in faces_data:
        result = CameraScanHistoryResult(
            history_id=history.id,
            camera_id=cam_int_id,
            floor_id=floor_id,
            zone_id=zone_id,
            scanned_at=occurred_at,
            partner_id=face.get("partner_id"),
            user_id=face.get("user_id"),
            is_employee=face.get("is_employee", False),
            confidence=face.get("confidence", 0.0),
            is_matched=face.get("is_matched", False),
            face_image=(
                face["face_base64"].encode() if face.get("face_base64") else None
            ),
            embedding=face.get("embedding"),
        )
        db.add(result)
        result_objects.append((result, face))

    await db.flush()

    scanned_at_str = occurred_at.strftime("%Y-%m-%d %H:%M:%S")
    scanned_at_iso = f"{occurred_at.isoformat()}Z"

    # Async Push Redis Socket
    for result, face in result_objects:
        await publish_event(
            {
                "type": "camera_recognition_event",
                "data": {
                    "id": result.id,
                    "camera_id": [cam_int_id, cam_name] if cam_int_id else False,
                    "partner_id": (
                        [face["partner_id"], face["partner_name"]]
                        if face.get("partner_id")
                        else False
                    ),
                    "confidence": face.get("confidence", 0.0),
                    "user_id": face.get("user_id"),
                    "is_employee": face.get("is_employee", False),
                    "is_matched": face.get("is_matched", False),
                    "face_image_url": (
                        f"data:image/jpeg;base64,{face['face_base64']}"
                        if face.get("face_base64")
                        else None
                    ),
                    "scanned_at": scanned_at_str,
                    "scanned_at_iso": scanned_at_iso,
                    "floor_id": [floor_id, floor_name] if floor_id else False,
                    "zone_id": [zone_id, zone_name] if zone_id else False,
                },
            }
        )


def _parse_iso(iso_str: str) -> datetime:
    """Parse an ISO timestamp string to a naive UTC datetime."""
    try:
        # 1. Chuyá»ƒn Ä‘á»•i chuá»—i ISO thÃ nh datetime cÃ³ mÃºi giá» UTC trÆ°á»›c
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        # 2. Ã‰p vá» naive datetime Ä‘á»ƒ khá»›p vá»›i kiá»ƒu dá»¯ liá»‡u cá»§a PostgreSQL
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        # CHá»– Cáº¦N Sá»¬A: Thay tháº¿ hoÃ n toÃ n cho datetime.utcnow() Ä‘á»ƒ xÃ³a cáº£nh bÃ¡o
        return datetime.now(timezone.utc).replace(tzinfo=None)


def _encode_frame_b64(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    scale = min(1280 / w, 720 / h)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


# ---------------------------------------------------------------------------
# NHÃ“M Táº¤T Cáº¢ DB IO VÃ€O ÄÃ‚Y (TRÃNH Láº¶P LAÌ£I OVERHEAD EVENT LOOP)
# ---------------------------------------------------------------------------
async def _execute_async_workflow(
    camera_id,
    snapshot_filename,
    snapshot_url,
    occurred_at_iso,
    task_id,
    start_time,
    faces_payload,
    full_image_b64,
    is_no_face,
    sys_config,
):
    dt_occurred = _parse_iso(occurred_at_iso)
    camera_info = await _resolve_camera_info(camera_id)

    async with get_db_session() as db:
        if is_no_face:
            await _handle_unknown(
                db,
                camera_id,
                snapshot_filename,
                snapshot_url,
                dt_occurred,
                task_id,
                (time.perf_counter() - start_time) * 1000,
                True,
            )
            await _save_scan_history(db, camera_info, dt_occurred, [], full_image_b64)
            await db.commit()
            return 0, 1

        matched_count = 0
        unknown_count = 0
        faces_for_history = []
        use_aws = _is_truthy(sys_config.get("camera.use_aws_recognition"), default=True)
        logger.info(
            "[%s] Match decision stage backend=%s faces=%d",
            camera_id,
            "aws_rekognition" if use_aws else "pgvector_insightface",
            len(faces_payload),
        )

        for payload in faces_payload:
            aws_api_result = payload["aws_api_result"]
            aws_error = payload["aws_error"]
            embedding = payload["embedding"]
            face_b64 = payload["face_b64"]
            match_result = None

            # Look up Match
            if use_aws and not aws_error:
                if aws_api_result:
                    ext_id = aws_api_result["ext_id"]
                    similarity = aws_api_result["similarity"]

                    logger.info("[%s] AWS matched ext_id=%s similarity=%.4f", camera_id, ext_id, similarity)

                    if ext_id.startswith("partner_"):
                        record_id = int(ext_id.split("_")[1])
                        stmt = select(ResPartner).where(ResPartner.id == record_id)
                        partner = (await db.execute(stmt)).scalar_one_or_none()
                        if partner:
                            stmt_usr = (
                                select(text("1"))
                                .where(text("partner_id = :pid AND active = TRUE"))
                                .select_from(text("res_users"))
                            )
                            is_user = (
                                await db.execute(stmt_usr, {"pid": record_id})
                            ).scalar() is not None
                            match_result = (partner, similarity, is_user)

                    elif ext_id.startswith("user_"):
                        record_id = int(ext_id.split("_")[1])
                        row = (
                            await db.execute(
                                text(
                                    "SELECT partner_id FROM res_users WHERE id = :uid AND active = TRUE"
                                ),
                                {"uid": record_id},
                            )
                        ).first()
                        if row:
                            partner = await db.get(ResPartner, row.partner_id)
                            if partner:
                                match_result = (partner, similarity, True)
                            else:
                                logger.warning(
                                    "[%s] AWS user_%s maps to missing partner_id=%s",
                                    camera_id,
                                    record_id,
                                    row.partner_id,
                                )
                else:
                    logger.warning(
                        "[%s] AWS returned no face match; keeping unknown because AWS is enabled",
                        camera_id,
                    )
            else:
                logger.info(
                    "[%s] Using pgvector/InsightFace fallback aws_error=%s use_aws=%s",
                    camera_id,
                    aws_error,
                    use_aws,
                )
                if embedding:
                    match_result = await _vector_search_odoo(db, embedding)
                else:
                    logger.warning(
                        "[%s] pgvector fallback skipped because payload has no embedding",
                        camera_id,
                    )

            # Process Results
            if match_result:
                partner, similarity, is_user = match_result
                staff_state = await _resolve_partner_staff_state(db, partner.id)
                user_id = staff_state.get("user_id")
                is_employee = bool(is_user or staff_state.get("is_employee"))

                if is_employee:
                    logger.info(
                        "[%s] Matched employee partner_id=%s user_id=%s; saved to history only",
                        camera_id,
                        partner.id,
                        user_id,
                    )
                    matched_count += 1
                    faces_for_history.append(
                        {
                            "partner_id": partner.id,
                            "partner_name": partner.name,
                            "user_id": user_id,
                            "is_employee": True,
                            "confidence": round(similarity * 100, 2),
                            "is_matched": True,
                            "face_base64": face_b64,
                            "embedding": json.dumps(embedding) if embedding else None,
                        }
                    )
                    continue

                if await _is_debounced(camera_id, str(partner.id), ttl=30):
                    continue

                await _handle_match(
                    db,
                    camera_id,
                    camera_info,
                    partner,
                    similarity,
                    similarity,
                    snapshot_filename,
                    snapshot_url,
                    dt_occurred,
                    task_id,
                    (time.perf_counter() - start_time) * 1000,
                )
                matched_count += 1
                faces_for_history.append(
                    {
                        "partner_id": partner.id,
                        "partner_name": partner.name,
                        "user_id": user_id,
                        "is_employee": False,
                        "confidence": round(similarity * 100, 2),
                        "is_matched": True,
                        "face_base64": face_b64,
                        "embedding": json.dumps(embedding) if embedding else None,
                    }
                )
            else:
                if embedding:
                    emb_hash = str(hash(tuple(round(v, 3) for v in embedding[:8])))
                else:
                    emb_hash = str(hash(face_b64[:128]))
                if await _is_debounced(camera_id, f"unk_{emb_hash}", ttl=15):
                    continue

                await _handle_unknown(
                    db,
                    camera_id,
                    snapshot_filename,
                    snapshot_url,
                    dt_occurred,
                    task_id,
                    (time.perf_counter() - start_time) * 1000,
                    False,
                )
                unknown_count += 1
                faces_for_history.append(
                    {
                        "partner_id": None,
                        "partner_name": None,
                        "confidence": 0.0,
                        "user_id": None,
                        "is_employee": False,
                        "is_matched": False,
                        "face_base64": face_b64,
                        "embedding": json.dumps(embedding) if embedding else None,
                    }
                )

        await _save_scan_history(
            db, camera_info, dt_occurred, faces_for_history, full_image_b64
        )
        await db.commit()
        return matched_count, unknown_count


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------


@shared_task(
    bind=True, name="app.tasks.process_event.process_camera_event", max_retries=3
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

    try:
        image = load_image_from_path(
            os.path.join(settings.raw_snapshot_dir, snapshot_filename)
        )
        if image is None or image.size == 0:
            return {"status": "invalid_image"}

        blurry, variance = _is_blurry(image, FULL_FRAME_BLUR_THRESHOLD)
        if blurry:
            return {"status": "skipped_blurry_frame"}

        full_image_b64 = _encode_frame_b64(image)

        sys_config = run_async(get_system_config())
        use_aws = _is_truthy(sys_config.get("camera.use_aws_recognition"), default=True)
        aws_summary = config_summary(sys_config)
        faces_payload = []

        if use_aws:
            aws_crops, aws_detect_error = detect_face_crops(image, sys_config)
            logger.info(
                "[%s] Recognition backend=aws_rekognition odoo_use_aws=%r aws_configured=%s collection=%s region=%s aws_detected_faces=%d detect_error=%s",
                camera_id,
                sys_config.get("camera.use_aws_recognition"),
                aws_summary["configured"],
                aws_summary["collection_id"],
                aws_summary["region"],
                len(aws_crops),
                aws_detect_error,
            )

            if not aws_detect_error:
                for face_crop in aws_crops:
                    _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    face_b64 = base64.b64encode(buf).decode()
                    aws_result, aws_error = search_face_by_image(face_crop, sys_config)
                    logger.info(
                        "[%s] AWS face #%d search result=%s error=%s",
                        camera_id,
                        len(faces_payload) + 1,
                        aws_result,
                        aws_error,
                    )
                    faces_payload.append(
                        {
                            "embedding": None,
                            "face_b64": face_b64,
                            "aws_api_result": aws_result,
                            "aws_error": aws_error,
                        }
                    )
            else:
                logger.warning(
                    "[%s] AWS DetectFaces failed; falling back to local InsightFace detection and pgvector matching",
                    camera_id,
                )
                use_aws = False

        if not use_aws:
            analyzer = FaceAnalyzer.get()
            faces = analyzer.extract(image)
            valid_faces = [
                f for f in faces if f.det_score >= settings.insightface_det_thresh
            ]
            logger.info(
                "[%s] Recognition backend=pgvector_insightface valid_faces=%d",
                camera_id,
                len(valid_faces),
            )

            for face in valid_faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                face_crop = image[max(0, y1) : y2, max(0, x1) : x2]
                if face_crop.size == 0:
                    continue

                crop_blurry, crop_var = _is_blurry(face_crop, FACE_CROP_BLUR_THRESHOLD)
                if crop_blurry:
                    logger.info(
                        "[%s] Skip local face crop: blurry variance=%.2f threshold=%.2f",
                        camera_id,
                        crop_var,
                        FACE_CROP_BLUR_THRESHOLD,
                    )
                    continue

                _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                face_b64 = base64.b64encode(buf).decode()
                faces_payload.append(
                    {
                        "embedding": face.to_list(),
                        "face_b64": face_b64,
                        "aws_api_result": None,
                        "aws_error": True,
                    }
                )

        is_no_face = len(faces_payload) == 0

        # Má»˜T Láº¦N CHáº Y ASYNC CHO Táº¤T Cáº¢ GIAO TIáº¾P DB/REDIS (TRÃNH OVERHEAD)
        matched, unknown = run_async(
            _execute_async_workflow(
                camera_id,
                snapshot_filename,
                snapshot_url,
                occurred_at_iso,
                task_id,
                start_time,
                faces_payload,
                full_image_b64,
                is_no_face,
                sys_config,
            )
        )

        logger.info(
            "[%s] Task done â€“ matched=%d unknown=%d elapsed=%.0f ms",
            camera_id,
            matched,
            unknown,
            (time.perf_counter() - start_time) * 1000,
        )
        return {"status": "processed", "matched": matched, "unknown": unknown}

    except Exception as exc:
        logger.error("[%s] Task failed: %s", camera_id, exc, exc_info=True)
        raise self.retry(exc=exc, countdown=5)



