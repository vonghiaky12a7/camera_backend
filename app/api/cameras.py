# backend/app/api/cameras.py
# =============================================================================
# Changes in this version:
#   - Tối ưu Async File I/O bằng aiofiles chống khóa Event Loop
#   - get_recognition_logs: DISTINCT ON partner to show only latest image per user
# =============================================================================

import logging
import os
import uuid
import re
import aiofiles
from datetime import datetime, timezone
import asyncio
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Body,
    Path,
)
from starlette.status import HTTP_200_OK, HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from app.core.config import settings
from app.core.redis import EventSubscriber
from app.core.database import get_db_session
from app.tasks.process_event import process_camera_event
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# DASHBOARD REST ENDPOINTS
# =============================================================================


@router.get("/api/v1/dashboard/stats")
async def get_dashboard_stats():
    """
    4 system-wide counters:
      - total_cameras / online_cameras
      - total_customers (matched faces today, per res_partner tracking fields)
      - total_waiting   (unmatched faces in last 30 min rolling window)
    """
    sql = text("""
        SELECT
            COALESCE(SUM(camera_count), 0)           AS total_cameras,
            COALESCE(SUM(camera_online_count), 0)    AS online_cameras,
            COALESCE(SUM(customer_count), 0)         AS total_customers,
            COALESCE(SUM(waiting_customer_count), 0) AS total_waiting
        FROM camera_floor
        WHERE active = TRUE
    """)
    async with get_db_session() as db:
        row = (await db.execute(sql)).mappings().one()
    return {
        "total_cameras": int(row["total_cameras"]),
        "online_cameras": int(row["online_cameras"]),
        "total_customers": int(row["total_customers"]),
        "total_waiting": int(row["total_waiting"]),
    }


@router.get("/api/v1/dashboard/floors")
async def get_floors():
    """All floors with their cameras for FloorTabs + CameraGrid."""
    floor_sql = text("""
        SELECT
            f.id,
            f.name,
            f.floor_number,
            f.camera_count,
            f.camera_online_count,
            f.customer_count,
            f.waiting_customer_count
        FROM camera_floor f
        WHERE f.active = TRUE
        ORDER BY f.sequence ASC, f.floor_number ASC
    """)

    camera_sql = text("""
        SELECT
            c.id,
            c.name,
            c.code,
            c.status,
            c.floor_id,
            c.go2rtc_url,
            c.go2rtc_sub_url
        FROM camera_camera c
        WHERE c.active = TRUE
          AND c.floor_id IS NOT NULL
        ORDER BY c.name ASC
    """)

    async with get_db_session() as db:
        floor_rows = (await db.execute(floor_sql)).mappings().all()
        camera_rows = (await db.execute(camera_sql)).mappings().all()

    cameras_by_floor: dict[int, list] = {}
    for cam in camera_rows:
        fid = cam["floor_id"]
        cameras_by_floor.setdefault(fid, []).append(
            {
                "id": str(cam["id"]),
                "name": cam["name"],
                "code": cam["code"],
                "status": "active" if cam["status"] == "online" else "inactive",
                "customers": 0,
                "go2rtcUrl": cam["go2rtc_url"] or None,
                "go2rtcSubUrl": cam["go2rtc_sub_url"] or None,
            }
        )

    floors = []
    for f in floor_rows:
        cameras = cameras_by_floor.get(f["id"], [])
        floors.append(
            {
                "id": str(f["id"]),
                "name": f["name"],
                "floorNumber": f["floor_number"],
                "totalCameras": int(f["camera_count"]),
                "activeCameras": int(f["camera_online_count"]),
                "customers": int(f["customer_count"]),
                "waiting": int(f["waiting_customer_count"]),
                "cameras": cameras,
            }
        )

    return floors


@router.get("/api/v1/dashboard/recognition-logs")
async def get_recognition_logs(limit: int = 20):
    """
    Latest recognition events for the sidebar and history table.
    """
    sql = text("""
        SELECT
            id,
            confidence,
            is_matched,
            scanned_at,
            partner_id,
            partner_name,
            partner_phone,
            camera_id,
            camera_name,
            zone_name
        FROM (
            SELECT DISTINCT ON (
                COALESCE(r.partner_id::text, r.id::text)
            )
                r.id,
                r.confidence,
                r.is_matched,
                r.scanned_at,
                p.id        AS partner_id,
                p.name      AS partner_name,
                p.phone     AS partner_phone,
                c.id        AS camera_id,
                c.name      AS camera_name,
                z.name      AS zone_name
            FROM camera_scan_history_result r
            LEFT JOIN res_partner   p ON p.id  = r.partner_id
            LEFT JOIN camera_camera c ON c.id  = r.camera_id
            LEFT JOIN camera_zone   z ON z.id  = c.zone_id
            WHERE r.is_hidden = FALSE
            ORDER BY
                COALESCE(r.partner_id::text, r.id::text),  
                r.scanned_at DESC                           
        ) sub
        ORDER BY scanned_at DESC
        LIMIT :limit
    """)

    async with get_db_session() as db:
        rows = (await db.execute(sql, {"limit": limit})).mappings().all()

    results = []
    for r in rows:
        scanned_at_str = str(r["scanned_at"]) if r["scanned_at"] else ""

        try:
            utc_dt = datetime.fromisoformat(
                scanned_at_str.replace(" ", "T").rstrip("Z")
            ).replace(tzinfo=timezone.utc)
            diff_min = int((datetime.now(timezone.utc) - utc_dt).total_seconds() / 60)
            if diff_min < 1:
                timestamp_rel = "Just now"
            elif diff_min < 60:
                timestamp_rel = f"{diff_min} min ago"
            else:
                timestamp_rel = f"{diff_min // 60}h ago"
        except Exception:
            timestamp_rel = scanned_at_str

        confidence_pct = float(r["confidence"] or 0)
        is_matched = bool(r["is_matched"])
        if is_matched:
            status = "verified"
        elif confidence_pct > 0:
            status = "pending"
        else:
            status = "flagged"

        location = (
            " — ".join(p for p in [r["camera_name"], r["zone_name"]] if p)
            or "Unknown location"
        )

        results.append(
            {
                "id": str(r["id"]),
                "customerId": str(r["partner_id"]) if r["partner_id"] else "",
                "customerName": r["partner_name"] or "Unknown visitor",
                "phoneNumber": r["partner_phone"] or "—",
                "gender": "—",
                "timestamp": timestamp_rel,
                "scannedAt": scanned_at_str,
                "confidence": round(confidence_pct / 100, 4),
                "status": status,
                "location": location,
                "isMatched": is_matched,
                "cameraName": r["camera_name"] or "Unknown camera",
                "faceImageUrl": (
                    f"{settings.odoo_base_url}/web/image"
                    f"?model=camera.scan.history.result&id={r['id']}&field=face_image"
                ),
            }
        )

    return results


@router.get("/api/v1/dashboard/analytics")
async def get_analytics():
    """Hourly scan counts for today (grouped into 3-hour slots)."""
    today_sql = text("""
        SELECT
            DATE_TRUNC('hour', scanned_at AT TIME ZONE 'UTC') AS hour_utc,
            COUNT(*)                                           AS total,
            COUNT(*) FILTER (WHERE status = 'success')        AS matched
        FROM camera_scan_history
        WHERE scanned_at >= CURRENT_DATE
        GROUP BY 1
        ORDER BY 1
    """)

    async with get_db_session() as db:
        rows = (await db.execute(today_sql)).mappings().all()

    slots = [
        {"time": f"{str(i * 3).zfill(2)}:00", "visits": 0, "conversions": 0}
        for i in range(8)
    ]
    for row in rows:
        if not row["hour_utc"]:
            continue
        slot_idx = row["hour_utc"].hour // 3
        if 0 <= slot_idx < 8:
            slots[slot_idx]["visits"] += int(row["total"] or 0)
            slots[slot_idx]["conversions"] += int(row["matched"] or 0)

    return slots


# =============================================================================
# WEBHOOK – receive events from Hikvision cameras
# =============================================================================


@router.post("/api/v1/cameras/events/{secret_key}", status_code=HTTP_200_OK)
async def receive_camera_event(
    request: Request,
    secret_key: str = Path(..., description="Webhook auth key"),
    payload: bytes = Body(..., media_type="application/octet-stream"),
):
    """Receive ISAPI webhook payload from a Hikvision camera."""
    raw_ip = request.client.host if request.client else "unknown"

    if secret_key != settings.hik_webhook_secret:
        logger.warning("Blocked invalid secret key: %s", secret_key)
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, detail="Invalid secret key"
        )

    allowed_ips = settings.allowed_camera_ips_list
    if allowed_ips and raw_ip not in allowed_ips:
        logger.warning("Blocked camera webhook from unauthorized IP: %s", raw_ip)
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Camera IP is not allowed"
        )

    try:
        body = await request.body()

        xml_start = body.find(b"<EventNotificationAlert")
        xml_end = body.find(b"</EventNotificationAlert>")
        if xml_start == -1 or xml_end == -1:
            return {"status": "ignored", "message": "No XML found"}

        xml_data = body[xml_start : xml_end + len(b"</EventNotificationAlert>")].decode(
            "utf-8", errors="ignore"
        )

        def _re(pattern: str) -> str:
            m = re.search(pattern, xml_data)
            return m.group(1) if m else ""

        camera_id = (
            _re(r"<macAddress>(.*?)</macAddress>")
            or _re(r"<ipAddress>(.*?)</ipAddress>")
            or raw_ip
        )
        event_type = _re(r"<eventType>(.*?)</eventType>").lower() or "unknown"
        occurred_at = (
            _re(r"<dateTime>(.*?)</dateTime>") or datetime.now(timezone.utc).isoformat()
        )

        if event_type in ("duration", "videoloss"):
            return {"status": "ok", "message": f"Ignored {event_type}"}

        jpg_start = body.find(b"\xff\xd8")
        jpg_end = body.rfind(b"\xff\xd9")
        if jpg_start == -1 or jpg_end == -1:
            return {"status": "ok", "message": "Event without image"}

        image_data = body[jpg_start : jpg_end + 2]
        image_size_kb = len(image_data) / 1024
        filename = f"{camera_id}_{event_type}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(settings.raw_snapshot_dir, filename)

        # TOI UU: Lưu file không chặn event loop
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(image_data)

        logger.info(
            "📥 [WEBHOOK] camera=%s event=%s size=%.1f KB time=%s",
            camera_id,
            event_type,
            image_size_kb,
            occurred_at,
        )

        process_camera_event.delay(
            camera_id=camera_id,
            snapshot_filename=filename,
            occurred_at_iso=occurred_at,
            event_type=event_type,
        )

        return {"status": "ok", "message": "Event queued"}

    except Exception as exc:
        logger.error("Webhook processing error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")


# =============================================================================
# WEBSOCKET – push real-time events to Next.js
# =============================================================================
@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected.")

    async with EventSubscriber() as subscriber:
        # Task 1: Nghe tin nhắn từ Redis và gửi xuống Client
        async def send_events():
            async for event in subscriber:
                await websocket.send_json(event)

        # Task 2: Chờ tín hiệu client ngắt kết nối (ĐÃ SỬA LỖI RUNTIMERROR)
        async def listen_disconnect():
            try:
                while True:
                    # Đọc message thô để kiểm tra loại message
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        logger.debug("Received disconnect message type from client.")
                        break
            except WebSocketDisconnect:
                logger.debug("WebSocketDisconnect exception caught inside listener.")
            except RuntimeError as e:
                # Bắt lỗi "Cannot call receive once a disconnect message..." nếu có
                if "disconnect" in str(e):
                    logger.debug(
                        "RuntimeError caught safely: Client already disconnected."
                    )
                else:
                    raise e

        # Chạy song song cả 2 tác vụ
        task_send = asyncio.create_task(send_events())
        task_listen = asyncio.create_task(listen_disconnect())

        done, pending = await asyncio.wait(
            [task_send, task_listen], return_when=asyncio.FIRST_COMPLETED
        )

        # Hủy tác vụ còn lại để giải phóng tài nguyên hệ thống
        for task in pending:
            task.cancel()

    logger.info("WebSocket client disconnected & cleaned up.")
