# backend/app/api/cameras.py
import logging
import os
import uuid
import re
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Body,
    Path,
)
from starlette.status import HTTP_200_OK, HTTP_401_UNAUTHORIZED

from app.core.config import settings
from app.core.redis import EventSubscriber
from app.core.database import get_db_session
from app.tasks.process_event import process_camera_event
from app.services.stats_service import update_dashboard_stats
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# DASHBOARD REST ENDPOINTS  (dùng cho Next.js frontend)
# =============================================================================


@router.get("/api/v1/dashboard/stats")
async def get_dashboard_stats():
    """
    Trả về 4 chỉ số tổng hợp toàn hệ thống:
      - total_cameras
      - online_cameras
      - total_customers  (face_match hôm nay)
      - total_waiting    (unknown_face hôm nay)

    Next.js gọi endpoint này mỗi lần load trang.
    Dữ liệu đã được backend AI cập nhật thẳng vào camera_floor,
    nên query ở đây chỉ là SUM đơn giản, rất nhanh.
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
    """
    Trả về danh sách tất cả floors kèm cameras.
    Next.js dùng để render FloorTabs + CameraGrid.
    """
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

    # Group cameras by floor_id
    cameras_by_floor: dict[int, list] = {}
    for cam in camera_rows:
        fid = cam["floor_id"]
        if fid not in cameras_by_floor:
            cameras_by_floor[fid] = []
        cameras_by_floor[fid].append(
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
    Trả về danh sách nhận diện khuôn mặt gần nhất.
    Dùng cho sidebar RecognitionLogsSidebar và bảng RecognitionTable.
    """
    sql = text("""
        SELECT
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
        LEFT JOIN res_partner p  ON p.id  = r.partner_id
        LEFT JOIN camera_camera c ON c.id = r.camera_id
        LEFT JOIN camera_zone   z ON z.id = c.zone_id
        ORDER BY r.scanned_at DESC
        LIMIT :limit
    """)

    async with get_db_session() as db:
        rows = (await db.execute(sql, {"limit": limit})).mappings().all()

    results = []
    for r in rows:
        scanned_at_str = str(r["scanned_at"]) if r["scanned_at"] else ""
        # Format timestamp thành "X min ago"
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

        # Status logic: matched=verified, confidence>0=pending, else=flagged
        confidence_pct = float(r["confidence"] or 0)
        is_matched = bool(r["is_matched"])
        if is_matched:
            status = "verified"
        elif confidence_pct > 0:
            status = "pending"
        else:
            status = "flagged"

        location_parts = [r["camera_name"], r["zone_name"]]
        location = " — ".join(p for p in location_parts if p) or "Unknown location"

        results.append(
            {
                # Dùng chung cho cả sidebar log lẫn bảng history
                "id": str(r["id"]),
                "customerId": str(r["partner_id"]) if r["partner_id"] else "",
                "customerName": r["partner_name"] or "Unknown visitor",
                "phoneNumber": r["partner_phone"] or "—",
                "gender": "—",
                "timestamp": timestamp_rel,
                "scannedAt": scanned_at_str,
                # Bảng history
                "confidence": round(confidence_pct / 100, 4),  # 0-1
                "status": status,
                "location": location,
                "isMatched": is_matched,
                "cameraName": r["camera_name"] or "Unknown camera",
                # URL ảnh khuôn mặt (qua Odoo image route)
                "faceImageUrl": f"{settings.odoo_base_url}/web/image?model=camera.scan.history.result&id={r['id']}&field=face_image",
            }
        )

    return results


@router.get("/api/v1/dashboard/analytics")
async def get_analytics():
    """
    Thống kê số lượt scan theo khung giờ trong ngày hôm nay.
    Dùng cho AnalyticsChart (visits = tổng scan, conversions = matched).
    """
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

    # Tạo 8 slot x 3h (00:00, 03:00, ..., 21:00)
    slots = [
        {"time": f"{str(i * 3).zfill(2)}:00", "visits": 0, "conversions": 0}
        for i in range(8)
    ]

    for row in rows:
        if not row["hour_utc"]:
            continue
        hour = row["hour_utc"].hour
        slot_idx = hour // 3
        if 0 <= slot_idx < 8:
            slots[slot_idx]["visits"] += int(row["total"] or 0)
            slots[slot_idx]["conversions"] += int(row["matched"] or 0)

    return slots


# =============================================================================
# WEBHOOK — nhận event từ camera Hikvision
# =============================================================================


@router.post("/api/v1/cameras/events/{secret_key}", status_code=HTTP_200_OK)
async def receive_camera_event(
    request: Request,
    secret_key: str = Path(..., description="Chuỗi key xác thực Webhook"),
    payload: bytes = Body(..., media_type="application/octet-stream"),
):
    """Nhận webhook ISAPI từ camera Hikvision."""
    if secret_key != settings.hik_webhook_secret:
        logger.warning(f"CHẶN TRUY CẬP: Secret key không hợp lệ ({secret_key})")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, detail="Invalid secret key"
        )

    raw_ip = request.client.host if request.client else "unknown"
    logger.info(f"Nhận webhook hợp lệ từ IP: {raw_ip}")

    try:
        body = await request.body()

        xml_start = body.find(b"<EventNotificationAlert")
        xml_end = body.find(b"</EventNotificationAlert>")
        if xml_start == -1 or xml_end == -1:
            return {"status": "ignored", "message": "No XML found"}

        xml_data = body[xml_start : xml_end + len(b"</EventNotificationAlert>")].decode(
            "utf-8", errors="ignore"
        )

        mac_match = re.search(r"<macAddress>(.*?)</macAddress>", xml_data)
        ip_match = re.search(r"<ipAddress>(.*?)</ipAddress>", xml_data)
        event_match = re.search(r"<eventType>(.*?)</eventType>", xml_data)
        time_match = re.search(r"<dateTime>(.*?)</dateTime>", xml_data)

        camera_id = (
            mac_match.group(1).replace(":", "").upper()
            if mac_match
            else (ip_match.group(1) if ip_match else raw_ip)
        )
        event_type = event_match.group(1).lower() if event_match else "unknown"
        occurred_at_iso = (
            time_match.group(1)
            if time_match
            else datetime.now(timezone.utc).isoformat()
        )

        if event_type in ["duration", "videoloss"]:
            return {"status": "ok", "message": f"Ignored {event_type} event"}

        jpg_start = body.find(b"\xff\xd8")
        jpg_end = body.rfind(b"\xff\xd9")
        if jpg_start == -1 or jpg_end == -1:
            return {"status": "ok", "message": "Event without image"}

        image_data = body[jpg_start : jpg_end + 2]
        image_size_kb = len(image_data) / 1024

        filename = f"{camera_id}_{event_type}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(settings.raw_snapshot_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)

        logger.info(
            f"📥 [WEBHOOK] Camera: {camera_id} | Event: {event_type} | "
            f"Payload: {image_size_kb:.1f} KB | Time: {occurred_at_iso}"
        )

        process_camera_event.delay(
            camera_id=camera_id,
            snapshot_filename=filename,
            occurred_at_iso=occurred_at_iso,
            event_type=event_type,
        )

        logger.info(f"[{camera_id}] Received '{event_type}', saved {filename}, queued.")
        return {"status": "ok", "message": "Event queued for processing"}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")


# =============================================================================
# WEBSOCKET — push real-time events xuống Next.js
# =============================================================================


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint — Next.js kết nối vào đây để nhận real-time events.
    Mỗi khi có face_match / unknown_face, backend publish lên Redis,
    EventSubscriber relay xuống WebSocket này, Next.js cập nhật UI ngay.
    """
    await websocket.accept()
    logger.info("New WebSocket connection accepted.")
    try:
        async with EventSubscriber() as subscriber:
            async for event in subscriber:
                await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close()
