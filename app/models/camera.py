# backend/app/models/odoo_camera_models.py
# =============================================================================
# SQLAlchemy ORM models mapped to Odoo's camera management tables.
#
# IMPORTANT:
#   - These models are used by the AI backend to READ camera topology
#     (building → floor → zone → camera) and WRITE aggregated stats
#     (camera_count, camera_online_count, customer_count, etc.) directly
#     into the Odoo PostgreSQL database.
#   - Odoo owns the schema. We only map the columns we need.
#   - All writes go through explicit UPDATE statements to avoid accidentally
#     touching Odoo-managed columns (write_date, __last_update, etc.).
# =============================================================================

from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


# ---------------------------------------------------------------------------
# camera_camera  (Odoo model: camera.camera)
# ---------------------------------------------------------------------------
class CameraCamera(Base):
    """
    Partial mapping of Odoo's ``camera_camera`` table.

    Used for:
      - Counting total cameras per floor
      - Counting online cameras per floor
      - Mapping camera MAC/code → floor via zone
    """

    __tablename__ = "camera_camera"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    # Camera identifier (MAC-derived or configured code)
    code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)

    # Status: 'online' | 'offline' | 'maintenance' | 'unknown'
    status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)

    # FK to camera_zone (via zone_id Many2one in Odoo)
    zone_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)

    # Denormalized FKs stored by Odoo (related fields with store=True)
    floor_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    building_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<CameraCamera id={self.id} code={self.code!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# camera_zone  (Odoo model: camera.zone)
# ---------------------------------------------------------------------------
class CameraZone(Base):
    """
    Partial mapping of Odoo's ``camera_zone`` table.

    Used to resolve: camera → zone → floor.
    """

    __tablename__ = "camera_zone"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    # FK to camera_floor
    floor_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)

    # Denormalized from floor (store=True in Odoo)
    building_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<CameraZone id={self.id} name={self.name!r} floor_id={self.floor_id}>"


# ---------------------------------------------------------------------------
# camera_floor  (Odoo model: camera.floor)
# ---------------------------------------------------------------------------
class CameraFloor(Base):
    """
    Partial mapping of Odoo's ``camera_floor`` table.

    The AI backend WRITES the following counter columns directly:
        camera_count            – total cameras on this floor
        camera_online_count     – cameras with status='online'
        camera_offline_count    – cameras with status='offline'
        customer_count          – faces matched (recognized partners) today
        waiting_customer_count  – unknown faces detected today (unregistered)

    All other columns are READ-ONLY from the AI system's perspective.
    """

    __tablename__ = "camera_floor"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    floor_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    building_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # ── Writable counter columns ───────────────────────────────────────────────
    # The AI backend is the SOLE WRITER for these fields.
    # Odoo defines them as plain Integer fields (no compute), default=0,
    # so direct SQL UPDATE is safe and will not break Odoo's ORM cache
    # (Odoo will re-read from DB on next request).

    zone_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    camera_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    camera_online_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    camera_offline_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    customer_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    waiting_customer_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    def __repr__(self) -> str:
        return (
            f"<CameraFloor id={self.id} name={self.name!r} "
            f"cameras={self.camera_count} online={self.camera_online_count} "
            f"customers={self.customer_count} waiting={self.waiting_customer_count}>"
        )
