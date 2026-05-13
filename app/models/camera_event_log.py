# backend/app/models/event_log.py
# =============================================================================
# SQLAlchemy model for the `camera_events_log` table.
#
# This table lives in the SAME Odoo PostgreSQL database but is owned by the
# AI Camera System (not by Odoo). It is created via the migration script
# (see scripts/create_event_log_table.sql) and written to by Celery workers
# after each face-recognition task completes.
#
# The AI system needs WRITE access to this table only:
#   GRANT SELECT, INSERT ON camera_events_log TO acs_readonly;
#   (rename the DB user to acs_writer if you prefer cleaner naming)
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

# ---------------------------------------------------------------------------
# Enum types (mirrored as PostgreSQL native ENUMs for efficient storage)
# ---------------------------------------------------------------------------


class EventType(str, PyEnum):
    FACE_MATCH = "face_match"
    UNKNOWN_FACE = "unknown_face"
    NO_FACE = "no_face"
    ERROR = "error"
    LINEDETECTION = "linedetection"
    FIELDDETECTION = "fielddetection"


class ProcessingStatus(str, PyEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PROCESSING = "processing"


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------


class CameraEventLog(Base):
    """
    Audit log written by the Celery worker after every camera event is processed.

    One row per Hikvision event. Includes the recognition result, confidence
    score, matched partner (nullable), and processing metadata.
    """

    __tablename__ = "camera_events_log"

    # ── Primary key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # ── Event identity ────────────────────────────────────────────────────────
    camera_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Hikvision camera identifier from the webhook payload",
    )
    event_type: Mapped[EventType] = mapped_column(
        PgEnum(
            EventType,
            name="camera_event_type",
            create_type=True,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        index=True,
    )

    processing_status: Mapped[ProcessingStatus] = mapped_column(
        PgEnum(
            ProcessingStatus,
            name="camera_processing_status",
            create_type=True,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=ProcessingStatus.SUCCESS,
    )

    # ── Recognition result ────────────────────────────────────────────────────
    matched_partner_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        # Soft FK – Odoo owns res_partner; we don't enforce a DB constraint
        # to avoid issues when a partner is archived/deleted in Odoo.
        nullable=True,
        index=True,
        comment="res_partner.id of the matched employee, NULL if no match",
    )
    matched_partner_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Denormalized partner name at the time of the event",
    )
    distance_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="pgvector L2 distance; lower = more similar (0.0 = identical)",
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Normalised confidence 0.0-1.0 derived from distance_score",
    )

    # ── Snapshot reference ────────────────────────────────────────────────────
    snapshot_filename: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Filename in the webhook_snapshots volume (may be purged after TTL)",
    )
    snapshot_url: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="Public URL served by FastAPI's static files handler",
    )

    # ── Celery task metadata ──────────────────────────────────────────────────
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Celery task UUID for distributed tracing",
    )
    processing_time_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Total wall-clock time for the Celery task in milliseconds",
    )
    error_detail: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Exception message / traceback if processing_status = failure",
    )

    # ── Timestamps ────────────────────────────────────────────────────────────
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When the camera sent the event (from webhook payload if available)",
    )
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When the Celery worker finished processing",
    )

    # ── Composite indexes for dashboard queries ───────────────────────────────
    __table_args__ = (
        Index("ix_cel_camera_occurred", "camera_id", "occurred_at"),
        Index("ix_cel_partner_occurred", "matched_partner_id", "occurred_at"),
        Index("ix_cel_event_type_occurred", "event_type", "occurred_at"),
    )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<CameraEventLog id={self.id} camera={self.camera_id!r} "
            f"type={self.event_type} partner={self.matched_partner_id}>"
        )

    @classmethod
    def for_match(
        cls,
        *,
        camera_id: str,
        partner_id: int,
        partner_name: str,
        distance: float,
        confidence: float,
        snapshot_filename: str | None,
        snapshot_url: str | None,
        occurred_at: datetime,
        celery_task_id: str | None = None,
        processing_time_ms: float | None = None,
    ) -> "CameraEventLog":
        """Factory: create a successful face-match log entry."""
        return cls(
            camera_id=camera_id,
            event_type=EventType.FACE_MATCH,
            processing_status=ProcessingStatus.SUCCESS,
            matched_partner_id=partner_id,
            matched_partner_name=partner_name,
            distance_score=distance,
            confidence=confidence,
            snapshot_filename=snapshot_filename,
            snapshot_url=snapshot_url,
            occurred_at=occurred_at,
            celery_task_id=celery_task_id,
            processing_time_ms=processing_time_ms,
        )

    @classmethod
    def for_no_face(cls, **kwargs) -> "CameraEventLog":
        """Factory cho trường hợp không thấy mặt nào."""
        kwargs.setdefault("event_type", EventType.NO_FACE)
        kwargs.setdefault("processing_status", ProcessingStatus.SUCCESS)
        return cls(**kwargs)

    @classmethod
    def for_unknown(
        cls,
        *,
        camera_id: str,
        snapshot_filename: str | None,
        snapshot_url: str | None,
        occurred_at: datetime,
        celery_task_id: str | None = None,
        processing_time_ms: float | None = None,
    ) -> "CameraEventLog":
        """Factory: create a log entry for an unrecognized face."""
        return cls(
            camera_id=camera_id,
            event_type=EventType.UNKNOWN_FACE,
            processing_status=ProcessingStatus.SUCCESS,
            snapshot_filename=snapshot_filename,
            snapshot_url=snapshot_url,
            occurred_at=occurred_at,
            celery_task_id=celery_task_id,
            processing_time_ms=processing_time_ms,
        )

    @classmethod
    def for_error(
        cls,
        *,
        camera_id: str,
        error_detail: str,
        occurred_at: datetime,
        celery_task_id: str | None = None,
    ) -> "CameraEventLog":
        """Factory: create a log entry for a failed processing attempt."""
        return cls(
            camera_id=camera_id,
            event_type=EventType.ERROR,
            processing_status=ProcessingStatus.FAILURE,
            error_detail=error_detail,
            occurred_at=occurred_at,
            celery_task_id=celery_task_id,
        )
