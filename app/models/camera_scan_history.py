# backend/app/models/camera_scan_history.py
from sqlalchemy import (
    Column,
    Integer,
    LargeBinary,
    String,
    DateTime,
    Float,
    Boolean,
    Text,
)
from app.core.database import Base


class CameraScanHistory(Base):
    __tablename__ = "camera_scan_history"
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, nullable=True)
    floor_id = Column(Integer, nullable=True)
    zone_id = Column(Integer, nullable=True)
    # INDEXED to prevent full table scan
    scanned_at = Column(DateTime, index=True)
    face_count = Column(Integer, default=0)
    matched_count = Column(Integer, default=0)
    status = Column(String)
    full_image = Column(LargeBinary)


class CameraScanHistoryResult(Base):
    __tablename__ = "camera_scan_history_result"
    id = Column(Integer, primary_key=True)
    history_id = Column(Integer)
    partner_id = Column(Integer, nullable=True)
    user_id = Column(Integer, nullable=True)
    is_employee = Column(Boolean, default=False, index=True)
    confidence = Column(Float)
    camera_id = Column(Integer, nullable=True)

    zone_id = Column(Integer, nullable=True, index=True)
    floor_id = Column(Integer, nullable=True, index=True)
    scanned_at = Column(DateTime, index=True)
    is_matched = Column(Boolean, default=False, index=True)

    face_image = Column(LargeBinary)
    embedding = Column(Text, nullable=True)

