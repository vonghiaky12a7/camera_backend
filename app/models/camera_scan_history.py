# backend/app/models/camera_scan_history.py
from sqlalchemy import Column, Integer, String, DateTime, Float
from app.core.database import Base


class CameraScanHistory(Base):
    __tablename__ = "camera_scan_history"
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, nullable=True)
    scanned_at = Column(DateTime)
    face_count = Column(Integer, default=0)
    matched_count = Column(Integer, default=0)
    status = Column(String)


class CameraScanHistoryResult(Base):
    __tablename__ = "camera_scan_history_result"
    id = Column(Integer, primary_key=True)
    history_id = Column(Integer)
    partner_id = Column(Integer, nullable=True)
    confidence = Column(Float)
