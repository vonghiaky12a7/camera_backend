# backend/app/models/face_embedding.py
from __future__ import annotations
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class FaceEmbedding(Base):
    """
    Model lưu trữ vector khuôn mặt, liên kết trực tiếp với res_partner.
    Tên bảng được đổi thành 'face_embedding'.
    """

    __tablename__ = "face_embedding"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Liên kết với bảng res_partner (Khách hàng hoặc Nhân viên)
    partner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("res_partner.id"), nullable=False
    )

    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
