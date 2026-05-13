# backend/app/models/partner.py
# =============================================================================
# SQLAlchemy ORM model mapped to Odoo's `res_partner` table.
#
# IMPORTANT CONSTRAINTS:
#   - This model is READ-ONLY from the AI system's perspective.
#     The Odoo application owns writes to this table.
#   - Only the columns needed for face recognition are mapped.
#     Unmapped Odoo columns are silently ignored by SQLAlchemy.
#   - `face_embedding` uses pgvector's `Vector` type (512 dimensions),
#     matching the `vector(512)` column added to Odoo's res_partner table.
#
# pgvector operator reference:
#   <->   L2 (Euclidean) distance       – use for cosine-normalised embeddings
#   <#>   negative inner product        – use when embeddings are L2-normalised
#   <=>   cosine distance               – 1 - cosine_similarity
#
# InsightFace buffalo_s produces L2-normalised embeddings, so both <-> and <#>
# give equivalent ranking. We default to <-> (L2) as it is more intuitive.
# =============================================================================

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ResPartner(Base):
    """
    Partial mapping of Odoo's ``res_partner`` table.

    Only columns required by the face-recognition pipeline are declared.
    SQLAlchemy will not touch unmapped Odoo columns.
    """

    __tablename__ = "res_partner"

    # ── Primary key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # ── Identity fields ───────────────────────────────────────────────────────
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(254), nullable=True)
    ref: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Odoo internal reference / employee ID",
    )

    # ── Status flags ──────────────────────────────────────────────────────────
    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Odoo soft-delete flag; False = archived partner",
    )
    is_company: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ── Face embedding (custom field added to Odoo) ───────────────────────────
    face_embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(512),
        nullable=True,
        comment="512-dim face embedding produced by InsightFace buffalo_s",
    )

    # ── Audit timestamps (Odoo standard) ──────────────────────────────────────
    create_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    write_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )

    # ── Relationship helpers (optional, read-only context) ────────────────────
    # Odoo stores the company of an employee as a many2one to res_partner itself.
    # We don't need the full join for face recognition, so we only map the FK.
    company_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:  # noqa: D105
        return f"<ResPartner id={self.id} name={self.name!r} active={self.active}>"

    @property
    def has_embedding(self) -> bool:
        """True if this partner has a stored face embedding."""
        return self.face_embedding is not None

    @property
    def display(self) -> str:
        """Best available display string for logs and API responses."""
        return self.name or f"Partner #{self.id}"
