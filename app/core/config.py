# backend/app/core/config.py
# =============================================================================
# Centralised settings – loaded once at startup from environment / .env file.
# All other modules import `settings` from here; never read os.environ directly.
# =============================================================================

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── Service identity ──────────────────────────────────────────────────────
    service_role: Literal["api", "worker", "beat"] = "api"
    log_level: str = "info"
    log_json: bool = False
    secret_key: str = Field(default="change-me", min_length=16)

    # ── Odoo PostgreSQL ───────────────────────────────────────────────────────
    odoo_database_url: PostgresDsn = Field(
        ...,
        description=("Async DSN: postgresql+asyncpg://user:pass@host:port/dbname"),
    )
    odoo_database_url_sync: PostgresDsn | None = None  # Alembic / scripts

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # ── Odoo HTTP / Webhooks ──────────────────────────────────────────────────
    odoo_base_url: str = "http://odoo:8069"
    odoo_webhook_db: str = "odoo"
    odoo_webhook_api_key: str = ""

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://redis:6379/0")  # broker
    redis_pubsub_url: RedisDsn = Field(default="redis://redis:6379/1")  # events
    redis_event_channel: str = "acs:events"

    # ── Celery ────────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"
    celery_concurrency: int = 4
    celery_task_soft_time_limit: int = 25
    celery_task_time_limit: int = 30

    # ── InsightFace / ONNX ────────────────────────────────────────────────────
    insightface_model_name: str = "buffalo_l"
    insightface_model_dir: str = "/app/models_cache"
    insightface_det_thresh: float = 0.5
    insightface_rec_thresh: float = 0.45
    ort_intra_op_num_threads: int = 2
    ort_inter_op_num_threads: int = 1

    # ── Vector search ─────────────────────────────────────────────────────────
    vector_distance_op: Literal["l2", "inner_product", "cosine"] = "cosine"
    vector_match_threshold: float = 0.6
    vector_top_k: int = 3

    # ── Snapshot storage ──────────────────────────────────────────────────────
    raw_snapshot_dir: str = "/tmp/snapshots"
    processed_snapshot_dir: str = "/app/snapshots"
    snapshot_ttl_seconds: int = 300

    # ── Camera / Hikvision ────────────────────────────────────────────────────
    allowed_camera_ips: str = ""
    hik_webhook_secret: str = ""

    # ── FastAPI ───────────────────────────────────────────────────────────────
    fastapi_workers: int = 2

    @property
    def allowed_camera_ips_list(self) -> list[str]:

        if not self.allowed_camera_ips:
            return []

        return [ip.strip() for ip in self.allowed_camera_ips.split(",") if ip.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Convenience alias used throughout the codebase:
#   from app.core.config import settings
settings: Settings = get_settings()
