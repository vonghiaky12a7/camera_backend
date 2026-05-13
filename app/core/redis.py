# backend/app/core/redis.py
# =============================================================================
# Redis client factory and Pub/Sub helpers.
#
# Two logical Redis databases are used:
#   DB 0  (redis_url)        – Celery broker + result backend
#   DB 1  (redis_pubsub_url) – Real-time event Pub/Sub channel
#
# This module exposes:
#   - get_redis()            – async context manager, general-purpose commands
#   - publish_event()        – publish a structured event dict to the channel
#   - EventSubscriber        – async iterator that yields events from the channel
#                              (used by the FastAPI SSE/WebSocket endpoint)
# =============================================================================

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pools (module-level singletons)
# One pool per Redis DB index so connections are not wasted.
# ---------------------------------------------------------------------------

_broker_pool: aioredis.ConnectionPool = aioredis.ConnectionPool.from_url(
    str(settings.redis_url),
    encoding="utf-8",
    decode_responses=True,
    max_connections=20,
)

_pubsub_pool: aioredis.ConnectionPool = aioredis.ConnectionPool.from_url(
    str(settings.redis_pubsub_url),
    encoding="utf-8",
    decode_responses=True,
    max_connections=50,  # Higher: one connection per SSE/WS client
)


def get_broker_client() -> aioredis.Redis:
    """Return a Redis client backed by the broker pool (DB 0)."""
    return aioredis.Redis(connection_pool=_broker_pool)


def get_pubsub_client() -> aioredis.Redis:
    """Return a Redis client backed by the pub/sub pool (DB 1)."""
    return aioredis.Redis(connection_pool=_pubsub_pool)


# ---------------------------------------------------------------------------
# General-purpose async context manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def get_redis(pubsub: bool = False) -> AsyncGenerator[aioredis.Redis, None]:
    """
    Async context manager that yields a Redis client.

    Args:
        pubsub: If True, uses the Pub/Sub pool (DB 1); else the broker pool (DB 0).

    Example::

        async with get_redis() as r:
            await r.set("key", "value", ex=60)

        async with get_redis(pubsub=True) as r:
            await r.publish(settings.redis_event_channel, payload)
    """
    client = get_pubsub_client() if pubsub else get_broker_client()
    try:
        yield client
    finally:
        # Pools manage connections; we don't close the client here.
        pass


# ---------------------------------------------------------------------------
# Lifespan helpers
# ---------------------------------------------------------------------------
async def connect_redis() -> None:
    """Ping both Redis DBs at startup to fail fast if unreachable."""
    for label, client in [
        ("broker (DB 0)", get_broker_client()),
        ("pubsub (DB 1)", get_pubsub_client()),
    ]:
        pong = await client.ping()
        if not pong:
            raise ConnectionError(f"Redis {label} did not respond to PING.")
        logger.info("Redis %s connection OK.", label)


async def disconnect_redis() -> None:
    """Disconnect all pooled Redis connections on shutdown."""
    await _broker_pool.aclose()
    await _pubsub_pool.aclose()
    logger.info("Redis connection pools closed.")


# ---------------------------------------------------------------------------
# Publishing helpers (called by Celery workers after processing)
# ---------------------------------------------------------------------------


async def publish_event(event: dict[str, Any]) -> int:
    """
    Serialize ``event`` to JSON and publish it to the configured Pub/Sub channel.

    Returns the number of subscribers that received the message.

    Args:
        event: Arbitrary dict. Must be JSON-serialisable. Recommended keys::

            {
                "camera_id":    "cam_01",
                "partner_id":   42,
                "partner_name": "Nguyen Van A",
                "confidence":   0.87,
                "image_url":    "/snapshots/abc123.jpg",
                "event_type":   "face_match" | "unknown_face",
                "timestamp":    "2024-01-15T08:30:00Z",
            }

    Example (in a Celery task)::

        from app.core.redis import publish_event
        receivers = await publish_event({"camera_id": "cam_01", ...})
    """
    payload = json.dumps(event, default=str)
    async with get_redis(pubsub=True) as r:
        receivers: int = await r.publish(settings.redis_event_channel, payload)

    logger.debug(
        "Event published to '%s' (%d subscriber(s)): camera=%s type=%s",
        settings.redis_event_channel,
        receivers,
        event.get("camera_id"),
        event.get("event_type"),
    )
    return receivers


# ---------------------------------------------------------------------------
# Subscriber – async iterator for the FastAPI SSE / WebSocket endpoint
# ---------------------------------------------------------------------------


class EventSubscriber:
    """
    Async iterator that subscribes to the Redis Pub/Sub event channel and
    yields deserialized event dicts as they arrive.

    Designed to be used inside a FastAPI streaming endpoint::

        @router.get("/stream")
        async def stream_events(request: Request):
            async def generator():
                async with EventSubscriber() as sub:
                    async for event in sub:
                        if await request.is_disconnected():
                            break
                        yield f"data: {json.dumps(event)}\\n\\n"
            return StreamingResponse(generator(), media_type="text/event-stream")
    """

    def __init__(self, channel: str | None = None) -> None:
        self._channel = channel or settings.redis_event_channel
        self._client: aioredis.Redis | None = None
        self._pubsub: PubSub | None = None

    async def __aenter__(self) -> "EventSubscriber":
        self._client = get_pubsub_client()
        self._pubsub = self._client.pubsub()
        await self._pubsub.subscribe(self._channel)
        logger.debug("EventSubscriber subscribed to '%s'.", self._channel)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._pubsub:
            await self._pubsub.unsubscribe(self._channel)
            await self._pubsub.aclose()
        logger.debug("EventSubscriber unsubscribed from '%s'.", self._channel)

    def __aiter__(self) -> "EventSubscriber":
        return self

    async def __anext__(self) -> dict[str, Any]:
        """
        Block until the next message arrives.
        Skips the initial ``subscribe`` confirmation message (type != 'message').
        Raises StopAsyncIteration if the connection is lost.
        """
        if self._pubsub is None:
            raise StopAsyncIteration

        while True:
            message = await self._pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=30.0,  # yield control to the event loop every 30 s
            )
            if message is None:
                # Timeout with no message – yield control then retry
                # (allows FastAPI to detect disconnected clients)
                continue

            if message.get("type") == "message":
                try:
                    return json.loads(message["data"])
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Malformed event message skipped: %s", exc)
                    continue
