from __future__ import annotations

import logging
from typing import Optional

import boto3
import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


def _aws_config(sys_config: dict[str, str] | None = None) -> dict[str, str]:
    sys_config = sys_config or {}
    return {
        "access_key_id": sys_config.get("camera.aws_access_key_id")
        or settings.aws_access_key_id,
        "secret_access_key": sys_config.get("camera.aws_secret_access_key")
        or settings.aws_secret_access_key,
        "region": sys_config.get("camera.aws_region_name")
        or settings.aws_region,
        "collection_id": sys_config.get("camera.aws_collection_id")
        or settings.aws_collection_id,
    }


def is_configured(sys_config: dict[str, str] | None = None) -> bool:
    cfg = _aws_config(sys_config)
    return all(
        [
            cfg["access_key_id"],
            cfg["secret_access_key"],
            cfg["collection_id"],
        ]
    )


def _client(cfg: dict[str, str]):
    return boto3.client(
        "rekognition",
        aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
        region_name=cfg["region"],
    )


def config_summary(sys_config: dict[str, str] | None = None) -> dict[str, object]:
    cfg = _aws_config(sys_config)
    return {
        "configured": is_configured(sys_config),
        "has_access_key": bool(cfg["access_key_id"]),
        "has_secret_key": bool(cfg["secret_access_key"]),
        "region": cfg["region"],
        "collection_id": cfg["collection_id"] or "",
    }


def search_face_by_image(
    face_crop: np.ndarray,
    sys_config: dict[str, str] | None = None,
) -> tuple[Optional[dict], bool]:
    """Search one cropped face in AWS Rekognition.

    Odoo ir.config_parameter values have priority. Environment variables are
    only a fallback for local/dev runs.
    """
    cfg = _aws_config(sys_config)
    summary = config_summary(sys_config)
    if not summary["configured"]:
        logger.warning(
            "[AWS Rekognition] disabled_by_config has_access_key=%s has_secret_key=%s region=%s collection=%s",
            summary["has_access_key"],
            summary["has_secret_key"],
            summary["region"],
            summary["collection_id"],
        )
        return None, True

    try:
        logger.info(
            "[AWS Rekognition] search start collection=%s region=%s crop_shape=%s threshold=70.0",
            cfg["collection_id"],
            cfg["region"],
            tuple(face_crop.shape) if hasattr(face_crop, "shape") else None,
        )

        client = _client(cfg)
        _, buffer = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        response = client.search_faces_by_image(
            CollectionId=cfg["collection_id"],
            Image={"Bytes": buffer.tobytes()},
            FaceMatchThreshold=70.0,
            MaxFaces=1,
        )

        matches = response.get("FaceMatches", [])
        logger.info(
            "[AWS Rekognition] search done collection=%s matches=%d searched_confidence=%s model=%s",
            cfg["collection_id"],
            len(matches),
            response.get("SearchedFaceConfidence"),
            response.get("FaceModelVersion"),
        )

        if not matches:
            logger.warning(
                "[AWS Rekognition] no match collection=%s searched_confidence=%s",
                cfg["collection_id"],
                response.get("SearchedFaceConfidence"),
            )
            return None, False

        match = matches[0]
        face = match.get("Face", {})
        ext_id = face.get("ExternalImageId", "")
        similarity = match.get("Similarity", 0.0) / 100.0
        logger.info(
            "[AWS Rekognition] match ext_id=%s face_id=%s similarity=%.4f",
            ext_id,
            face.get("FaceId"),
            similarity,
        )
        return {"ext_id": ext_id, "similarity": similarity}, False

    except Exception as exc:
        logger.error("[AWS Rekognition] error: %s", exc, exc_info=True)
        return None, True

def detect_face_crops(
    image_bgr: np.ndarray,
    sys_config: dict[str, str] | None = None,
    max_faces: int = 5,
) -> tuple[list[np.ndarray], bool]:
    """Detect faces with AWS Rekognition and return BGR crops.

    Returns (crops, has_error). Crops are sorted by AWS confidence descending.
    """
    cfg = _aws_config(sys_config)
    summary = config_summary(sys_config)
    if not summary["configured"]:
        logger.warning(
            "[AWS Rekognition] detect disabled_by_config has_access_key=%s has_secret_key=%s region=%s collection=%s",
            summary["has_access_key"],
            summary["has_secret_key"],
            summary["region"],
            summary["collection_id"],
        )
        return [], True

    try:
        h, w = image_bgr.shape[:2]
        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info(
            "[AWS Rekognition] detect start region=%s frame_shape=%s max_faces=%d",
            cfg["region"],
            tuple(image_bgr.shape),
            max_faces,
        )

        response = _client(cfg).detect_faces(
            Image={"Bytes": buffer.tobytes()},
            Attributes=["DEFAULT"],
        )
        details = response.get("FaceDetails", [])
        details.sort(key=lambda item: item.get("Confidence", 0.0), reverse=True)
        logger.info(
            "[AWS Rekognition] detect done faces=%d model=%s",
            len(details),
            response.get("FaceModelVersion"),
        )

        crops: list[np.ndarray] = []
        for idx, detail in enumerate(details[:max_faces], start=1):
            box = detail.get("BoundingBox") or {}
            left = int(max(0, box.get("Left", 0.0) * w))
            top = int(max(0, box.get("Top", 0.0) * h))
            width = int(max(0, box.get("Width", 0.0) * w))
            height = int(max(0, box.get("Height", 0.0) * h))

            pad_x = int(width * 0.18)
            pad_y = int(height * 0.22)
            x1 = max(0, left - pad_x)
            y1 = max(0, top - pad_y)
            x2 = min(w, left + width + pad_x)
            y2 = min(h, top + height + pad_y)
            crop = image_bgr[y1:y2, x1:x2]
            logger.info(
                "[AWS Rekognition] detect face #%d confidence=%.2f box=(%d,%d,%d,%d) crop_shape=%s",
                idx,
                detail.get("Confidence", 0.0),
                x1,
                y1,
                x2,
                y2,
                tuple(crop.shape) if crop.size else None,
            )
            if crop.size:
                crops.append(crop)

        return crops, False

    except Exception as exc:
        logger.error("[AWS Rekognition] detect error: %s", exc, exc_info=True)
        return [], True