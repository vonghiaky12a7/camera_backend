# backend/app/services/face.py
# =============================================================================
# AI Inference Engine – Face Detection & Embedding Extraction
#
# Responsibilities:
#   1. Initialize InsightFace (buffalo_s) with strict CPU / ONNX thread limits.
#   2. Expose a process-level singleton (FaceAnalyzer) that is created ONCE
#      per Celery worker process via the worker_process_init signal in
#      app/tasks/celery_app.py and reused across all subsequent tasks.
#   3. Provide a clean public API:
#        FaceAnalyzer.initialize()                – called by worker init hook
#        FaceAnalyzer.get()                       – returns the live instance
#        FaceAnalyzer.teardown()                  – called by worker shutdown hook
#        analyzer.extract(image) -> list[Face]    – run detection + embedding
#        analyzer.primary_face(image) -> Face|None– largest / highest-conf face
#
# CPU discipline (per spec §4):
#   - CPUExecutionProvider only; GPU provider is never registered.
#   - ORT_SEQUENTIAL execution mode.
#   - intra_op / inter_op threads pinned via SessionOptions AND env vars.
#   - Thread env vars are set at the top of this module as a last-resort guard
#     (the canonical enforcement is in docker-compose.yml environment: section).
#
# InsightFace buffalo_s model internals:
#   - det_10g.onnx      – RetinaFace detector (MobileNet backbone)
#   - w600k_mbf.onnx    – MobileFaceNet recognizer → 512-dim L2-normalised vector
#   Both ONNX sessions are created with the same restricted SessionOptions.
# =============================================================================

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── Enforce thread limits BEFORE any numpy/onnxruntime import ─────────────────
# This module may be imported in contexts other than the Docker worker
# (e.g. local scripts, tests). Re-applying here is intentional belt-and-braces.
_THREAD_GUARDS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _k, _v in _THREAD_GUARDS.items():
    os.environ.setdefault(_k, _v)

# ── Now safe to import ort + insightface ──────────────────────────────────────
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Public result dataclass
# =============================================================================


@dataclass(slots=True)
class FaceResult:
    """
    Structured output for a single detected face.

    Returned by :meth:`FaceAnalyzer.extract` and
    :meth:`FaceAnalyzer.primary_face`.

    Attributes:
        embedding:    512-dim L2-normalised numpy array from MobileFaceNet.
        det_score:    Detection confidence from RetinaFace (0.0 – 1.0).
        bbox:         Bounding box [x1, y1, x2, y2] in pixel coordinates.
        landmark:     5-point facial landmark array, shape (5, 2). May be None
                      if the model did not produce landmarks.
        embedding_norm: L2 norm of the raw embedding (should be ≈ 1.0 for
                      buffalo_s; useful for sanity checking).
    """

    embedding: np.ndarray  # shape (512,)
    det_score: float
    bbox: np.ndarray  # shape (4,)  [x1,y1,x2,y2]
    landmark: Optional[np.ndarray] = None  # shape (5, 2)
    embedding_norm: float = field(init=False)

    def __post_init__(self) -> None:
        self.embedding_norm = float(np.linalg.norm(self.embedding))

    @property
    def bbox_area(self) -> float:
        """Pixel area of the bounding box (width × height)."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float((x2 - x1) * (y2 - y1)))

    def to_list(self) -> list[float]:
        """Return embedding as a plain Python list (for JSON / pgvector insert)."""
        return self.embedding.tolist()


# =============================================================================
# ONNX SessionOptions factory
# =============================================================================


def _build_session_options() -> ort.SessionOptions:
    """
    Build a strictly CPU-limited ONNX Runtime SessionOptions object.

    Settings applied:
        execution_mode       = ORT_SEQUENTIAL   – runs graph nodes sequentially,
                                                  avoiding intra-graph parallelism
                                                  that fights Celery's prefork pool.
        intra_op_num_threads = ORT_INTRA (2)    – threads inside a single operator
                                                  (e.g. convolution). 2 gives a
                                                  slight speed gain over 1 on most
                                                  modern CPUs without blowing up
                                                  total thread count.
        inter_op_num_threads = ORT_INTER (1)    – threads used to run independent
                                                  graph nodes in parallel. Set to 1
                                                  because buffalo_s graphs are
                                                  largely sequential.
        graph_optimization_level = ORT_ENABLE_ALL – fold constants, fuse ops.
                                                  Safe for CPU; improves latency
                                                  ~10-20% with no accuracy change.
        enable_mem_pattern   = True             – reuse memory allocations across
                                                  runs with the same input shape.
        enable_cpu_mem_arena = True             – use ORT's arena allocator;
                                                  reduces malloc/free overhead.
    """
    opts = ort.SessionOptions()

    # Sequential execution – no intra-graph thread parallelism
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Thread limits from settings (backed by env vars)
    opts.intra_op_num_threads = settings.ort_intra_op_num_threads
    opts.inter_op_num_threads = settings.ort_inter_op_num_threads

    # Graph optimisation (safe, no model mutations on disk)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Memory management
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    # Disable telemetry / profiling in production
    opts.enable_profiling = False

    logger.debug(
        "ONNX SessionOptions: mode=SEQUENTIAL intra=%d inter=%d",
        opts.intra_op_num_threads,
        opts.inter_op_num_threads,
    )
    return opts


# =============================================================================
# InsightFace provider configuration
# =============================================================================


def _cpu_providers() -> list:
    """
    Return the ONNX Runtime provider list restricted to CPU only.

    CPUExecutionProvider options:
        arena_extend_strategy   = kNextPowerOfTwo – standard growth strategy.
        cpu_memory_arena_extend_size = 0          – let ORT decide arena size.

    We explicitly do NOT include CUDAExecutionProvider or
    TensorrtExecutionProvider even if onnxruntime-gpu were installed by mistake;
    the CPU provider is the only registered provider so the model can never
    accidentally run on GPU.
    """
    return [
        (
            "CPUExecutionProvider",
            {
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cpu_memory_arena_extend_size": 0,
            },
        )
    ]


# =============================================================================
# Monkey-patch: inject SessionOptions into InsightFace ONNX session creation
# =============================================================================


def _patch_insightface_ort_session(sess_opts: ort.SessionOptions) -> None:
    """
    InsightFace creates its ONNX sessions internally via
    ``onnxruntime.InferenceSession(model_path, providers=...)``.
    It does not expose a SessionOptions parameter in its public API.

    This patch replaces ``onnxruntime.InferenceSession`` with a wrapper
    that injects our pre-configured SessionOptions transparently, so that
    EVERY session InsightFace creates (detector + recognizer) respects the
    thread limits without requiring a fork of the insightface library.

    The patch is applied once, before ``FaceAnalysis.prepare()`` is called,
    and is scoped to this process only (each Celery worker process applies it
    independently after fork).
    """
    _original_inference_session = ort.InferenceSession

    class _PatchedInferenceSession(_original_inference_session):  # type: ignore[misc]
        def __init__(
            self,
            path_or_bytes: str | bytes,
            sess_options: ort.SessionOptions | None = None,
            providers: list | None = None,
            **kwargs,
        ) -> None:
            # Always use our controlled SessionOptions; ignore whatever
            # InsightFace passed (usually None)
            super().__init__(
                path_or_bytes,
                sess_options=sess_opts,
                providers=providers or _cpu_providers(),
                **kwargs,
            )

    ort.InferenceSession = _PatchedInferenceSession  # type: ignore[misc]
    logger.debug("onnxruntime.InferenceSession patched with CPU SessionOptions.")


# =============================================================================
# FaceAnalyzer – process-level singleton
# =============================================================================


class FaceAnalyzer:
    """
    Process-level singleton wrapping InsightFace's FaceAnalysis pipeline.

    Lifecycle (managed by Celery worker signals in app/tasks/celery_app.py):

        worker_process_init    → FaceAnalyzer.initialize()
        task execution         → FaceAnalyzer.get().extract(image)
        worker_process_shutdown→ FaceAnalyzer.teardown()

    Thread safety:
        Each Celery prefork worker process has exactly ONE FaceAnalyzer
        instance. Multiple concurrent Celery tasks do NOT share the same
        process, so no locking is required.

    Usage (inside a Celery task)::

        from app.services.face import FaceAnalyzer

        analyzer = FaceAnalyzer.get()
        face = analyzer.primary_face(image_array)
        if face:
            embedding = face.to_list()   # → list[float] ready for pgvector
    """

    _instance: Optional["FaceAnalyzer"] = None

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    def __init__(self, app: FaceAnalysis) -> None:
        self._app = app
        self._calls = 0  # rolling call counter for metrics / logging

    @classmethod
    def initialize(cls) -> None:
        """
        Load and warm up the buffalo_s model.

        Steps:
          1. Apply ONNX SessionOptions monkey-patch.
          2. Construct FaceAnalysis with CPUExecutionProvider.
          3. Call prepare() to download (first run) or load cached weights.
          4. Run a single warm-up pass with a blank image to JIT-compile the
             ONNX graph and pre-allocate memory arenas. Without this, the
             first real task pays a cold-start penalty of 200-800 ms.

        Raises:
            RuntimeError: If model files cannot be loaded.
        """
        if cls._instance is not None:
            logger.warning(
                "FaceAnalyzer.initialize() called more than once — skipping."
            )
            return

        logger.info(
            "Initialising InsightFace model '%s' from '%s' …",
            settings.insightface_model_name,
            settings.insightface_model_dir,
        )
        t0 = time.perf_counter()

        # ── Step 1: patch ort.InferenceSession ────────────────────────────────
        sess_opts = _build_session_options()
        _patch_insightface_ort_session(sess_opts)

        # ── Step 2: build FaceAnalysis ─────────────────────────────────────────
        # `allowed_modules` restricts loading to detector + recognizer only.
        # Omitting 'landmark' and 'attribute' reduces memory and init time.
        face_app = FaceAnalysis(
            name=settings.insightface_model_name,
            root=settings.insightface_model_dir,
            allowed_modules=["detection", "recognition"],
            providers=_cpu_providers(),
        )

        # ── Step 3: prepare (download / load from cache) ──────────────────────
        # ctx_id=-1 forces CPU regardless of what InsightFace auto-detects.
        # det_thresh sets the minimum detection confidence to forward to the
        # recognizer (saves CPU on low-confidence blobs).
        face_app.prepare(
            ctx_id=-1,  # CPU
            det_thresh=settings.insightface_det_thresh,
            det_size=(640, 640),  # standard input resolution
        )

        # ── Step 4: warm-up pass ───────────────────────────────────────────────
        _warmup_image = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            face_app.get(_warmup_image)
            logger.debug("ONNX warm-up pass completed.")
        except Exception as exc:  # noqa: BLE001
            # Warm-up failure is non-fatal; log and continue.
            logger.warning("Warm-up pass raised (non-fatal): %s", exc)

        elapsed = (time.perf_counter() - t0) * 1000
        cls._instance = cls(face_app)

        logger.info(
            "FaceAnalyzer ready — model=%s det_thresh=%.2f init_time=%.0f ms",
            settings.insightface_model_name,
            settings.insightface_det_thresh,
            elapsed,
        )

    @classmethod
    def get(cls) -> "FaceAnalyzer":
        """
        Return the live singleton instance.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called yet.
        """
        if cls._instance is None:
            raise RuntimeError(
                "FaceAnalyzer has not been initialised. "
                "Call FaceAnalyzer.initialize() first (normally via the "
                "Celery worker_process_init signal)."
            )
        return cls._instance

    @classmethod
    def teardown(cls) -> None:
        """Release resources and clear the singleton on worker shutdown."""
        if cls._instance is not None:
            cls._instance = None
            logger.info("FaceAnalyzer singleton released.")

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> list[FaceResult]:
        if image is None or image.size == 0:
            logger.warning("extract() received an empty/invalid image; returning [].")
            return []

        image = _ensure_bgr(image)
        logger.debug(f"Starting Face extraction. Image shape: {image.shape}")

        t0 = time.perf_counter()
        raw_faces = self._app.get(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._calls += 1

        # Đổi từ logger.debug sang logger.info để dễ theo dõi ở Production
        logger.info(
            f"AI Inference (Call #{self._calls}): {len(raw_faces)} face(s) detected in {elapsed_ms:.1f}ms"
        )

        results: list[FaceResult] = []
        for i, face in enumerate(raw_faces):
            embedding = getattr(face, "embedding", None)
            det_score = float(getattr(face, "det_score", 0.0))

            if embedding is None:
                logger.warning(
                    f"Face #{i} detected but NO embedding returned. Score: {det_score:.4f}"
                )
                continue

            results.append(
                FaceResult(
                    embedding=np.asarray(embedding, dtype=np.float32),
                    det_score=det_score,
                    bbox=np.asarray(face.bbox, dtype=np.float32),
                    landmark=(
                        np.asarray(face.kps, dtype=np.float32)
                        if getattr(face, "kps", None) is not None
                        else None
                    ),
                )
            )

        results.sort(key=lambda f: f.det_score, reverse=True)
        return results

    def primary_face(self, image: np.ndarray) -> Optional[FaceResult]:
        """
        Return the single most prominent face in ``image``, or ``None``.

        "Most prominent" is defined as the largest bounding-box area among
        faces whose detection score meets the configured threshold. This
        heuristic correctly selects the employee facing the camera directly
        when a crowd is visible in the background.

        Args:
            image: BGR uint8 numpy array.

        Returns:
            The best :class:`FaceResult`, or ``None`` if no face meets the
            detection threshold.
        """
        faces = self.extract(image)

        # Filter to faces that meet the detection confidence threshold
        qualified = [f for f in faces if f.det_score >= settings.insightface_det_thresh]

        if not qualified:
            return None

        # Among qualified faces, pick the largest (closest to camera)
        return max(qualified, key=lambda f: f.bbox_area)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of extract() calls since initialization."""
        return self._calls

    def health_check(self) -> dict:
        """
        Return a dict suitable for inclusion in the /health endpoint response.

        Example response::

            {
                "status": "ok",
                "model": "buffalo_s",
                "call_count": 1042,
                "det_thresh": 0.5,
                "ort_intra_threads": 2,
                "ort_inter_threads": 1,
            }
        """
        return {
            "status": "ok",
            "model": settings.insightface_model_name,
            "call_count": self._calls,
            "det_thresh": settings.insightface_det_thresh,
            "ort_intra_threads": settings.ort_intra_op_num_threads,
            "ort_inter_threads": settings.ort_inter_op_num_threads,
        }


# =============================================================================
# Image pre-processing utilities
# =============================================================================


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Ensure ``image`` is a 3-channel BGR uint8 array, as expected by InsightFace.

    Handles:
        - BGRA (4-channel)  → drop alpha channel
        - Grayscale (2-D)   → convert to BGR
        - Float images      → scale to uint8
    """
    if image.dtype != np.uint8:
        # Float image in [0, 1] or [0, 255]; normalise to uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if image.ndim == 2:
        # Grayscale → BGR
        import cv2  # noqa: PLC0415

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # BGRA → BGR
        image = image[:, :, :3]

    return image


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Decode raw image bytes (JPEG / PNG) to a BGR numpy array.

    This is the standard entry point for images received from Hikvision
    webhook payloads or read from the webhook_snapshots volume.

    Args:
        data: Raw image bytes.

    Returns:
        BGR uint8 numpy array.

    Raises:
        ValueError: If the bytes cannot be decoded to a valid image.
    """
    import cv2  # noqa: PLC0415

    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # always BGR

    if image is None:
        raise ValueError(
            f"cv2.imdecode failed — received {len(data)} bytes that could not "
            "be decoded as a valid image (expected JPEG or PNG)."
        )

    return image


def load_image_from_path(path: str | Path) -> np.ndarray:
    """
    Load an image from disk to a BGR numpy array.

    Args:
        path: Absolute or relative path to a JPEG/PNG file.

    Returns:
        BGR uint8 numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file cannot be decoded.
    """
    import cv2  # noqa: PLC0415

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cv2.imread failed for: {p}")

    return image


def resize_for_detection(
    image: np.ndarray,
    target_size: int = 640,
) -> np.ndarray:
    """
    Resize ``image`` so its longest side equals ``target_size``, preserving
    aspect ratio. Pads with black to make it exactly ``target_size × target_size``.

    InsightFace resizes internally to det_size, but pre-resizing here
    reduces the bytes transferred into the ONNX session and makes latency
    more predictable when cameras send large snapshots (e.g. 4K).

    Args:
        image:       BGR uint8 array.
        target_size: Side length of the output square (default 640).

    Returns:
        Square BGR uint8 array of shape (target_size, target_size, 3).
    """
    import cv2  # noqa: PLC0415

    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    return canvas
