# =============================================================================
# AI Camera System – Backend Dockerfile (Optimized for 2026)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 – base: Cập nhật OS và Python 3.12
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS base

# Cập nhật thông tin maintainer
LABEL maintainer="your-team@example.com"

# Thiết lập môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tối ưu cho AI Inference trên CPU
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

# FIX VULNERABILITIES & TỐI ƯU HÓA: Đã bỏ libgl1, libglib2.0-0
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo user không có quyền root
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# -----------------------------------------------------------------------------
# Stage 2 – builder: Sử dụng 'uv' để cài đặt dependencies cực nhanh
# -----------------------------------------------------------------------------
FROM base AS builder

# Cài đặt uv - công cụ thay thế pip/poetry nhanh nhất hiện nay
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Cài đặt build tool cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements
COPY requirements.txt .

# Tạo virtual environment và cài đặt dependencies bằng uv (nhanh hơn pip gấp 10-100 lần)
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3 – production: Image cuối cùng siêu nhẹ
# -----------------------------------------------------------------------------
FROM base AS production

# Copy venv từ builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy mã nguồn với quyền sở hữu của appuser
COPY --chown=appuser:appgroup ./app /app/app

# Tạo thư mục cache cho AI models và snapshots
RUN mkdir -p /tmp/snapshots /app/models_cache && \
    chown -R appuser:appgroup /tmp/snapshots /app/models_cache

# Chuyển sang user bảo mật
USER appuser

EXPOSE 8888

# Sử dụng uvicorn với các thiết lập tối ưu
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "2"]