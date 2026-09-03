# Pharmacy demo agent. Thin audio bridge: FastAPI + uvicorn, no build step,
# no native deps. Everything heavy happens inside the Voice Agent API.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Deps in their own layer so source edits don't re-resolve. --frozen keeps the
# websockets <14 pin honest; va.py uses extra_headers=, removed in 14.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# pyproject declares no [build-system], so uv installs deps only and never
# builds `server` — we run it from WORKDIR instead.
COPY server/ ./server/
COPY web/ ./web/
COPY prompts/ ./prompts/
# Every data file server/ opens at IMPORT time must be here. Missing one does not
# fail the build -- it crashes the container on boot and Fly reports only "the app
# appears to be crashing". scripts/check_container_files.py guards this.
COPY catalog.json knowledge-base.json account.json ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
