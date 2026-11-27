FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AETHERIS_HOME=/var/lib/aetheris

WORKDIR /app
COPY pyproject.toml README.md ./
COPY aetheris_microbrains ./aetheris_microbrains

RUN pip install --no-cache-dir ".[mcp]" && \
    groupadd --system aetheris && \
    useradd --system --gid aetheris --create-home aetheris && \
    mkdir -p /var/lib/aetheris && \
    chown -R aetheris:aetheris /var/lib/aetheris /app

USER aetheris
EXPOSE 8080

# Requires AETHERIS_MCP_BEARER_TOKEN at runtime; fails closed if missing.
CMD ["aetheris-mcp", "--transport", "streamable-http", "--profile", "manus", "--host", "0.0.0.0", "--port", "8080"]
