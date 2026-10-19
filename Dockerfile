FROM python:3.11-slim

LABEL org.opencontainers.image.title="contextual-bandit-decision-ops"
LABEL org.opencontainers.image.description="Real-time offer recommendations with contextual Thompson Sampling"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docker/entrypoint.sh /usr/local/bin/offer-service-entrypoint

RUN python -m pip install --upgrade pip && \
    python -m pip install . && \
    chmod 0755 /usr/local/bin/offer-service-entrypoint

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()" "http://127.0.0.1:${PORT}/health"

CMD ["offer-service-entrypoint"]
