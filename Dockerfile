FROM python:3.11-slim

LABEL org.opencontainers.image.title="contextual-bandit-decision-ops"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.description="CPU-only local/staging contextual bandit service"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install .

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"

CMD ["contextual-bandit-service", "--host", "0.0.0.0", "--port", "8000", "--decision-log", "/tmp/contextual-bandit-logs/decisions.jsonl", "--feedback-log", "/tmp/contextual-bandit-logs/feedback.jsonl"]
