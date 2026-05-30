FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml ./
COPY hermes_trading ./hermes_trading
COPY state ./state
COPY entrypoint.sh ./entrypoint.sh

RUN uv sync && chmod +x entrypoint.sh

ENV HERMES_TRADING_MODE=paper

# Runtime state lives in /app/data (mount a Railway persistent volume here).
# entrypoint.sh seeds it on first run and updates config on every redeploy.
ENV STATE_DIR=/app/data

CMD ["./entrypoint.sh"]
