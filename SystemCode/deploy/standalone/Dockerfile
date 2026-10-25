# Responsibilities: Build the web application and its existing isolated PDF runtime without UE.
# Implementation: Compile the browser module in Node; install the unchanged Linux dependency lock;
# install parser-only dependencies into /opt/pdf-sandbox using the distribution interpreter.
# Related Modules: compose.yaml runs this image as ASGI, Celery and the evaluation worker.
# Declaration Index: None.
# Variable Index: PYTHON* avoids bytecode writes and buffered logs; PATH selects the app interpreter.
FROM node:22.14.0-bookworm-slim AS player
WORKDIR /player
COPY backend/frontend/digital-human/package.json backend/frontend/digital-human/pnpm-lock.yaml backend/frontend/digital-human/pnpm-workspace.yaml ./
RUN npm install --global pnpm@11.25.0 && pnpm install --frozen-lockfile
COPY backend/frontend/digital-human/src ./src
RUN pnpm run build && node --check dist/pixel-player.js

FROM python:3.12.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install --yes --no-install-recommends \
      python3 python3-pip bubblewrap libseccomp2 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/requirements-linux.lock.txt /tmp/requirements-linux.lock.txt
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/requirements-linux.lock.txt \
    && python -m pip check
COPY backend/sandbox/requirements.txt /tmp/pdf-requirements.txt
RUN mkdir -p /opt/pdf-sandbox/lib /opt/pdf-sandbox/tools/usr/bin \
    && /usr/bin/python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
       --target /opt/pdf-sandbox/lib -r /tmp/pdf-requirements.txt \
    && ln -s /usr/bin/bwrap /opt/pdf-sandbox/tools/usr/bin/bwrap \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin intellihire \
    && mkdir -p /var/lib/intellihire/capacity \
    && chown -R 10001:10001 /var/lib/intellihire
WORKDIR /app
COPY agents ./agents
COPY app ./app
COPY shared ./shared
COPY evaluation ./evaluation
COPY ai_security ./ai_security
COPY rag ./rag
COPY backend ./backend
COPY deploy ./deploy
COPY --from=player /player/dist/pixel-player.js ./backend/frontend/digital-human/dist/pixel-player.js
RUN python deploy/smoke.py --check-frontend-bundle
WORKDIR /app/backend
USER 10001:10001
CMD ["python", "-m", "uvicorn", "config.asgi:application", "--host", "127.0.0.1", "--port", "8765", "--ws", "websockets-sansio", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1", "--no-access-log"]
