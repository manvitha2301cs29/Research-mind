# ResearchMind — single-container image (frontend + backend on one port)
# Works on Render, Railway, Fly.io, or any Docker host.
#   docker build -t researchmind .
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... researchmind

# ---- 1. build the React frontend ----
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ---- 2. python backend that also serves the built frontend ----
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY --from=frontend /fe/dist ./static
RUN mkdir -p uploads exports cache db

EXPOSE 8000
# Render/Railway inject $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
