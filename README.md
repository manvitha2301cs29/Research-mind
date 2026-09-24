# ResearchMind 🧠

> AI-powered research platform — from paper upload to deep mastery.

**🔗 Live demo:** [research-mind-lmvg.onrender.com](https://research-mind-lmvg.onrender.com/)

> Note: hosted on Render's free tier — the app may take 30–50s to wake up if it's been idle. Bring your own OpenAI API key when using the hosted version (entered in-app).

## Architecture

```
researchmind/
├── backend/                 # FastAPI + LangGraph
│   ├── main.py              # All API routes
│   ├── core/
│   │   ├── config.py        # Settings (env vars)
│   │   ├── clients.py       # LLM + Qdrant clients
│   │   └── models.py        # Shared Pydantic models
│   ├── agents/
│   │   ├── pipeline.py      # LangGraph orchestration
│   │   ├── diagnostic_agent.py
│   │   ├── discovery_agent.py
│   │   ├── graph_builder.py
│   │   ├── scraper_agent.py
│   │   ├── quiz_agent.py
│   │   ├── explainer_agent.py
│   │   └── export_agent.py
│   └── requirements.txt
│
└── frontend/                # React + Vite + Zustand
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   ├── styles/global.css
    │   ├── lib/api.js        # All API calls
    │   ├── stores/useStore.js
    │   ├── components/
    │   │   ├── Primitives.jsx
    │   │   ├── KnowledgeGraph.jsx
    │   │   ├── ChatPanel.jsx
    │   │   ├── ConfidenceQuiz.jsx
    │   │   └── SectionExplainer.jsx
    │   └── pages/
    │       ├── HomeScreen.jsx
    │       ├── Mode1Screen.jsx
    │       └── Mode2Screen.jsx
    ├── package.json
    └── vite.config.js
```

## Setup

### 1. Prerequisites

- Python 3.11+
- Node.js 18+
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally or cloud
- OpenAI API key

### 2. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY at minimum

# Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Run the server
uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend

npm install
npm run dev
# Opens at http://localhost:5173
```

### 4. Open the app

Visit **http://localhost:5173**

---

## Deployment

### Option A — one container (Render / Railway / Fly.io / any VM)

The root `Dockerfile` builds the React app and serves it from FastAPI, so the whole app runs on a single port.

```bash
docker build -t researchmind .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... researchmind
# open http://localhost:8000
```

**Render (free tier):** New → Web Service → connect this repo → Runtime: *Docker*. Add env var `OPENAI_API_KEY` (optional — without it, each user enters their own key in the app). Render sets `$PORT` automatically.

Without `QDRANT_URL`, RAG uses an in-memory vector store (resets on restart). For persistence, create a free [Qdrant Cloud](https://cloud.qdrant.io) cluster and set `QDRANT_URL` + `QDRANT_API_KEY`.

### Option B — Docker Compose (frontend + backend + Qdrant)

```bash
cp backend/.env.example backend/.env   # add OPENAI_API_KEY if you want a server-side key
docker compose up -d --build
# open http://localhost
```

## Features

| Feature | Description |
|---|---|
| **Mode 1 — Deep Dive** | Enter any paper title or arXiv link → full AI explanation with concept linking, section explainer, and 3-mode chat |
| **Mode 2 — Topic Mastery** | Enter a topic → 5-agent pipeline discovers papers, builds knowledge graph, fetches prerequisites, generates quiz |
| **Diagnostic Quiz** | 5 questions to calibrate the learning path before the pipeline runs |
| **Agent Pipeline** | Diagnostic → Discovery → Graph Builder → Scraper → Quiz Gen — live SSE stream |
| **Knowledge Graph** | Interactive canvas showing concept dependencies with mastered/unmastered states |
| **Prerequisites Panel** | YouTube recommendations + key concepts for each unmastered node |
| **Confidence Quiz** | GPT-generated questions with 70% pass gate; fail routes back to prereqs |
| **Section Explainer** | Sequential unlock of 7 paper sections with concept linking (mastered terms highlighted green) |
| **3-Mode Chat** | Standard / Socratic (guides to answer) / Devil's Advocate (challenges claims) |
| **Spaced Repetition** | Concepts scheduled for review with memory-strength bars |
| **PDF Export** | Full session compiled: profile, papers, quiz, Q&A, spaced rep schedule |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Optional | Server-side key. If empty, users enter their own key in the app |
| `OPENAI_MODEL` | Optional | Default: `gpt-4o` |
| `OPENAI_FAST_MODEL` | Optional | Default: `gpt-4o-mini` |
| `QDRANT_URL` | Optional | Default: `http://localhost:6333`; falls back to in-memory if unreachable |
| `QDRANT_API_KEY` | Optional | For Qdrant Cloud |
| `CORS_ORIGINS` | Optional | Extra allowed origins, comma-separated (only if frontend is hosted separately) |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Increases rate limits |
| `YOUTUBE_API_KEY` | Optional | For richer video results |

## Tech Stack

**Backend:** FastAPI · LangGraph · LangChain · OpenAI GPT-4o · Qdrant · pdfplumber · BeautifulSoup · ReportLab

**Frontend:** React 18 · Vite · Zustand · Canvas API · react-hot-toast
