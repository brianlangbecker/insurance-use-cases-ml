# insurance-use-case-ml

Interactive ROC/AUC insurance use-case demo served by a single **Python FastAPI** service.

- **Frontend**: static UI in `public/` (entrypoint: `public/index.html`)
- **Backend**: `api/app.py`
  - Serves the static site (SPA-style fallback)
  - Exposes ML endpoints under `/api/...`
  - Provides `/healthz` for deploy/liveness checks

## Quick start (Docker Compose)

The easiest way to run this on macOS is via Docker Compose. A helper script is included that also avoids common “port already in use” issues.

```sh
./scripts/docker.sh up
```

It prints the URL it selected (defaults to `8080` when free, otherwise tries `18080`, `3000`, `5000`, …).

Stop everything:

```sh
./scripts/docker.sh down
```

### Choose a specific host port

```sh
HOST_PORT=18080 ./scripts/docker.sh up
```

### Dev mode in Docker (auto-reload + bind mount)

```sh
./scripts/docker.sh dev
```

### Docker Compose (manual)

```sh
docker compose up --build
```

Custom host port:

```sh
HOST_PORT=18080 docker compose up --build
```

Dev mode (auto-reload + bind mount):

```sh
docker compose --profile dev up --build app-dev
```

### Docker (plain)

```sh
docker build -t insurance-use-case-ml .
docker run --rm -p 8080:8080 insurance-use-case-ml
```

## Run locally (Python)

Requirements: Python 3.12+ recommended.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.app:app --reload --host 0.0.0.0 --port 8080
```

Open:

- http://localhost:8080

Health check:

```sh
curl -fsS http://localhost:8080/healthz
```

## API

### POST /api/fraud/roc

Trains a scikit-learn model on synthetic insurance claim data and returns a ROC curve + AUC.

**Request:**
- `model_type`: `logistic`, `hist_gbdt`, `extra_trees`, or `random_forest`
- `fraud_rate`: 0.001–0.50
- `sample_size`: 200–50000
- `seed`: random seed (default 1337)

**Response:** `auc`, `fpr`, `tpr`, `thresholds`, `feature_importance` (tree models only)

Example:

```sh
curl -sS http://localhost:8080/api/fraud/roc \
  -H 'content-type: application/json' \
  -d '{"model_type":"hist_gbdt","fraud_rate":0.05,"sample_size":8000,"seed":1337}' | python -m json.tool
```

## Repo layout

- `api/app.py`: FastAPI app (static serving + API)
- `public/index.html`: UI markup + styles
- `public/app.js`: chart logic (Plotly) + calls `/api/fraud/roc`
- `docs/MODELS.md`: ML model technical reference
- `Dockerfile`: container image
- `compose.yaml`: compose setup (prod-ish + dev profile)

## Port forwarding / macOS troubleshooting

If you see an error like:

- `Bind for 0.0.0.0:8080 failed: port is already allocated`

…it means something on your Mac is already listening on that port. Options:

1) Pick a different host port:

```sh
HOST_PORT=18080 ./scripts/docker.sh up
```

2) Find what is using the port:

```sh
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

If `lsof` shows nothing but Docker still reports the port is allocated, it’s likely another Docker container is already publishing it:

```sh
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
```

### Note about `server.js` / Node

This repo also contains a small Express server (`server.js`) that can serve `public/`.

- It does **not** implement the FastAPI ML endpoint (`/api/fraud/roc`), so the “Fraud Detection” section will not work when using the Node server.
- It commonly binds to port `8080`, which can conflict with Docker Compose.

## Tests / lint

No test runner or linter is currently configured.
