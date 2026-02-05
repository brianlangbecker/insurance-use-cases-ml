# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common commands

### Requirements
- Python 3.12+ recommended

### Install dependencies
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

### Run locally
- `uvicorn api.app:app --reload --host 0.0.0.0 --port 8080`
  - Server listens on `PORT` (default `8080` in Dockerfile)

### Docker
- `docker build -t insurance-use-case-ml .`
- `docker run --rm -p 8080:8080 insurance-use-case-ml`

### Health check
- `curl -fsS http://localhost:8080/healthz` (returns `ok`)

### Tests / lint
No test runner or linter is currently configured.

## High-level architecture

### Runtime shape
This repo is a single FastAPI service that:

- Serves the static site from `public/`.
- Exposes ML endpoints under `/api/...`.
- Provides `/healthz` for deploy/liveness checks.
- Provides SPA-style fallback (unknown GET routes return `public/index.html`).

- **Backend**: `api/app.py`
- **Frontend**: `public/index.html` + `public/app.js`
  - Plotly is loaded via CDN for charts.

### App behavior (what’s “dynamic” vs “static”)
The page contains three “use case” sections:

- **Fraud Detection (interactive / “real ML”)**
  - Sends slider params to the backend (`POST /api/fraud/roc`).
  - Backend generates synthetic insurance claim data and trains a scikit-learn model.
  - Backend returns ROC curve + AUC.
  - Frontend renders charts via `Plotly.react()` in `updateFraudChart()`.
  - Includes caching/invalidations so the backend is only called when data-affecting controls change.

- **Underwriting / Risk Pricing (static teaching curve)**
  - Uses a parametric ROC curve generator (`generateROCCurve`) and fixed business-impact arrays.

- **Claims Triage (static teaching curve)**
  - Same pattern as underwriting: parametric ROC curve + fixed threshold arrays.

### Chart rendering + layout pitfalls
The charts are inside `<details>` elements. When a panel is collapsed, Plotly can compute a bad initial size.
`public/index.html` includes `safeResizePlotly()` / `resizeAllCharts()` and `details` toggle handlers to resize/re-render after opening.

## Where to make changes
- **Change site content, styling, charts**: `public/index.html`
- **Change frontend behavior (including calling the backend)**: `public/app.js` (search for `updateFraudChart`)
- **Change model behavior / data generation / API**: `api/app.py`
