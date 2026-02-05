# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Local Development (Python 3.12+)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.app:app --reload --host 0.0.0.0 --port 8080
```

### Docker Compose
```bash
./scripts/docker.sh up      # Production (auto-picks free port)
./scripts/docker.sh dev     # Dev mode (auto-reload + bind mount)
./scripts/docker.sh down    # Stop
HOST_PORT=18080 ./scripts/docker.sh up  # Specific port
```

### Health Check
```bash
curl -fsS http://localhost:8080/healthz
```

### No Tests/Lint Configured
This project currently has no test runner or linter.

## Architecture

Single FastAPI service (`api/app.py`) that serves static files and ML endpoints.

### Request Flow
```
Browser → public/index.html
           └→ public/app.js (Plotly charts, event handlers)
               └→ POST /api/fraud/roc
                   └→ api/app.py (trains scikit-learn model, returns ROC data)
```

### Three Use Case Sections
1. **Fraud Detection (interactive)**: Real ML via backend - sends slider params to `/api/fraud/roc`, backend generates synthetic data, trains model, returns ROC curve + AUC
2. **Underwriting (static)**: Parametric ROC curve generator in frontend only
3. **Claims Triage (static)**: Parametric ROC curve generator in frontend only

### Key Backend Components (`api/app.py`)
- `_generate_insurance_data()`: Creates synthetic claim data with overlapping fraud/legit distributions
- `_build_model()`: Supports 4 model types: logistic, extra_trees, random_forest, hist_gbdt
- `_cached_fraud_roc()`: LRU cache (maxsize=128) prevents redundant training
- Static file serving with SPA fallback and path traversal protection

### Key Frontend Components (`public/app.js`)
- `updateFraudChart()`: Coordinates model training and chart updates
- `fetchFraudRoc()`: Calls backend with params (modelType, fraudRate, sampleSize, seed)
- Request deduplication via `cachedDataHash` and `pendingDataHash`
- `safeResizePlotly()`: Handles chart sizing in collapsed `<details>` elements

### API Endpoint

**POST /api/fraud/roc**
```json
{
  "model_type": "logistic|extra_trees|random_forest|hist_gbdt",
  "fraud_rate": 0.001-0.50,
  "sample_size": 200-50000,
  "seed": 1337
}
```
Returns: `auc`, `fpr`, `tpr`, `thresholds`, `feature_importance` (tree models only)

## Where to Make Changes

| Change | Location |
|--------|----------|
| Site content, styling, markup | `public/index.html` |
| Frontend behavior, API calls, charts | `public/app.js` |
| Model behavior, data generation, API | `api/app.py` |

## Chart Rendering Note

Charts are inside `<details>` elements. When collapsed, Plotly computes bad initial sizes. The codebase includes toggle handlers that call `safeResizePlotly()` / `resizeAllCharts()` after opening panels.

## Note on `server.js`

The Express server does NOT implement `/api/fraud/roc` - fraud detection won't work with it. Use FastAPI (Python) for full functionality.
