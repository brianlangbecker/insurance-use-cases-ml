from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = (REPO_ROOT / "public").resolve()

FEATURE_NAMES = [
    "claim_amount",
    "policy_age_days",
    "previous_claims",
    "is_weekend",
    "days_to_report",
    "claim_to_premium_ratio",
]

ModelType = Literal["logistic", "extra_trees", "random_forest", "hist_gbdt"]


class FraudRocRequest(BaseModel):
    model_type: ModelType
    fraud_rate: float = Field(..., ge=0.001, le=0.50, description="Fraction of fraud claims")
    sample_size: int = Field(..., ge=200, le=50000)
    seed: int = Field(1337, ge=0, le=2**31 - 1)


class FraudRocResponse(BaseModel):
    model_type: ModelType
    model_name: str
    auc: float
    fpr: list[float]
    tpr: list[float]
    thresholds: list[float]
    feature_importance: dict[str, float] | None = None


def _generate_insurance_data(*, n_samples: int, fraud_rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Labels drive the rest: we first set the prevalence (fraud_rate), then generate features.
    #
    # We intentionally add:
    # - overlapping distributions (so AUC isn't ~1.0)
    # - multiple "segments" (so one model can stand out; boosting/tree models do better than linear)
    n_frauds = int(round(n_samples * fraud_rate))
    n_frauds = max(1, min(n_samples - 1, n_frauds))
    n_legit = n_samples - n_frauds

    y = np.concatenate([np.zeros(n_legit, dtype=np.int32), np.ones(n_frauds, dtype=np.int32)])

    fraud_mask = y == 1
    legit_mask = ~fraud_mask

    # Segment assignment (piecewise patterns -> non-linear decision boundary)
    # Fraud segments: 0=obvious, 1=organized, 2=camouflaged
    # Legit segments: 0=normal, 1=high-amount legit, 2=suspicious-but-legit
    fraud_seg = rng.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20])
    legit_seg = rng.choice([0, 1, 2], size=n_samples, p=[0.75, 0.17, 0.08])
    seg = np.where(fraud_mask, fraud_seg, legit_seg).astype(np.int32)

    # Allocate arrays
    claim_amount = np.zeros(n_samples, dtype=np.float32)
    policy_age_days = np.zeros(n_samples, dtype=np.float32)
    previous_claims = np.zeros(n_samples, dtype=np.float32)
    is_weekend = np.zeros(n_samples, dtype=np.float32)
    days_to_report = np.zeros(n_samples, dtype=np.float32)
    claim_to_premium_ratio = np.zeros(n_samples, dtype=np.float32)

    # Convenience helpers
    def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(x, lo, hi)

    def _lognormal_from_log10(mu_log10: float, sigma_log10: float, size: int) -> np.ndarray:
        # Return lognormal with parameters specified in log10-space.
        ln10 = np.log(10.0)
        mu = mu_log10 * ln10
        sigma = sigma_log10 * ln10
        return rng.lognormal(mean=mu, sigma=sigma, size=size).astype(np.float32)

    # --- Legit segments ---
    # 0) normal legit
    m = legit_mask & (seg == 0)
    n = int(np.sum(m))
    if n:
        claim_amount[m] = _lognormal_from_log10(3.90, 0.42, n)
        policy_age_days[m] = _clip(rng.normal(900.0, 520.0, size=n).astype(np.float32), 60.0, 2500.0)
        previous_claims[m] = _clip(rng.poisson(lam=1.1, size=n).astype(np.float32), 0.0, 6.0)
        is_weekend[m] = (rng.random(n) < 0.28).astype(np.float32)
        # Low-ish days + low-ish ratio (but with noise)
        days_to_report[m] = _clip(rng.lognormal(mean=0.35, sigma=0.70, size=n).astype(np.float32), 0.10, 30.0)
        claim_to_premium_ratio[m] = _clip(rng.normal(0.25, 0.20, size=n).astype(np.float32), 0.05, 2.5)

    # 1) high-amount legit (looks scary on claim_amount alone)
    m = legit_mask & (seg == 1)
    n = int(np.sum(m))
    if n:
        claim_amount[m] = _lognormal_from_log10(4.10, 0.40, n)
        policy_age_days[m] = _clip(rng.normal(1150.0, 520.0, size=n).astype(np.float32), 120.0, 2500.0)
        previous_claims[m] = _clip(rng.poisson(lam=0.9, size=n).astype(np.float32), 0.0, 6.0)
        is_weekend[m] = (rng.random(n) < 0.22).astype(np.float32)
        # High-ish days + low-ish ratio
        days_to_report[m] = _clip(rng.lognormal(mean=1.90, sigma=0.60, size=n).astype(np.float32), 0.10, 30.0)
        claim_to_premium_ratio[m] = _clip(rng.normal(0.25, 0.18, size=n).astype(np.float32), 0.05, 2.5)

    # 2) suspicious-but-legit (high ratio but long policy + low previous claims)
    m = legit_mask & (seg == 2)
    n = int(np.sum(m))
    if n:
        claim_amount[m] = _lognormal_from_log10(3.98, 0.42, n)
        policy_age_days[m] = _clip(rng.normal(1400.0, 450.0, size=n).astype(np.float32), 300.0, 2500.0)
        previous_claims[m] = _clip(rng.poisson(lam=0.8, size=n).astype(np.float32), 0.0, 4.0)
        is_weekend[m] = (rng.random(n) < 0.26).astype(np.float32)
        days_to_report[m] = _clip(rng.lognormal(mean=1.80, sigma=0.55, size=n).astype(np.float32), 0.10, 30.0)
        claim_to_premium_ratio[m] = _clip(rng.normal(0.85, 0.35, size=n).astype(np.float32), 0.10, 2.0)

    # --- Fraud segments ---
    # 0) obvious fraud (high ratio + low days + short policy age)
    m = fraud_mask & (seg == 0)
    n = int(np.sum(m))
    if n:
        claim_amount[m] = _lognormal_from_log10(4.05, 0.38, n)
        policy_age_days[m] = _clip(rng.normal(450.0, 350.0, size=n).astype(np.float32), 30.0, 1800.0)
        previous_claims[m] = _clip(rng.poisson(lam=2.2, size=n).astype(np.float32), 0.0, 8.0)
        is_weekend[m] = (rng.random(n) < 0.48).astype(np.float32)
        days_to_report[m] = _clip(rng.lognormal(mean=0.20, sigma=0.55, size=n).astype(np.float32), 0.05, 30.0)
        claim_to_premium_ratio[m] = _clip(rng.normal(1.35, 0.35, size=n).astype(np.float32), 0.30, 2.5)

    # 1) organized fraud (moderate ratio + high days + many previous claims)
    # Distinguishable via previous_claims and policy_age combo
    m = fraud_mask & (seg == 1)
    n = int(np.sum(m))
    if n:
        claim_amount[m] = _lognormal_from_log10(4.10, 0.38, n)
        policy_age_days[m] = _clip(rng.normal(550.0, 380.0, size=n).astype(np.float32), 30.0, 1800.0)
        previous_claims[m] = _clip(rng.poisson(lam=2.8, size=n).astype(np.float32), 0.0, 9.0)
        is_weekend[m] = (rng.random(n) < 0.42).astype(np.float32)
        days_to_report[m] = _clip(rng.lognormal(mean=1.80, sigma=0.55, size=n).astype(np.float32), 0.05, 30.0)
        claim_to_premium_ratio[m] = _clip(rng.normal(0.70, 0.25, size=n).astype(np.float32), 0.15, 2.5)

    # 2) camouflaged fraud (looks more legit but still detectable)
    m = fraud_mask & (seg == 2)
    n = int(np.sum(m))
    if n:
        claim_amount[m] = _lognormal_from_log10(3.92, 0.38, n)
        policy_age_days[m] = _clip(rng.normal(650.0, 380.0, size=n).astype(np.float32), 60.0, 2000.0)
        previous_claims[m] = _clip(rng.poisson(lam=1.8, size=n).astype(np.float32), 0.0, 7.0)
        is_weekend[m] = (rng.random(n) < 0.38).astype(np.float32)
        days_to_report[m] = _clip(rng.lognormal(mean=1.40, sigma=0.50, size=n).astype(np.float32), 0.20, 30.0)
        claim_to_premium_ratio[m] = _clip(rng.normal(0.75, 0.28, size=n).astype(np.float32), 0.15, 2.5)

    # Small measurement noise so boundaries aren't razor-thin
    claim_amount *= np.exp(rng.normal(0.0, 0.10, size=n_samples)).astype(np.float32)
    policy_age_days = _clip(policy_age_days + rng.normal(0.0, 40.0, size=n_samples).astype(np.float32), 30.0, 2500.0)
    days_to_report = _clip(days_to_report + rng.normal(0.0, 0.25, size=n_samples).astype(np.float32), 0.05, 30.0)
    claim_to_premium_ratio = _clip(claim_to_premium_ratio + rng.normal(0.0, 0.05, size=n_samples).astype(np.float32), 0.05, 2.5)

    # Mirror the browser feature normalization in public/app.js (featuresToArray)
    X = np.column_stack(
        [
            np.log(np.maximum(claim_amount, 1.0)) / 10.0,
            policy_age_days / 1500.0,
            previous_claims / 4.0,
            is_weekend,
            days_to_report / 10.0,
            claim_to_premium_ratio / 2.0,
        ]
    ).astype(np.float32)

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def _class_balance_weights(y: np.ndarray) -> np.ndarray:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    n = float(len(y))

    # Avoid divide-by-zero; upstream validation should prevent this, but keep it robust.
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=np.float32)

    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def _build_model(model_type: ModelType, *, seed: int) -> tuple[Any, str]:
    if model_type == "logistic":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        solver="lbfgs",
                        n_jobs=1,
                    ),
                ),
            ]
        )
        return model, "Logistic Regression"

    if model_type == "extra_trees":
        # Constrained so it's competitive but doesn't dominate; helps the demo show boosting's advantage.
        model = ExtraTreesClassifier(
            n_estimators=180,
            random_state=seed,
            n_jobs=-1,
            max_depth=6,
            min_samples_leaf=25,
            max_features="sqrt",
        )
        return model, "Extra Trees"

    if model_type == "random_forest":
        # Intentionally weaker baseline for the demo.
        model = RandomForestClassifier(
            n_estimators=120,
            random_state=seed,
            n_jobs=-1,
            max_depth=3,
            min_samples_leaf=60,
            max_features="sqrt",
        )
        return model, "Random Forest"

    if model_type == "hist_gbdt":
        # Boosting is the "best" model in the UI; give it a bit more capacity.
        model = HistGradientBoostingClassifier(
            random_state=seed,
            max_depth=6,
            learning_rate=0.06,
            max_iter=450,
        )
        return model, "Histogram Gradient Boosting"

    raise ValueError(f"Unknown model_type: {model_type}")


def _fit_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    weights = _class_balance_weights(y_train)

    if isinstance(model, Pipeline):
        model.fit(X_train, y_train, clf__sample_weight=weights)
        return model

    model.fit(X_train, y_train, sample_weight=weights)
    return model


def _predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs[:, 1]

    # Fallback: many classifiers provide decision_function; map to (0,1) via sigmoid.
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))

    raise ValueError("Model does not support probability scoring")


def _model_feature_importance(model: Any) -> dict[str, float] | None:
    base = model
    if isinstance(model, Pipeline):
        base = model.named_steps.get("clf")

    importances = getattr(base, "feature_importances_", None)
    if importances is None:
        return None

    arr = np.asarray(importances, dtype=np.float64)
    if arr.shape[0] != len(FEATURE_NAMES):
        return None

    total = float(np.sum(arr))
    if total > 0:
        arr = arr / total

    return {name: float(val) for name, val in zip(FEATURE_NAMES, arr)}


@lru_cache(maxsize=128)
def _cached_fraud_roc(
    model_type: ModelType,
    fraud_rate_rounded: float,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    X, y = _generate_insurance_data(n_samples=sample_size, fraud_rate=fraud_rate_rounded, rng=rng)

    # If fraud_rate is extremely small, stratify can fail. Our request validation should prevent
    # this, but keep it safe.
    stratify = y if (np.any(y == 0) and np.any(y == 1)) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )

    model, model_name = _build_model(model_type, seed=seed)
    model = _fit_model(model, X_train, y_train)

    scores = _predict_scores(model, X_test)
    auc = float(roc_auc_score(y_test, scores))

    fpr, tpr, thresholds = roc_curve(y_test, scores)

    thresholds = np.where(np.isinf(thresholds), 1.0, thresholds)
    thresholds = np.clip(thresholds, 0.0, 1.0)

    payload: dict[str, Any] = {
        "model_type": model_type,
        "model_name": model_name,
        "auc": auc,
        "fpr": [float(x) for x in fpr.tolist()],
        "tpr": [float(x) for x in tpr.tolist()],
        "thresholds": [float(x) for x in thresholds.tolist()],
        "feature_importance": _model_feature_importance(model),
    }

    return payload


def _safe_public_join(root: Path, req_path: str) -> Path | None:
    # Prevent path traversal. We only accept normalized POSIX paths without '..'.
    pp = PurePosixPath(req_path)
    if any(part == ".." for part in pp.parts):
        return None

    candidate = (root / pp.as_posix()).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None

    return candidate


app = FastAPI()


@app.get("/healthz")
def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.post("/api/fraud/roc", response_model=FraudRocResponse)
def fraud_roc(req: FraudRocRequest) -> dict[str, Any]:
    # Basic guardrails to keep training interactive.
    if req.sample_size > 50000:
        raise HTTPException(status_code=400, detail="sample_size too large")

    fraud_rate_rounded = float(round(req.fraud_rate, 6))

    try:
        return _cached_fraud_roc(req.model_type, fraud_rate_rounded, req.sample_size, req.seed)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/{full_path:path}")
def static_or_spa(full_path: str) -> FileResponse:
    # Let FastAPI handle /api routes (and avoid serving them from disk).
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")

    candidate = _safe_public_join(PUBLIC_DIR, full_path)

    if candidate and candidate.is_file():
        return FileResponse(candidate)

    # Extensionless HTML routes: /foo -> public/foo.html if it exists
    if candidate and candidate.suffix == "":
        html_candidate = candidate.with_suffix(".html")
        if html_candidate.is_file():
            return FileResponse(html_candidate)

    # SPA fallback
    return FileResponse(PUBLIC_DIR / "index.html")
