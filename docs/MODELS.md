# Model Selection Guide

Technical reference for the ML models in this demo.

## Models

### Histogram Gradient Boosting (`hist_gbdt`)

**Best overall performer.** Use when you need maximum discriminative power.

```python
HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.06,
    max_iter=450
)
```

- Handles the non-linear segment boundaries well (fraud vs legit segments have piecewise decision surfaces)
- Native handling of class imbalance via sample weights
- Fast training due to histogram binning
- No feature scaling required

**When to use:** Production fraud detection, when accuracy matters more than interpretability.

---

### Extra Trees (`extra_trees`)

**Strong performer with built-in regularization.** Competitive with boosting on this dataset.

```python
ExtraTreesClassifier(
    n_estimators=180,
    max_depth=6,
    min_samples_leaf=25,
    max_features="sqrt"
)
```

- Extremely randomized splits reduce overfitting
- `min_samples_leaf=25` prevents fitting noise in small fraud clusters
- Parallelizes well (`n_jobs=-1`)
- Provides `feature_importances_`

**When to use:** When you want ensemble strength with less tuning than boosting.

---

### Logistic Regression (`logistic`)

**Linear baseline.** Fast, interpretable, but limited by linearity assumption.

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
])
```

- Requires feature scaling (StandardScaler in pipeline)
- Can't capture the non-linear fraud segments (organized fraud has low ratio but high days—a linear model struggles here)
- Provides coefficients for feature interpretation
- Sub-second training

**When to use:** Baseline comparison, regulatory environments requiring explainability, or when training speed is critical.

---

### Random Forest (`random_forest`)

**Intentionally weak baseline** in this demo.

```python
RandomForestClassifier(
    n_estimators=120,
    max_depth=3,
    min_samples_leaf=60,
    max_features="sqrt"
)
```

- `max_depth=3` severely limits expressiveness
- `min_samples_leaf=60` prevents learning fine-grained patterns
- Demonstrates that a poorly-tuned ensemble underperforms

**When to use:** Showing stakeholders why hyperparameter tuning matters.

---

## Synthetic Data Design

The data has intentional structure to test model capabilities:

| Segment | % of Class | Key Pattern |
|---------|------------|-------------|
| Obvious fraud | 55% of fraud | High ratio, short policy age, many prior claims |
| Organized fraud | 25% of fraud | Moderate ratio, many prior claims, long reporting delay |
| Camouflaged fraud | 20% of fraud | Moderate features, harder to detect |
| Normal legit | 75% of legit | Low ratio, long policy age, few prior claims |
| High-amount legit | 17% of legit | High claim amount but otherwise normal |
| Suspicious legit | 8% of legit | High ratio but very long policy, few prior claims |

**Key discriminators:**
- `policy_age_days`: Fraud clusters around 450-650 days; legit around 900-1400
- `previous_claims`: Fraud has λ=2.2-2.8; legit has λ=0.8-1.1
- `claim_to_premium_ratio`: Fraud centers at 0.70-1.35; legit at 0.25-0.85
- `is_weekend`: Fraud 38-48%; legit 22-28%

The "suspicious-but-legit" segment (8%) creates false positives—these have high ratios but are distinguishable by long policy tenure and few prior claims. Non-linear models capture this; logistic regression cannot.

---

## Class Imbalance Handling

All models use inverse-frequency sample weights:

```python
w_pos = n / (2 * n_pos)
w_neg = n / (2 * n_neg)
```

At 5% fraud rate, frauds get ~10x weight. This shifts the decision boundary toward detecting more fraud at the cost of some false positives.

---

## Feature Normalization

Features are normalized before modeling:

| Feature | Normalization |
|---------|---------------|
| claim_amount | log(amount) / 10.0 |
| policy_age_days | days / 1500.0 |
| previous_claims | count / 4.0 |
| is_weekend | binary (0/1) |
| days_to_report | days / 10.0 |
| claim_to_premium_ratio | ratio / 2.0 |

Tree models don't strictly need this, but it keeps feature scales comparable for logistic regression.

---

## Expected Performance

At 5% fraud rate, 8000 samples:

| Model | Typical AUC | False Alarm Rate* |
|-------|-------------|-------------------|
| hist_gbdt | 0.88-0.92 | 5-12% |
| extra_trees | 0.85-0.90 | 8-15% |
| logistic | 0.78-0.84 | 12-20% |
| random_forest | 0.75-0.82 | 15-25% |

*At business-optimal threshold (max net benefit)

Variance depends on seed. The random forest is intentionally capped to show the cost of under-tuning.
