# Monitoring Plan — Churn Prediction API

> **Version:** 2.0.0  
> **Last updated:** 2026-05-01  
> **Model:** ChurnMLP (PyTorch) — Telco Churn Prediction  
> **Environment:** Production

---

## Overview

This document defines the production monitoring strategy for the churn prediction model and its serving API. The plan covers five monitoring layers: data quality, model drift, latency, errors, and business KPIs. Each layer has defined alert thresholds, ownership, and escalation procedures.

```
┌─────────────────────────────────────────────────────────┐
│                  MONITORING LAYERS                      │
├─────────────────┬───────────────────────────────────────┤
│ Layer 1         │ Data Quality (input distribution)     │
│ Layer 2         │ Model / Concept Drift                 │
│ Layer 3         │ API Latency & Throughput              │
│ Layer 4         │ Error Rate & Availability             │
│ Layer 5         │ Business KPIs (churn reduction ROI)   │
└─────────────────┴───────────────────────────────────────┘
```

---

## 1. Model Drift Monitoring

Model drift occurs when the statistical relationship between input features and the churn outcome changes over time. Two sub-types are monitored:

### 1.1 Data Drift (Covariate Shift)

Input feature distributions shift without necessarily affecting model accuracy immediately.

**Method:** Population Stability Index (PSI) calculated weekly against the training distribution.

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

| PSI Range | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | No significant drift | Continue monitoring |
| 0.10 – 0.20 | Moderate drift | Investigate root cause |
| > 0.20 | Significant drift (**alert**) | Trigger drift investigation; consider retraining |
| > 0.25 | Critical drift (**escalate**) | Emergency retraining procedure |

**Features to prioritize (top 5 by SHAP importance):**

| Feature | Monitoring frequency | PSI alert threshold |
|---------|---------------------|---------------------|
| `charges_x_monthly` | Weekly | 0.20 |
| `tenure` | Weekly | 0.20 |
| `TotalCharges` | Weekly | 0.20 |
| `Contract_Two year` | Weekly | 0.15 |
| `InternetService_Fiber optic` | Weekly | 0.15 |

### 1.2 Concept Drift (Label Shift)

The relationship between features and churn outcome changes (e.g., due to market events, pricing changes, new competitors).

**Detection approach:**
- Compare predicted churn probability distribution (weekly rolling window) against the training baseline
- When labeled data becomes available (30–60 day lag), evaluate AUC-ROC on the fresh cohort
- Use the Kolmogorov-Smirnov (KS) test on prediction score distributions

**Alert thresholds:**

| Metric | Baseline (training) | Warning | Critical |
|--------|--------------------|---------|---------||
| AUC-ROC (labeled cohort) | 0.836 | < 0.815 | < 0.800 |
| Predicted churn rate | ~26.5% | ±10% deviation | ±20% deviation |
| KS statistic (score distribution) | — | > 0.10 | > 0.15 |

---

## 2. Data Quality Checks

Applied to every incoming request and logged for batch analysis.

### 2.1 Per-Request Checks (Inline — Enforced by API)

| Check | Mechanism | Failure action |
|-------|-----------|---------------|
| All 19 raw feature fields present | Pydantic schema required fields | HTTP 422, field list in response |
| Categorical values valid | Pydantic `Literal` type validators | HTTP 422 |
| Numeric types valid (int/float) | Pydantic field types | HTTP 422 |

### 2.2 Batch Statistical Checks (Offline — Daily)

Run against the daily batch of API request logs:

| Check | Method | Alert threshold |
|-------|--------|-----------------|
| Missing value rate per feature | Count nulls / total requests | > 5% missingness in any feature |
| Outlier rate per numeric feature | Values > µ ± 5σ of training distribution | > 2% of daily requests |
| `MonthlyCharges` range | Min/max bounds from training data | Values < R$0 or > R$500 |
| `tenure` range | Min/max bounds (0–72 months in training) | Values < 0 or > 120 |
| Categorical feature integrity | Valid values per Literal schema | Any out-of-vocabulary value |

### 2.3 Recommended Tooling

| Tool | Purpose |
|------|---------|
| **Great Expectations** | Define and run data quality assertion suites |
| **Evidently AI** | Automated data drift and quality reports |
| **Custom logging** | Persist request payloads to structured log store for offline analysis |

---

## 3. Latency Monitoring

### 3.1 Metrics to Track

| Metric | Description | Collection method |
|--------|-------------|------------------|
| P50 latency | Median inference time | Prometheus histogram |
| P95 latency | 95th percentile (SLO target) | Prometheus histogram |
| P99 latency | Tail latency (worst-case) | Prometheus histogram |
| Throughput (RPS) | Requests per second | Prometheus counter |
| Batch size distribution | Number of customers per batch request | Histogram |

### 3.2 Alert Thresholds

| Metric | Warning | Critical | SLO |
|--------|---------|----------|-----|
| P95 latency | > 150 ms | > 300 ms | < 100 ms |
| P99 latency | > 300 ms | > 500 ms | < 200 ms |
| Throughput (RPS) | Drop > 30% from baseline | Drop > 60% from baseline | ≥ 100 RPS |

### 3.3 Latency Breakdown Targets

| Stage | Expected duration |
|-------|------------------|
| Pydantic validation | < 1 ms |
| Feature engineering (`_feature_engineering`) | < 2 ms |
| One-hot encoding (`_encode`) | < 2 ms |
| `imputer.transform()` | < 2 ms |
| `scaler.transform()` | < 2 ms |
| `ChurnMLP.forward()` + sigmoid | < 5 ms |
| JSON serialization | < 1 ms |
| **Total (single prediction)** | **< 20 ms** (model inference only) |
| Network + FastAPI overhead | < 80 ms |
| **End-to-end P95 target** | **< 100 ms** |

### 3.4 Tooling

- **Prometheus + Grafana:** Instrument FastAPI with `prometheus-fastapi-instrumentator`
- **Distributed tracing:** OpenTelemetry + Jaeger for multi-service latency attribution
- **Structured logging:** Log `inference_time_ms` per request for trend analysis

---

## 4. Error Monitoring

### 4.1 Error Categories

| Category | HTTP Status | Description | Alert threshold |
|----------|-------------|-------------|-----------------|
| Validation errors | 422 | Missing fields, invalid categorical values | > 5% of hourly requests |
| Inference errors | 400 | Feature conversion or type errors | > 2% of hourly requests |
| Model load failure | 503 | `model.pt`, `scaler.joblib` or `imputer.joblib` not found | **Any occurrence** |
| Unhandled exceptions | 500 | Unexpected inference errors | > 1% of hourly requests |
| Not found | 404 | Unknown endpoint called | > 10% of hourly requests |
| Timeouts | 504 | Gateway timeout (upstream proxy) | Any occurrence |

### 4.2 Key Error Signals

- **HTTP 503 from `/health`:** Model artifact unavailable — PagerDuty alert (P1)
- **HTTP 422 spike:** Client integration bug or schema change — notify API consumers
- **HTTP 500 spike:** Regression in inference code — rollback deployment

### 4.3 Error Logging Schema

Every error response should be logged with:

```json
{
  "timestamp": "2026-05-01T14:30:00Z",
  "endpoint": "/api/v1/predict",
  "http_status": 422,
  "error_type": "validation_error",
  "error_message": "Input should be 'Male' or 'Female'",
  "request_id": "uuid-...",
  "client_ip": "10.0.0.1"
}
```

### 4.4 Recommended Tooling

- **Sentry:** Exception tracking with stack traces and context
- **ELK Stack / OpenSearch:** Centralized log aggregation and search
- **PagerDuty / OpsGenie:** On-call alert routing

---

## 5. Business KPI Monitoring

Business-level monitoring closes the feedback loop between model predictions and real-world outcomes.

### 5.1 KPIs to Track

| KPI | Description | Target | Measurement frequency |
|-----|-------------|--------|----------------------|
| **Churn rate (all customers)** | % of customers who cancelled in the period | Reduce by 15% quarterly | Monthly |
| **Precision-in-practice** | % of flagged customers who actually churned | ≥ 50% | Monthly (requires 30-60d label lag) |
| **Recall-in-practice** | % of actual churners that were flagged | ≥ 75% | Monthly |
| **Retention campaign conversion rate** | % of contacted at-risk customers who stayed | ≥ 30% | Monthly |
| **Net business value** | (TP × R$400) − (FP × R$50) − (FN × R$500) | Positive (≥ R$0) | Monthly |
| **ROI of ML system** | Net business value / total infrastructure cost | > 3× | Quarterly |
| **Coverage** | % of churning customers that were identified | ≥ 75% | Monthly |

### 5.2 Business Alert Thresholds

| Signal | Warning | Critical |
|--------|---------|----------|
| Net business value | < R$20,000/month | < R$0 (negative ROI) |
| Retention conversion rate | < 20% | < 10% |
| Actual churn rate | Increases > 5% month-over-month | Increases > 15% month-over-month |
| Precision-in-practice | < 40% | < 25% |

---

## 6. Alert Thresholds Summary

| Signal | Warning | Critical | Owner |
|--------|---------|----------|-------|
| PSI any key feature | > 0.15 | > 0.25 | ML Engineer |
| AUC-ROC (labeled cohort) | < 0.815 | < 0.800 | ML Engineer |
| Predicted churn rate shift | ±10% | ±20% | ML Engineer |
| P95 API latency | > 150 ms | > 300 ms | Platform Engineer |
| Error rate (4xx/5xx) | > 5% | > 10% | Backend Engineer |
| API unavailability | — | Any 503 from /health | On-call |
| Net business value | < R$20k/month | < R$0 | Product Manager |

---

## 7. Incident Response Process

### Severity Levels

| Severity | Definition | Response time |
|----------|-----------|---------------|
| **P1 — Critical** | API down or model not loading; business value negative | 30 minutes (24/7) |
| **P2 — High** | Significant drift detected; error rate > 10% | 2 hours (business hours) |
| **P3 — Medium** | Warning thresholds breached; latency degraded | 1 business day |
| **P4 — Low** | Informational alerts; no immediate impact | Next sprint |

### P1 Response Runbook

```
1. Alert fires (PagerDuty) → On-call engineer paged
2. Check /health endpoint:
   - HTTP 200: Model loaded, investigate error logs
   - HTTP 503: Model artifact missing
     a. Verify artifact paths in DATA_PATHS (settings.py)
     b. Restore model.pt, scaler.joblib and imputer.joblib from backup
     c. Restart application container
3. Verify resolution: /health → HTTP 200
4. Write incident report within 24 hours
5. Root cause analysis within 5 business days
```

### P2 Response Runbook — Drift Detected

```
1. Alert fires → ML Engineer notified
2. Run data quality report (Great Expectations / Evidently)
3. Identify affected features and magnitude of drift
4. Assess impact on business KPIs
5. Decision: Can existing model still meet SLOs?
   - YES: Adjust monitoring thresholds, schedule retraining
   - NO: Trigger emergency retraining (see Section 8)
6. Document in incident log
```

---

## 8. Retraining Triggers

### Automatic Retraining Triggers

| Trigger | Condition | Priority |
|---------|-----------|----------|
| Time-based | 3 months since last retraining | Scheduled |
| AUC-ROC degradation | AUC-ROC drops below 0.80 on labeled cohort | High |
| PSI threshold | PSI > 0.25 on 2+ critical features simultaneously | High |
| Business value | Net monthly business value turns negative | Critical |
| Actual churn rate shift | Baseline churn rate changes by > 5 percentage points | High |

### Retraining Process

```
┌─────────────────────────────────────────────────────┐
│              RETRAINING PIPELINE                    │
│                                                     │
│  1. Collect new labeled data (≥ 2,000 records)     │
│     └─ Include recent production data               │
│                                                     │
│  2. Run data validation (Great Expectations)        │
│     └─ Fail fast on quality issues                  │
│                                                     │
│  3. Retrain model (src/models/trainer.py)           │
│     └─ Update settings.py with new data paths       │
│                                                     │
│  4. Evaluate on held-out test set                   │
│     └─ All 4 SLO metrics must be met:              │
│        AUC-ROC ≥ 0.80, Recall ≥ 0.75              │
│        F2-Score ≥ 0.70, PR-AUC ≥ 0.65             │
│                                                     │
│  5. Log to MLflow (new experiment run)              │
│     └─ Compare vs. current production model         │
│                                                     │
│  6. Challenger A/B test (optional but recommended)  │
│     └─ Route 10% traffic to challenger for 2 weeks  │
│                                                     │
│  7. Promote to production if challenger wins        │
│     └─ Archive current model.pt artifacts           │
│                                                     │
│  8. Update Model Card with new metrics and date     │
└─────────────────────────────────────────────────────┘
```

### Rollback Procedure

If a newly deployed model shows degraded metrics after promotion:

```
1. Restore previous model.pt, scaler.joblib and imputer.joblib from archive
2. Restart API instances (zero-downtime rolling restart)
3. Verify /health returns HTTP 200
4. Investigate new model's failure mode
5. Log rollback event in MLflow and incident tracker
```

---

## 9. Dashboard Recommendations

### Operational Dashboard (Grafana)

Panels to include:
- Real-time request rate (RPS)
- P50 / P95 / P99 latency time series
- Error rate by HTTP status code
- Prediction distribution over time (% churn predicted)
- Health check status

### Model Performance Dashboard

Panels to include:
- PSI per feature (heatmap, updated weekly)
- Predicted vs. actual churn rate (monthly, 30-day label lag)
- AUC-ROC trend over time
- Business value (R$) trend per month

### Business Dashboard (for stakeholders)

Panels to include:
- Quarterly churn rate trend vs. target (-15%)
- Retention campaign conversion rate
- Net business value (R$) generated by ML system
- Coverage: % of churners identified by model

---

*This monitoring plan is a living document. Review and update after each model retraining or significant infrastructure change.*