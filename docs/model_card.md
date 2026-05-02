# Model Card — Telecom Churn Prediction

> **Version:** 2.0.0  
> **Last updated:** 2026-05-01  
> **Model type:** Multi-Layer Perceptron (PyTorch)  
> **Stage:** Production

---

## 1. Business Objective

A telecommunications operator is experiencing accelerating customer churn, directly impacting recurring revenue and customer acquisition costs. The business goal is to **reduce quarterly churn by 15%** by proactively identifying at-risk customers and enabling targeted retention actions before cancellation occurs.

**Financial rationale:**

| Decision | Financial Impact |
|----------|-----------------|
| True Positive (correctly flagged churner, retained) | +R$ 400 net |
| False Positive (flagged, but customer wouldn't have churned) | −R$ 50 (wasted retention action) |
| False Negative (missed churner, lost to churn) | −R$ 500 (lost lifetime value) |
| True Negative (correctly identified non-churner) | R$ 0 |

False Negatives are **10× more costly** than False Positives. The model must therefore prioritize **Recall** as its primary optimization objective.

---

## 2. Model Overview

| Property | Value |
|----------|-------|
| Algorithm | Multi-Layer Perceptron (MLP) |
| Framework | PyTorch 2.x |
| Architecture | Linear → BatchNorm1d → ReLU → Dropout × 3 hidden layers |
| Hidden layers | [128, 64, 32] neurons |
| Dropout | 0.3 |
| Loss function | BCEWithLogitsLoss with pos_weight (handles class imbalance) |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping | patience=10 on validation loss |
| Max epochs | 80 |
| Batch size | 256 |
| Random state | 42 (reproducible) |
| Persistence format | `model.pt` (PyTorch state dict) |
| Preprocessing artifacts | `scaler.joblib` (StandardScaler), `imputer.joblib` (SimpleImputer) |

---

## 3. Dataset Description

| Property | Value |
|----------|-------|
| Dataset name | Telco Customer Churn (IBM) |
| Source | [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Total observations | 7,043 customers |
| Raw features | 20 independent variables + 1 binary target |
| Target variable | `Churn` (Yes / No → 1 / 0) |
| Class distribution | 26.5% Churn (positive) / 73.5% No Churn (negative) |
| Train split | 80% (≈ 5,634 samples), stratified |
| Test split | 20% (≈ 1,409 samples), stratified |

**Variable categories:**

- **Demographic:** gender, SeniorCitizen, Partner, Dependents
- **Service subscriptions:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Contractual:** tenure, Contract type (month-to-month / one-year / two-year)
- **Financial:** MonthlyCharges, TotalCharges, PaymentMethod, PaperlessBilling

---

## 4. Features Used

The model uses **19 features** selected via Random Forest importance + SHAP analysis from an initial pool of 36 engineered features. The API receives raw customer data; all feature engineering is applied at inference time.

**Engineered features (derived from raw inputs):**

| Feature | Description |
|---------|-------------|
| `charges_x_monthly` | MonthlyCharges × is_monthly (interaction term) |
| `charges_per_month` | TotalCharges / tenure (average monthly spend) |
| `is_monthly` | 1 if Contract == "Month-to-month" |
| `service_count` | Number of active add-on services |

**Top predictive features** (SHAP analysis):

1. `charges_x_monthly` — high monthly spend on month-to-month contracts is the strongest churn signal
2. `tenure` — longer tenure strongly predicts retention
3. `TotalCharges` — total spend proxy for customer lifetime
4. `Contract_Two year` — two-year contracts are the strongest retention signal
5. `InternetService_Fiber optic` — fiber optic customers churn at higher rates

---

## 5. Preprocessing Pipeline

All preprocessing steps are applied in training order and reproduced identically at inference time:

1. **Feature engineering:** Derive `charges_per_month`, `is_monthly`, `service_count`, `has_no_services`, `is_senior_alone`, `charges_x_monthly`
2. **One-hot encoding:** `pd.get_dummies` with `drop_first=True` on all categorical columns, using `pd.Categorical` with fixed categories to ensure consistent columns for single-row inference
3. **Missing value imputation:** `SimpleImputer(strategy="median")` — fit on training set only, persisted as `imputer.joblib`
4. **Feature scaling:** `StandardScaler` applied to all 36 features — fit on training set only, persisted as `scaler.joblib`
5. **Feature selection:** Top 19 features selected by index from the scaled 36-column matrix

> **Data leakage prevention:** Both the imputer and scaler are fit exclusively on `X_train`. All production inference uses `.transform()` only, never `.fit_transform()`.

---

## 6. Training Process

```
Dataset (7,043 customers)
       │
       ▼ Stratified 80/20 split (random_state=42)
┌──────────────┐       ┌──────────────┐
│  Train set   │       │   Test set   │
│ ~5,634 rows  │       │ ~1,409 rows  │
└──────┬───────┘       └──────────────┘
       │
       ▼ Feature engineering + get_dummies
       │
       ▼ SimpleImputer.fit_transform()
       │
       ▼ StandardScaler.fit_transform()
       │
       ▼ Select TOP_19_FEATURES
       │
       ▼ ChurnMLP(128→64→32→1).fit()
         BCEWithLogitsLoss + pos_weight
         Adam + ReduceLROnPlateau
         Early stopping (patience=10)
       │
       ▼ Persist: model.pt + scaler.joblib + imputer.joblib
```

**MLflow experiment tracking:**
- Experiment: `telco-churn`
- Run name: `mlp-churn`
- Logged: hyperparameters, train_loss, val_loss, val_auc per epoch, test_auc_roc, test_recall, test_f2, best_threshold, best_business_value

---

## 7. Evaluation Metrics

Results on the held-out **test set (20%)**, never seen during training:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| AUC-ROC | **0.836** | ≥ 0.80 | ✅ Achieved |
| PR-AUC | **0.641** | ≥ 0.65 | ⚠️ Near target |
| Recall | **0.781** | ≥ 0.75 | ✅ Achieved |
| F2-Score | **0.704** | ≥ 0.70 | ✅ Achieved |

**Business value on test set (threshold = 0.5):**

```
Business Value = (TP × +R$400) + (FP × -R$50) + (FN × -R$500)
Result: +R$ 40,316 net profit (R$ 28.61 per customer)
```

**Threshold strategies explored:**

| Strategy | Threshold | Business Value |
|----------|-----------|---------------|
| Default | 0.50 | +R$ 40,316 |
| Business-optimized | 0.44 | +R$ 41,588 |
| P80 (top 20% risk) | 0.76 | +R$ 28,020 |

---

## 8. Baseline Comparison

| Model | AUC-ROC | F2-Score | Recall | Business Value |
|-------|---------|----------|--------|----------------|
| DummyClassifier (Stratified) | 0.516 | 0.291 | 0.291 | −R$ 102,300 |
| Logistic Regression | 0.841 | 0.704 | 0.781 | +R$ 61,500 |
| Random Forest | 0.842 | — | — | — |
| **MLP PyTorch (production)** | **0.836** | **0.704** | **0.781** | **+R$ 40,316** |

> The MLP was selected as the production model per the project requirements (PyTorch neural network). The Logistic Regression baseline achieved marginally higher AUC-ROC but lacks the capacity to model non-linear feature interactions required for future phases.

---

## 9. Limitations

1. **Static dataset:** The model was trained on a historical snapshot and does not adapt to evolving customer behavior without retraining.
2. **BatchNorm inference variance:** BatchNorm layers behave differently with batch size = 1 (single-customer inference) vs. large batches. This introduces a small (~3-4%) probability divergence for borderline cases near the 0.5 threshold.
3. **Threshold fixed at 0.5:** The default decision threshold may not be optimal for all operational scenarios. Business-optimized thresholds are documented above but not yet exposed via API configuration.
4. **No temporal modeling:** The model treats each record as an i.i.d. sample and does not capture time-series patterns in customer behavior.
5. **Dataset scope:** The IBM Telco dataset is a benchmark dataset. Real-world performance may differ due to distribution shift or unmeasured behavioral signals.
6. **Missing external signals:** The dataset lacks competitor offers, network outages, or support ticket frequency — known industry predictors of churn.

---

## 10. Biases and Fairness Considerations

| Protected Attribute | Presence in Dataset | Handling |
|--------------------|---------------------|---------|
| `gender` | ✅ Present as a feature | Included — no statistically significant differential impact observed |
| `isSeniorCitizen` | ✅ Present as a feature | Included — senior citizens show higher churn rates; must be monitored |
| `hasPartner` / `hasDependents` | ✅ Present | Included — proxy for household income |

**Known risks:**
- **Senior citizen bias:** Senior customers churn at a higher rate. Aggressive retention campaigns targeting this group exclusively could create age-based discriminatory treatment.
- **Socioeconomic proxy features:** `monthlyCharges` and `paymentElectronicCheck` may act as proxies for financial status.

**Mitigation recommendations:**
- Conduct fairness audits quarterly using demographic parity and equalized odds metrics.
- Do not use churn score as the sole decision variable for service denial or pricing changes.
- Ensure LGPD compliance: obtain explicit consent for data processing used in retention decisions.

---

## 11. Failure Scenarios

| Scenario | Impact | Mitigation |
|----------|--------|-----------|
| `model.pt` not found at startup | API returns HTTP 503 | Health check endpoint monitors artifact availability |
| Corrupted `scaler.joblib` or `imputer.joblib` | Silent incorrect predictions | Checksum validation at load time (recommended enhancement) |
| Input feature missing from request | HTTP 422 with descriptive error | Validated by Pydantic schema |
| Input contains invalid categorical value | HTTP 422 with field error | Literal type validation in Pydantic schema |
| Data drift causes degraded predictions | Correct format, incorrect predictions (silent failure) | Drift monitoring (see `docs/monitoring.md`) |
| Sudden spike in request volume | Latency degradation | Stateless design enables horizontal scaling |

---

## 12. Ethical Considerations

1. **Transparency:** The model's predictions are used to initiate customer outreach — customers are not denied service based on model output alone.
2. **Human oversight:** Retention team agents review predictions before contacting customers. The model provides a risk score, not an automated decision.
3. **Explainability:** SHAP feature importance is documented and auditable. Top 5 features are monitored for drift.
4. **LGPD Compliance:** All data processing must adhere to Brazil's Lei Geral de Proteção de Dados. Customer data used for inference must be covered by valid consent.
5. **Audit trail:** MLflow experiment tracking provides full lineage of model versions, hyperparameters, and evaluation results.

---

## 13. Monitoring Recommendations

| Signal | Metric | Alert Threshold | Frequency |
|--------|--------|-----------------|-----------|
| Prediction distribution | % predicted churn | Deviation > ±10% from baseline | Daily |
| Feature distributions | Population Stability Index (PSI) | PSI > 0.2 per feature | Weekly |
| API performance | P95 latency | > 200 ms | Real-time |
| Model performance | AUC-ROC on labeled production data | Drop > 5% from baseline (0.836) | Monthly |
| Business KPIs | Actual churn rate among flagged customers | Below 15% retention target | Monthly |

See `docs/monitoring.md` for a full monitoring and alerting strategy.

---

## 14. Retraining Recommendations

**Trigger conditions (any one is sufficient):**

- AUC-ROC on a monthly labeled validation batch drops below **0.80**
- PSI exceeds **0.25** for more than 2 critical features
- Business churn rate reduction falls below the **15% quarterly target**
- More than **3 months** have elapsed since last retraining

**Retraining process:**
1. Collect and label new production data (minimum 2,000 new records)
2. Retrain using `src/models/trainer.py` with updated `settings.py` paths
3. Evaluate against all 4 SLO metrics on held-out test split
4. Log new run to MLflow with full hyperparameter and metric history
5. Deploy only if **all** performance targets are met or exceeded
6. Archive the previous `model.pt` artifact before promotion

---

*This model card was authored in conformance with the Google Model Cards framework and Mitchell et al. (2019). It should be updated at each model version promotion.*