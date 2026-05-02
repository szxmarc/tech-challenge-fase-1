# Deployment Documentation — Churn Prediction API

> **Version:** 2.0.0  
> **Last updated:** 2026-05-01  
> **API framework:** FastAPI  
> **Model:** Multi-Layer Perceptron — ChurnMLP (PyTorch)

---

## 1. Real-Time Inference Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLIENT LAYER                      │
│        (CRM System / Internal Dashboard)            │
└────────────────────┬────────────────────────────────┘
                     │ HTTP POST /api/v1/predict
                     ▼
┌─────────────────────────────────────────────────────┐
│               FASTAPI APPLICATION                   │
│                                                     │
│  ┌───────────┐   ┌────────────┐   ┌─────────────┐  │
│  │  Pydantic │   │   Route    │   │  Prediction │  │
│  │  Schema   │──▶│  Handler   │──▶│   Service   │  │
│  │ Validation│   │ (routes.py)│   │ (service.py)│  │
│  └───────────┘   └────────────┘   └──────┬──────┘  │
│                                          │          │
│                  ┌───────────────────────┘          │
│                  ▼                                  │
│  ┌──────────────────────────────────────────────┐   │
│  │             INFERENCE PIPELINE               │   │
│  │                                              │   │
│  │  Feature Engineering (charges_per_month etc) │   │
│  │  pd.get_dummies (fixed categories)           │   │
│  │  SimpleImputer.transform()                   │   │
│  │  StandardScaler.transform()                  │   │
│  │  Select TOP_19_FEATURES                      │   │
│  │  ChurnMLP.forward() → sigmoid → probability  │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │           IN-MEMORY ARTIFACT CACHE          │   │
│  │   model.pt + scaler.joblib + imputer.joblib  │   │
│  │   (singletons loaded once at first request)  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                     │ JSON response
                     ▼
         { prediction, predictionLabel,
           probabilityChurn, probabilityNoChurn,
           confidence }
```

---

## 2. Why FastAPI Was Chosen

FastAPI was selected for the following reasons:

| Criterion | FastAPI Advantage |
|-----------|-----------------|
| **Performance** | Built on Starlette (ASGI); throughput competitive with Node.js and Go for I/O-bound workloads |
| **Automatic validation** | Pydantic v2 models validate and coerce request payloads automatically, reducing boilerplate and error surface |
| **Type safety** | Full Python type hint support; models, routes, and services are fully typed |
| **Auto-generated docs** | Swagger UI (`/docs`) and ReDoc (`/redoc`) generated from type annotations with zero extra configuration |
| **Async support** | Native `async/await` support for future enhancements (database calls, external APIs) |
| **Production maturity** | Widely adopted in ML serving (Hugging Face, Netflix, Uber); strong ecosystem |
| **Testing** | `TestClient` from `httpx` enables synchronous integration tests without a running server |

---

## 3. Request Lifecycle

### Single Prediction — `POST /api/v1/predict`

```
1. Client sends JSON payload (19 raw feature fields — original dataset format)
        │
        ▼
2. FastAPI deserializes into CustomerFeatures (Pydantic BaseModel)
   → Literal type validation (e.g. gender: "Male"|"Female")
   → HTTP 422 returned immediately if schema validation fails
        │
        ▼
3. routes.predict() calls predict_single(body.model_dump())
        │
        ▼
4. service._feature_engineering() derives engineered features:
   charges_per_month, is_monthly, service_count,
   has_no_services, is_senior_alone, charges_x_monthly
        │
        ▼
5. service._encode() applies pd.get_dummies with fixed pd.Categorical
   categories to ensure consistent 36-column output for single rows
        │
        ▼
6. dependencies.get_model() returns (model, scaler, imputer) from cache
   (disk load only on first call after startup)
        │
        ▼
7. imputer.transform(X_36) → median imputation for any NaN values
        │
        ▼
8. scaler.transform(X_36) → zero-mean, unit-variance scaled features
   using training distribution
        │
        ▼
9. Select TOP_19_FEATURES by column index from scaled 36-column matrix
        │
        ▼
10. torch.no_grad(): model(X_tensor) → logits
    torch.sigmoid(logits).item() → proba_churn
        │
        ▼
11. _build_prediction_result() formats output dict
        │
        ▼
12. FastAPI serializes to JSON → HTTP 200 OK
```

**Typical response:**
```json
{
  "prediction": 1,
  "predictionLabel": "Churn",
  "probabilityNoChurn": "22.4%",
  "probabilityChurn": "77.6%",
  "confidence": "77.6%"
}
```

### Batch Prediction — `POST /api/v1/predict/batch`

```
Client sends: { "customers": [ {…}, {…}, … ] }
        │
        ▼
BatchPredictRequest validates the list of CustomerFeatures
        │
        ▼
For each customer i:
  predict_single(customer[i]) → result
  result["customerIndex"] = i
  Errors per customer are captured without aborting the batch
        │
        ▼
Response: { "total": N, "predictions": [ result_0, result_1, … ] }
```

---

## 4. Preprocessing Flow

The preprocessing pipeline at inference time mirrors the training pipeline exactly:

```
Raw API input (original dataset field names)
        │
        ▼ _feature_engineering()
DataFrame with 6 derived features added (36 total after encoding)
        │
        ▼ _encode() — pd.get_dummies with fixed pd.Categorical
36-column encoded DataFrame (consistent regardless of input values)
        │
        ▼ imputer.transform(X)
NaN values replaced with training-set medians
        │
        ▼ scaler.transform(X)
Scaled matrix (µ=0, σ=1) using training distribution
        │
        ▼ Select TOP_19_FEATURES by index
19-column matrix matching training feature order
        │
        ▼ torch.tensor → model.forward() → sigmoid
Raw probability → binary prediction
```

**Critical invariants:**
- `pd.Categorical` with fixed categories ensures consistent column output for single-row inference
- Column order must match `TOP_FEATURES` exactly (enforced by index selection)
- The imputer and scaler's statistics come from the training set; never recomputed at inference
- The model runs in `eval()` mode — BatchNorm uses running statistics, Dropout is disabled

---

## 5. Model Loading Strategy

The API uses a **lazy singleton pattern** implemented in `src/api/dependencies.py`:

```python
_model = None
_scaler = None
_imputer = None

def get_model():
    global _model, _scaler, _imputer
    if _model is None:
        _model, _scaler, _imputer = load_model()
    return _model, _scaler, _imputer
```

**Properties of this pattern:**

| Property | Detail |
|----------|--------|
| Load time | Only once per process lifecycle (first request after startup) |
| Memory | Shared across all requests; no per-request allocation |
| Thread safety | Safe for read-only access; writes happen exactly once during initialization |
| Startup | Application starts fast; model load is deferred to first request |

**Artifact paths** (configured in `src/config/settings.py`):
```
models/baseline/mlp/
├── model.pt        ← ChurnMLP state dict (PyTorch)
├── scaler.joblib   ← Fitted StandardScaler
├── imputer.joblib  ← Fitted SimpleImputer
└── config.json     ← Hyperparameters (audit trail)
```

---

## 6. Scalability Considerations

### Current Architecture Characteristics

| Aspect | Characteristic |
|--------|---------------|
| Statefulness | **Stateless** — no session state, all state in model artifacts |
| Concurrency | ASGI (async I/O) handles concurrent connections efficiently |
| CPU per request | Low — MLP inference on 19 features is negligible |
| Memory footprint | ~200 MB for model + scaler + imputer + PyTorch runtime |
| Horizontal scaling | ✅ Trivially scalable — spin up N identical instances behind a load balancer |

### Horizontal Scaling Pattern

```
                    ┌─────────────┐
                    │ Load Balancer│
                    │ (Nginx/ALB) │
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ FastAPI  │ │ FastAPI  │ │ FastAPI  │
        │ Worker 1 │ │ Worker 2 │ │ Worker N │
        └──────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┼────────────┘
                           ▼
                   ┌───────────────┐
                   │ Shared Storage│
                   │ (model files) │
                   └───────────────┘
```

### Performance Targets (SLOs)

| Metric | Target |
|--------|--------|
| P95 latency (single prediction) | < 100 ms |
| Throughput | ≥ 100 predictions/second per instance |
| API uptime | ≥ 99% |

### Bottlenecks and Mitigations

| Potential Bottleneck | Mitigation |
|--------------------|-----------|
| Cold start latency | Pre-warm model at startup with `@app.on_event("startup")` hook |
| Model artifact access | Mount artifacts from shared volume or object storage (S3/GCS) |
| Large batch requests | Implement vectorized batch inference (single forward pass for full batch) |
| Memory pressure at scale | Use multi-process workers (`uvicorn --workers N`) |
| BatchNorm variance at batch=1 | Known ~3-4% probability divergence for borderline cases — acceptable for production |

---

## 7. Batch vs. Real-Time Comparison

| Dimension | Real-Time (current) | Batch Processing |
|-----------|---------------------|-----------------|
| **Use case** | CRM integration, inline scoring | Nightly churn risk refresh for entire customer base |
| **Latency requirement** | < 100 ms per request | Minutes to hours acceptable |
| **Throughput** | 1–100 customers per call | Thousands to millions of records |
| **Infrastructure** | FastAPI + ASGI server | Pandas pipeline on scheduled job |
| **Current implementation** | ✅ `POST /api/v1/predict` and `POST /api/v1/predict/batch` | ⚠️ Not yet implemented as offline pipeline |
| **Recommended trigger** | Customer interaction events | Nightly cron job (00:00 UTC) |

---

## 8. Cloud Deployment Approaches

### Option A: Container on Managed Kubernetes (GKE / EKS / AKS)

```
┌────────────────────────────────────────────┐
│  Docker Container                          │
│  ├── Python 3.11 runtime                  │
│  ├── FastAPI + Uvicorn                     │
│  ├── PyTorch (CPU), scikit-learn, joblib   │
│  └── Model artifacts (baked in or mounted) │
└────────────────────┬───────────────────────┘
                     │
              ┌──────┴──────┐
              │  Kubernetes │
              │  Deployment │
              │  (N replicas)│
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │  Ingress /  │
              │  Load Balancer│
              └─────────────┘
```

**Pros:** Full control, auto-scaling (HPA), rolling deployments, multi-cloud portability  
**Cons:** Higher operational overhead

---

### Option B: Serverless — AWS Lambda + API Gateway

```
Client → API Gateway → Lambda Function → Model inference → Response
                              │
                    Model artifacts in S3
                    (loaded and cached per container reuse)
```

**Pros:** Zero infrastructure management, pay-per-request pricing  
**Cons:** Cold start latency, PyTorch package size may exceed Lambda limits (consider torch-cpu slim build)

---

### Option C: Managed ML Serving — AWS SageMaker / Google Vertex AI

Deploy the PyTorch model using the platform's model registry and serving infrastructure.

```
Training pipeline → Model Registry → Endpoint (managed)
                                           │
                              Auto-scaling, monitoring, A/B
                              testing built in
```

**Pros:** Integrated MLOps tooling, SLA guarantees  
**Cons:** Vendor lock-in, higher per-inference cost

---

### Recommended Path for Phase 2

```
1. Containerize with Docker (multi-stage build, torch-cpu for smaller image)
2. Publish to container registry (ECR/GCR/ACR)
3. Deploy to Kubernetes with:
   - Readiness/liveness probes on /health
   - Horizontal Pod Autoscaler (CPU ≥ 70%)
   - ConfigMap for environment-based model path overrides
   - PersistentVolumeClaim for model artifact storage
4. CI/CD pipeline:
   - On merge to main → build image → run tests → deploy to staging
   - On tag → promote to production with canary rollout
```

---

## 9. Running Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Start the API server
python run_app.py

# Interactive API docs
open http://localhost:5000/docs

# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
  }'
```