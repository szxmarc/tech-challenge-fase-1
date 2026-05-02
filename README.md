# Tech Challenge - Fase 1: Previsão de Churn com MLP PyTorch

Solução end-to-end de Machine Learning para previsão de churn em uma operadora de telecomunicações. O projeto cobre EDA, modelagem com MLP PyTorch, API REST de inferência e documentação completa de MLOps.

## 🎯 Resultado

| Métrica | Meta | Alcançado |
|---------|------|-----------|
| AUC-ROC | ≥ 0.80 | **0.836** ✅ |
| Recall | ≥ 0.75 | **0.781** ✅ |
| F2-Score | ≥ 0.70 | **0.704** ✅ |
| Valor de negócio | > R$0 | **+R$40.316** ✅ |

## 👥 Integrantes

- **Giovanni de Aguirre Tamanini** (RM371630) - giovanni.tnini@gmail.com
- **Integrante 2** - [Nome e contato]
- **Integrante 3** - [Nome e contato]
- **Integrante 4** - [Nome e contato]

---

## 📂 Estrutura do Projeto

```
tech-challenge-fase-1/
├── data/
│   ├── raw/                          # Dataset original (nunca modificar)
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       └── baseline/                 # Dados pré-processados
│           ├── X_train.csv / X_test.csv
│           ├── y_train.csv / y_test.csv
│           └── scaler.joblib
│
├── models/
│   ├── feature_names.txt             # Features do contrato da API
│   └── baseline/
│       ├── logistic_regression/      # Baseline
│       │   ├── model.joblib
│       │   └── metrics.json
│       └── mlp/                      # Modelo de produção
│           ├── model.pt              # ChurnMLP state dict (PyTorch)
│           ├── scaler.joblib
│           ├── imputer.joblib
│           └── config.json
│
├── notebooks/
│   ├── 01_eda_e_baselines.ipynb      # EDA completa + baselines (Dummy + LR)
│   └── telco_churn_mlp.ipynb         # MLP PyTorch + Random Forest + SHAP
│
├── src/
│   ├── api/                          # FastAPI
│   │   ├── app.py
│   │   ├── dependencies.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── config/
│   │   └── settings.py               # Configurações centralizadas
│   ├── data/
│   │   ├── loader.py
│   │   └── processor.py
│   ├── evaluation/
│   │   ├── features.py               # Permutation importance
│   │   └── metrics.py                # Métricas técnicas e de negócio
│   ├── models/
│   │   └── trainer.py                # ChurnMLP, train_mlp, save/load
│   ├── prediction/
│   │   └── service.py                # Pipeline completo de inferência
│   └── utils/
│       └── helpers.py
│
├── tests/                            # 100 testes, 97% cobertura
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── prediction/
│   └── utils/
│
├── docs/
│   ├── ML_Canvas.md
│   ├── model_card.md
│   ├── deployment.md
│   ├── monitoring.md
│   └── star_script.md
│
├── mlruns/                           # Experimentos MLflow
├── reports/                          # Relatórios de cobertura
├── Makefile
├── pyproject.toml
├── .gitignore
└── run_app.py
```

---

## 🛠️ Setup

### Pré-requisitos

- Python 3.11+
- Git

### Instalação

```bash
# 1. Clone o repositório
git clone <URL_DO_REPOSITORIO>
cd tech-challenge-fase-1

# 2. Crie e ative o ambiente virtual
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instale as dependências
pip install -e ".[notebooks]"

# Para desenvolvimento (inclui testes e linting):
pip install -e ".[dev,notebooks]"
```

---

## 🚀 Como Executar

### API de Inferência

```bash
make run
# ou
python run_app.py
```

API disponível em `http://localhost:5000`  
Documentação interativa: `http://localhost:5000/docs`

### Testes

```bash
make test
```

### Linting

```bash
make lint
```

### Notebooks

```bash
jupyter notebook
```

- `01_eda_e_baselines.ipynb` — EDA completa + baselines (DummyClassifier + Regressão Logística)
- `telco_churn_mlp.ipynb` — MLP PyTorch + Random Forest + SHAP + otimização de threshold

### MLflow UI

```bash
mlflow ui --port 5001
```

Acesse `http://localhost:5001` para visualizar experimentos, métricas e artefatos.

---

## 🔌 API — Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Status da API |
| GET | `/api/v1/health` | Status da API + modelo |
| GET | `/api/v1/model/info` | Metadados do modelo |
| GET | `/api/v1/features` | Features usadas pelo modelo |
| POST | `/api/v1/predict` | Predição para um cliente |
| POST | `/api/v1/predict/batch` | Predição para múltiplos clientes |

### Exemplo de Request

```bash
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

### Exemplo de Response

```json
{
  "prediction": 1,
  "predictionLabel": "Churn",
  "probabilityNoChurn": "27.7%",
  "probabilityChurn": "72.3%",
  "confidence": "72.3%"
}
```

---

## 🧠 Modelo

### Arquitetura — ChurnMLP (PyTorch)

```
Input (19 features)
    ↓
Linear(19 → 128) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(128 → 64) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(64 → 32) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(32 → 1) → Sigmoid (na inferência)
```

### Pipeline de Inferência

```
Dados brutos do cliente
    ↓ Feature engineering (charges_per_month, is_monthly, service_count, ...)
    ↓ One-hot encoding (pd.get_dummies, categorias fixas)
    ↓ Imputação com mediana (SimpleImputer)
    ↓ Padronização (StandardScaler)
    ↓ Seleção das 19 TOP_FEATURES
    ↓ ChurnMLP.forward() → sigmoid → probabilidade de churn
```

### Comparação de Modelos

| Modelo | AUC-ROC | F2-Score | Recall | Valor de Negócio |
|--------|---------|----------|--------|-----------------|
| DummyClassifier | 0.516 | 0.291 | 0.291 | −R$ 102.300 |
| Regressão Logística | 0.841 | 0.704 | 0.781 | +R$ 61.500 |
| Random Forest | 0.842 | — | — | — |
| **MLP PyTorch (produção)** | **0.836** | **0.704** | **0.781** | **+R$ 40.316** |

---

## 📊 Dataset

**Telco Customer Churn (IBM)**  
Fonte: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7.043 clientes
- 20 features + 1 target binário (Churn: Yes/No)
- 26.5% churn / 73.5% não-churn

---

## ✅ Qualidade

- Seeds fixados (`random_state=42`) para reprodutibilidade
- Split estratificado 80/20
- Prevenção de data leakage (scaler/imputer fit apenas no treino)
- Early stopping (patience=10) + ReduceLROnPlateau
- 100 testes automatizados, 97% de cobertura
- Linting com ruff (zero erros)
- Experimentos rastreados no MLflow

---

## 📚 Documentação

| Documento | Descrição |
|-----------|-----------|
| `docs/ML_Canvas.md` | Canvas de ML com stakeholders, métricas e SLOs |
| `docs/model_card.md` | Model Card completo (limitações, vieses, monitoramento) |
| `docs/deployment.md` | Arquitetura de deploy e lifecycle de requests |
| `docs/monitoring.md` | Plano de monitoramento (drift, latência, KPIs) |
| `docs/star_script.md` | Script do vídeo STAR (5 minutos) |

---

## 📝 Licença

Projeto desenvolvido como parte do Tech Challenge — Fase 1, FIAP PosTech ML Engineering, 2026.