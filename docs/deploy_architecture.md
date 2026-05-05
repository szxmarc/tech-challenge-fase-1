# Arquitetura de Deploy — Churn Prediction API

## 1. Visão Geral

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENTE                              │
│          (curl / Postman / Sistema CRM / Front-end)         │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/REST (porta 8000)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI + Uvicorn                         │
│                                                             │
│  GET  /health                GET  /api/v1/features          │
│  GET  /api/v1/health         POST /api/v1/predict           │
│  GET  /api/v1/model/info     POST /api/v1/predict/batch     │
│  GET  /api/v1/model/comparison                              │
│                                                             │
│  ┌───────────────┐   ┌──────────────────────────────────┐   │
│  │ Prediction    │   │ Startup Pipeline (lifespan)      │   │
│  │ Service       │   │                                  │   │
│  │               │   │ load_raw_data()                  │   │
│  │ predict_      │   │ encode_features()                │   │
│  │ single()      │   │ engineer_features()              │   │
│  │ predict_      │   │ select_features() [RF ≥90%]      │   │
│  │ batch()       │   │ train_logistic_regression()      │   │
│  └───────┬───────┘   │ train_mlp()                      │   │
│          │           │ compare_models()                 │   │
│          ▼           │ mlflow.log_*()                   │   │
│  ┌───────────────┐   └──────────────────────────────────┘   │
│  │ MLP Model     │                                          │
│  │ (joblib)      │                                          │
│  │ + Scaler      │                                          │
│  └───────────────┘                                          │
└─────────────────────────┬───────────────────────────────────┘
                          │ mlflow logging (porta 5000)
                          ▼
             ┌────────────────────────┐
             │  MLflow Tracking Server │
             │  (mlruns/ local)        │
             └────────────────────────┘
```

---

## 2. Pipeline de Startup

A cada inicialização do servidor, o `lifespan` do FastAPI executa `_train_on_startup()`.
Isso garante que os modelos em memória e em disco estejam sempre atualizados com os dados disponíveis.

| Passo | Ação | Saída |
|---|---|---|
| 1 | `load_raw_data()` | DataFrame bruto (7.043 linhas) |
| 2 | `encode_features()` | LabelEncoder + get_dummies |
| 3 | `engineer_features()` | +6 features derivadas |
| 4 | `select_features()` | N features com ≥ 90% importância (RF) |
| 5 | `split_and_scale_data()` | X_train/test escalados + scaler |
| 6 | `train_logistic_regression()` | modelo LR salvo em `models/baseline/` |
| 7 | `train_mlp()` | modelo MLP salvo em `models/mlp/` |
| 8 | `compare_models()` | `reports/model_comparison.json` |
| 9 | MLflow logging | parâmetros + métricas + artefatos em `mlruns/` |
| 10 | `_save_selected_api_names()` | `models/feature_names.txt` (contrato da API) |

**Trade-off:** treinar no startup garante reprodutibilidade mas aumenta o tempo de cold start
(~30-60 segundos dependendo do hardware). Em produção real, considerar carregar artefatos
pré-treinados ao invés de re-treinar.

---

## 3. Endpoints

| Método | Caminho | Descrição | Latência esperada |
|---|---|---|---|
| GET | `/health` | Saúde básica da aplicação | < 5ms |
| GET | `/api/v1/health` | Saúde + disponibilidade do modelo | < 20ms |
| GET | `/api/v1/model/info` | Tipo, nº features, classes | < 20ms |
| GET | `/api/v1/features` | Lista de features camelCase | < 10ms |
| GET | `/api/v1/model/comparison` | JSON com LR vs MLP completo | < 10ms |
| POST | `/api/v1/predict` | Predição individual | < 50ms |
| POST | `/api/v1/predict/batch` | Predição em lote | < 100ms (100 registros) |

---

## 4. Executar Localmente

### Com Make (recomendado)

```bash
# Instalar dependências
make install

# Iniciar a API
make run
```

### Sem Make

```bash
pip install -e ".[dev]"
python run_app.py
```

### Verificar funcionamento

```bash
# Health check
curl http://localhost:8000/health

# Listar features esperadas
curl http://localhost:8000/api/v1/features

# Swagger UI (navegador)
open http://localhost:8000/docs
```

---

## 5. Deploy com Docker

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Copiar arquivos de dependências primeiro (cache de camadas)
COPY pyproject.toml ./

# Instalar dependências sem extras de dev/notebooks
RUN pip install --no-cache-dir -e "."

# Copiar código e dados
COPY src/ ./src/
COPY data/ ./data/
COPY run_app.py ./

# Criar diretórios necessários em runtime
RUN mkdir -p models/baseline/logistic_regression models/mlp reports

EXPOSE 8000

CMD ["python", "run_app.py"]
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models        # persiste modelos treinados entre restarts
      - ./reports:/app/reports      # persiste model_comparison.json
      - ./mlruns:/app/mlruns        # persiste experimentos MLflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlruns
```

### Comandos Docker

```bash
# Build e start
docker compose up --build

# Apenas start (imagem já construída)
docker compose up

# Background
docker compose up -d

# Parar
docker compose down
```

---

## 6. Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `mlruns/` (local) | URI do servidor MLflow |
| `PORT` | `8000` | Porta de exposição da API (se customizado no `run_app.py`) |

---

## 7. Conformidade com SLOs

| SLO | Meta | Status atual |
|---|---|---|
| Latência de predição (P95) | < 100ms | ✅ Estimado < 50ms para predição individual |
| Throughput | ≥ 100 pred/s | ⚠️ Não testado com carga; uvicorn single-worker pode ser gargalo |
| Uptime da API | ≥ 99% | ⚠️ Requer orquestrador (Kubernetes, ECS) para restart automático |
| Data freshness | Diária | ⚠️ Pipeline manual; falta automação via cron ou CI/CD |
| Data drift | Semanal | ⚠️ Não implementado; ver `docs/monitoring_plan.md` |
| Retreinamento | Mensal ou queda de 5% | ⚠️ Manual; sem automação |

---

## 8. Gaps Conhecidos

Os itens abaixo não estão implementados no escopo atual do Tech Challenge:

- **CI/CD pipeline:** sem GitHub Actions ou equivalente para build/test/deploy automático
- **Escalonamento horizontal:** uvicorn em processo único; sem load balancer ou réplicas
- **Modelo pré-treinado no startup:** atualmente re-treina a cada restart; produção real deveria carregar artefato persistido
- **Monitoramento de latência:** sem coleta de métricas de P95/P99 por endpoint
- **Rollback de modelo:** sem estratégia de versionamento com rollback automático
- **Autenticação:** API sem autenticação; qualquer cliente pode consumir os endpoints

---

**Documento criado por:** Giovanni de Aguirre Tamanini  
**Data:** Maio 2026  
**Versão:** 1.0
