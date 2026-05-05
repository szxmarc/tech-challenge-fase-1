# Tech Challenge - Fase 1: Previsão de Churn com Machine Learning

## Descrição do Projeto

Solução end-to-end de Machine Learning para previsão de churn em uma operadora de telecomunicações.
O projeto segue boas práticas de ML Engineering, com código modularizado, testes automatizados,
API REST, rastreamento de experimentos e documentação completa.

## 🎯 Objetivo de Negócio

Uma operadora de telecomunicações enfrenta alta taxa de cancelamento de clientes (churn).
Este projeto constrói um modelo preditivo para classificar clientes em risco de cancelamento,
viabilizando ações preventivas de retenção.

**Métricas de Sucesso:**
- **Técnicas**: AUC-ROC ≥ 0.80, Recall ≥ 0.75, F2-Score ≥ 0.70
- **Negócio**: Valor positivo de retenção (break-even ou lucro)

---

## Integrantes do Grupo

- **Giovanni de Aguirre Tamanini** (RM371630) - 
- **Cristiano Lima do Sacramento** (RM3709482) -
- **Marcelo Santos Souza** (RM370295) - 
- **Yan Levi Martins Meira** (RM370673)

---

## 📂 Estrutura do Projeto

```
tech-challenge-fase-1/
├── data/
│   ├── raw/                        # Dataset original (nunca modificar)
│   │   └── Telco-Customer-Churn.csv
│   ├── processed/
│   │   └── baseline/               # Dados processados (gerados no startup)
│   └── samples/                    # Payloads JSON de exemplo
├── models/
│   ├── baseline/
│   │   ├── dummy_classifier/
│   │   └── logistic_regression/    # Modelo baseline (model.joblib, scaler.joblib, metrics.json)
│   ├── mlp/                        # Modelo de produção MLP PyTorch
│   └── feature_names.txt           # Contrato de features da API (gerado no startup)
├── notebooks/
│   └── 01_eda_e_baselines.ipynb
├── src/
│   ├── api/                        # FastAPI: app.py, routes.py, schemas.py, dependencies.py
│   ├── config/                     # settings.py — fonte única de verdade para configurações
│   ├── data/                       # loader.py, processor.py
│   ├── models/                     # trainer.py, mlp_torch.py
│   ├── evaluation/                 # metrics.py, features.py
│   ├── prediction/                 # service.py, feature_mapping.py
│   ├── utils/                      # helpers.py
│   └── visualization/              # plots.py
├── tests/                          # Suíte pytest (cobertura ≥ 90%)
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── prediction/
│   └── utils/
├── docs/
│   ├── ml_canvas.md
│   ├── model_card.md
│   ├── deploy_architecture.md
│   └── monitoring_plan.md
├── reports/
│   └── model_comparison.json       # Gerado no startup: LR vs MLP
├── Makefile
├── pyproject.toml
├── run_app.py
└── .gitignore
```

---

## 🚀 Etapas de Desenvolvimento

### ✅ Etapa 1 - Entendimento e Preparação (CONCLUÍDA)

- ML Canvas completo (stakeholders, métricas de negócio, SLOs)
- EDA completa: volume, qualidade, distribuição, data readiness
- Métricas técnicas definidas: AUC-ROC, PR-AUC, F1, F2
- Baselines treinados: DummyClassifier e Regressão Logística
- Experimentos registrados no MLflow

**Resultados dos Baselines:**

| Modelo | AUC-ROC | F2-Score | Valor de Negócio |
|--------|---------|----------|------------------|
| DummyClassifier | 0.516 | 0.291 | -R$ 102.300 |
| **Regressão Logística** | **0.841** | **0.704** | **+R$ 61.500** ✅ |

---

### ✅ Etapa 2 - Modelagem com Redes Neurais (CONCLUÍDA)

- MLP implementado em PyTorch com arquitetura (64→32), dropout=0.3
- Loop de treinamento com early stopping (patience=10)
- `pos_weight` no BCEWithLogitsLoss para tratamento do desbalanceamento de classes
- Comparação MLP vs. Regressão Logística com valor de negócio e métricas técnicas
- Todos os experimentos registrados no MLflow

---

### ✅ Etapa 3 - Engenharia e API (CONCLUÍDA)

- Código refatorado em módulos (`src/`) com responsabilidades separadas
- Pipeline reprodutível executado automaticamente no startup da API
- Suíte de testes com pytest (cobertura configurada ≥ 90%)
- API FastAPI com 6 endpoints: `/health`, `/model/info`, `/features`,
  `/predict`, `/predict/batch`, `/model/comparison`
- Logging estruturado com módulo `logging` do Python
- `pyproject.toml` com dependências, ruff e pytest configurados
- `Makefile` com targets: `install`, `run`, `test`, `lint`, `clean`, `mlflow`

---

### ✅ Etapa 4 - Documentação e Entrega Final (CONCLUÍDA)

- Model Card completo (`docs/model_card.md`)
- Arquitetura de deploy documentada (`docs/deploy_architecture.md`)
- Plano de monitoramento (`docs/monitoring_plan.md`)
- README atualizado
- Vídeo STAR (5 minutos) — a ser gravado pelo grupo

---

## 🛠️ Setup e Instalação

### Pré-requisitos

- Python 3.11 ou superior
- Git
- Make (Windows: via [chocolatey](https://chocolatey.org/packages/make) ou Git Bash)

### Instalação

```bash
git clone [URL_DO_REPOSITORIO]
cd tech-challenge-fase-1

# Instala dependências (produção + dev)
make install

# Ou diretamente com pip:
pip install -e ".[dev]"
```

Para notebooks (matplotlib, evidently, jupyter, etc.):

```bash
make install-notebooks
```

---

## 🎓 Como Executar

### Iniciar a API

```bash
make run
# Ou: python run_app.py
```

A API estará disponível em **http://localhost:8000**  
Documentação interativa (Swagger): **http://localhost:8000/docs**

> **Nota:** no primeiro start a API executa o pipeline completo de treinamento
> (carrega dados → engenharia de features → treina LR + MLP → compara → loga no MLflow).
> Aguarde até ver `Modelos treinados e salvos com sucesso.` nos logs.

### Executar Testes

```bash
make test
```

Relatório HTML de cobertura gerado em `reports/coverage/index.html`.

### Verificar Qualidade do Código

```bash
make lint       # apenas verifica
make lint-fix   # corrige automaticamente
```

### Rastreamento de Experimentos (MLflow)

```bash
make mlflow
# Acesse: http://localhost:5000
```

---

## 🌐 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Saúde básica da aplicação |
| GET | `/api/v1/health` | Saúde da API + disponibilidade do modelo |
| GET | `/api/v1/model/info` | Tipo do modelo, nº de features e classes |
| GET | `/api/v1/features` | Lista de features esperadas (camelCase) |
| GET | `/api/v1/model/comparison` | Comparativo LR vs MLP em JSON |
| POST | `/api/v1/predict` | Predição para um único cliente |
| POST | `/api/v1/predict/batch` | Predição para múltiplos clientes |

**Exemplo de predição (curl):**

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "male",
    "isSeniorCitizen": 0,
    "hasPartner": 1,
    "hasDependents": 0,
    "tenureMonths": 24,
    "hasPhoneService": 1,
    "hasPaperlessBilling": 1,
    "monthlyCharges": 75.5,
    "totalCharges": 1812.0,
    "multipleLinesNoPhone": 0,
    "multipleLinesActive": 1,
    "internetFiberOptic": 1,
    "internetNone": 0,
    "onlineSecurityNoInternet": 0,
    "onlineSecurityActive": 0,
    "onlineBackupNoInternet": 0,
    "onlineBackupActive": 1,
    "deviceProtectionNoInternet": 0,
    "deviceProtectionActive": 0,
    "techSupportNoInternet": 0,
    "techSupportActive": 0,
    "streamingTvNoInternet": 0,
    "streamingTvActive": 1,
    "streamingMoviesNoInternet": 0,
    "streamingMoviesActive": 1,
    "contractOneYear": 0,
    "contractTwoYear": 0,
    "paymentCreditCardAutomatic": 0,
    "paymentElectronicCheck": 1,
    "paymentMailedCheck": 0
  }'
```

---

## 📊 Dataset

**Dataset:** Telco Customer Churn (IBM)  
**Fonte:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Característica | Valor |
|---|---|
| Observações | 7.043 clientes |
| Features | 20 variáveis independentes + 1 target |
| Tipo | Classificação binária (Churn: Yes/No) |
| Balanceamento | 26.5% churn, 73.5% não-churn |

**Variáveis Principais:**
- **Demográficas:** gender, SeniorCitizen, Partner, Dependents
- **Serviços:** PhoneService, InternetService, OnlineSecurity, StreamingTV, etc.
- **Contratuais:** tenure, Contract, MonthlyCharges, TotalCharges
- **Target:** Churn (Yes/No)

---

## 📚 Documentação Adicional

| Documento | Descrição |
|---|---|
| [`docs/ml_canvas.md`](docs/ml_canvas.md) | Canvas de negócio: stakeholders, métricas, SLOs, riscos |
| [`docs/model_card.md`](docs/model_card.md) | Model Card: arquitetura, avaliação, limitações, uso ético |
| [`docs/deploy_architecture.md`](docs/deploy_architecture.md) | Arquitetura de deploy, Docker, endpoints, SLOs |
| [`docs/monitoring_plan.md`](docs/monitoring_plan.md) | Plano de monitoramento, drift, retreinamento |

---

## ✅ Critérios de Qualidade

- [x] Seeds fixados para reprodutibilidade (`random_state=42`)
- [x] Validação com split estratificado (80/20)
- [x] Logging estruturado com MLflow + Python `logging`
- [x] Prevenção de data leakage (scaler ajustado apenas no treino)
- [x] Código modular com responsabilidades separadas (`src/`)
- [x] Testes automatizados com cobertura ≥ 90%
- [x] Linter configurado (ruff, line-length=100, py311)
- [x] Persistência de modelos com `.joblib`
- [x] Contrato de features versionado (`models/feature_names.txt`)

---

## 📝 Licença

Projeto desenvolvido como parte do Tech Challenge da Fase 1 de Machine Learning Engineering — FIAP.

---

**Última atualização:** Maio 2026  
**Versão:** 2.0

**Changelog:**
- **v2.0** (05/2026): Documentação final completa, Makefile, Model Card, plano de monitoramento e arquitetura de deploy adicionados. README atualizado para refletir estado real do projeto.
- **v1.2** (11/04/2026): Reestruturação completa com organização por tipo de modelo e documentação expandida
- **v1.1** (09/04/2026): Implementação dos modelos baseline
- **v1.0** (09/04/2026): Setup inicial e EDA
