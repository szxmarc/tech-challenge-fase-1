# Tech Challenge - Fase 1: Previsão de Churn com Machine Learning

## Descrição do Projeto

Este projeto desenvolve uma solução end-to-end de Machine Learning para previsão de churn em uma operadora de telecomunicações. O projeto está estruturado seguindo as melhores práticas de ML Engineering, com foco em reprodutibilidade, documentação e rastreamento de experimentos.

## 🎯 Objetivo de Negócio

Uma operadora de telecomunicações está enfrentando uma taxa acelerada de cancelamento de clientes (churn). Este projeto visa construir um modelo preditivo que classifique clientes com risco de cancelamento, permitindo ações preventivas para reduzir o churn.

**Métricas de Sucesso:**
- **Técnicas**: AUC-ROC ≥ 0.80, Recall ≥ 0.75, F2-Score ≥ 0.70
- **Negócio**: Valor positivo de retenção (break-even ou lucro)

## Integrantes do Grupo

- **Giovanni de Aguirre Tamanini** (RM371630) - giovanni.tnini@gmail.com
- **Integrante 2** - [Nome e contato a ser preenchido]
- **Integrante 3** - [Nome e contato a ser preenchido]
- **Integrante 4** - [Nome e contato a ser preenchido]

## 📂 Estrutura do Projeto

```
tech_challenge_previsao_churn/
├── data/
│   ├── raw/                                    # Dados originais (nunca modificar)
│   │   └── Telco-Customer-Churn.csv
│   └── processed/                              # Dados processados
│       ├── baseline/                           # Dados dos modelos baseline
│
├── models/                                     # Modelos treinados
│   ├── baseline/                               # Modelos baseline
│   │   ├── dummy_classifier/                   # DummyClassifier
│   │   │   ├── model.joblib
│   │   │   ├── scaler.joblib
│   │   │   └── metrics.json
│   │   └── logistic_regression/                # Regressão Logística
│   │       ├── model.joblib
│   │       ├── scaler.joblib
│   │       └── metrics.json
│
├── notebooks/                                  # Notebooks de análise
│   ├── 01_eda_e_baselines.ipynb               # EDA e modelos baseline
│
├── src/                                        # Código modularizado (Etapa 3)
├── tests/                                      # Testes automatizados (Etapa 3)
├── docs/                                       # Documentação adicional
│   └── ML_Canvas.md                            # ML Canvas (planejado)
├── requirements.txt                            # Dependências do projeto
├── .gitignore                                  # Arquivos ignorados pelo Git
└── README.md                                   # Este arquivo
```

## 🚀 Etapas de Desenvolvimento

### ✅ Etapa 1 - Entendimento e Preparação (CONCLUÍDA)

**Status:** ✅ **COMPLETA**

**Objetivos:**
- [x] Preencher ML Canvas (stakeholders, métricas de negócio, SLOs)
- [x] EDA completa: volume, qualidade, distribuição, data readiness
- [x] Definir métricas técnicas (AUC-ROC, PR-AUC, F1, F2) e métricas de negócio
- [x] Treinar baselines com DummyClassifier e Regressão Logística
- [x] Registrar experimentos no MLflow

**Entregável:** 
- ✅ `notebooks/01_eda_e_baselines.ipynb`
- ✅ Baselines registrados no MLflow
- ✅ Documentação completa de features e decisões

**Resultados dos Baselines:**

| Modelo | AUC-ROC | F2-Score | Valor de Negócio | Status |
|--------|---------|----------|------------------|--------|
| DummyClassifier | 0.516 | 0.291 | -R$ 102.300 | Baseline mínimo |
| **Regressão Logística** | **0.841** | **0.704** | **+R$ 61.500** | ✅ **Promissor** |

**Principais Insights:**
- 📊 Dataset balanceado: 26.5% churn vs 73.5% não-churn
- 🔍 Features mais importantes: `tenure`, `MonthlyCharges`, `Contract_Two year`
- 💰 Regressão Logística já é lucrativa: +R$ 61.500 (R$ 43,65 por cliente)
- 🎯 F2-Score de 0.704 atinge a meta de negócio (≥0.70)
- ✅ AUC-ROC de 0.841 indica excelente capacidade de discriminação (≥0.80)

---

### 📋 Etapa 2 - Modelagem com Redes Neurais (Planejado)

**Objetivos:**
- [ ] Construir MLP em PyTorch
- [ ] Implementar loop de treinamento com early stopping
- [ ] Comparar MLP vs. baselines
- [ ] Analisar trade-off de custo
- [ ] Registrar todos os experimentos no MLflow
- [ ] Otimização de threshold para maximizar valor de negócio

**Modelos a Implementar:**
1. Random Forest
2. XGBoost
3. MLP (PyTorch)
4. Análise de Feature Importance (SHAP)

---

### Etapa 3 - Engenharia e API (Planejado)

**Objetivos:**
- [ ] Refatorar código em módulos (`src/`)
- [ ] Criar pipeline reprodutível
- [ ] Escrever testes (pytest)
- [ ] Construir API FastAPI
- [ ] Adicionar logging estruturado
- [ ] Configurar pyproject.toml, ruff, Makefile

---

### Etapa 4 - Documentação e Entrega Final (Planejado)

**Objetivos:**
- [ ] Gerar Model Card completo
- [ ] Documentar arquitetura de deploy
- [ ] Criar plano de monitoramento
- [ ] Finalizar README
- [ ] Gravar vídeo STAR (5 minutos)
- [ ] (Opcional) Deploy em nuvem

---

## 🛠️ Bibliotecas e Tecnologias

### Core ML:
- **PyTorch** - Construção e treinamento de redes neurais
- **Scikit-Learn** - Pipelines de pré-processamento e modelos baseline
- **MLflow** - Tracking de experimentos

### Análise e Visualização:
- **Pandas, NumPy** - Manipulação e análise de dados
- **Matplotlib, Seaborn** - Visualização de dados (estilo: `whitegrid`)
- **Scipy** - Análises estatísticas

### API e Deploy:
- **FastAPI** - API de inferência (Etapa 3)

---

## 📦 Setup e Instalação

### Pré-requisitos

- Python 3.10 ou superior
- Git
- (Opcional) Jupyter Notebook

### Instalação

1. **Clone o repositório:**
   ```bash
   git clone [URL_DO_REPOSITORIO]
   cd tech_challenge_previsao_churn
   ```

2. **Crie um ambiente virtual:**
   ```bash
   python -m venv venv
   
   # Windows:
   venv\Scripts\activate
   
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎓 Como Executar

### Etapa 1 - Análise Exploratória e Baselines

1. **Navegue até a pasta de notebooks:**
   ```bash
   cd notebooks
   ```

2. **Execute o Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Abra e execute:** `01_eda_e_baselines.ipynb`

   **Estrutura do Notebook:**
   - Seção 1: Setup e Configuração
   - Seção 2: Carregamento dos Dados (com dicionário completo)
   - Seção 3: EDA (análises descritivas, outliers, correlações)
   - Seção 4: Preparação dos Dados
   - Seção 5: Definição de Métricas
   - Seção 6: Modelos Baseline
   - Seção 7: Comparação de Modelos
   - Seção 8: Conclusões e Próximos Passos

---

### MLflow Tracking

Para visualizar os experimentos registrados:

```bash
mlflow ui
```

Acesse: **http://localhost:5000**

**Experimentos Registrados:**
- `Telco Churn - Baseline Models`
  - DummyClassifier (Stratified)
  - Logistic Regression

---

## 📊 Dataset

**Dataset:** Telco Customer Churn (IBM)  
**Fonte:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Características:**
- **Observações:** 7.043 clientes
- **Features:** 20 variáveis independentes + 1 target
- **Tipo:** Classificação binária (Churn: Yes/No)
- **Balanceamento:** 26.5% churn, 73.5% não-churn

**Variáveis Principais:**
- **Demográficas:** gender, SeniorCitizen, Partner, Dependents
- **Serviços:** PhoneService, InternetService, OnlineSecurity, StreamingTV, etc.
- **Contratuais:** tenure, Contract, MonthlyCharges, TotalCharges
- **Target:** Churn (Yes/No)

---

## ✅ Critérios de Qualidade

- [x] Seeds fixados para reprodutibilidade (`random_state=42`)
- [x] Validação com split estratificado (80/20)
- [x] Logging estruturado com MLflow
- [x] Documentação completa em markdown
- [x] Prevenção de data leakage
- [x] Commits significativos e histórico limpo
- [x] Notebook padronizado seguindo referência do curso
- [x] **Uso exclusivo de `.joblib` para persistência de modelos**
- [x] **Estrutura organizada por tipo de modelo e experimento**
- [x] **Código modular e reutilizável**
- [x] **Documentação técnica detalhada**

---

## 📋 Convenções de Nomenclatura

### Arquivos de Modelos
- **Modelos scikit-learn/XGBoost:** `model.joblib`
- **Modelos PyTorch:** `model.pt`
- **Scalers:** `scaler.joblib`
- **Métricas:** `metrics.json`

### Diretórios de Experimentos
- **Formato:** `experiment_XXX` (ex: `experiment_001`, `experiment_002`)
- Cada experimento contém: modelo, scaler e métricas
- Melhor modelo de cada tipo fica em `best_model/`
- Modelo final de produção em `production/`

### Por que `.joblib`?
- Mais eficiente que pickle para arrays NumPy grandes
- Padrão recomendado pelo scikit-learn
- Compressão automática de dados
- Melhor performance de leitura/escrita

---

## 🔄 Como Adicionar Novos Experimentos

### 1. Carregar Dados Processados
```python
import joblib

# Carregar dados já preparados
X_train = joblib.load('data/processed/baseline/X_train_scaled.joblib')
y_train = joblib.load('data/processed/baseline/y_train.joblib')
```

### 2. Treinar Modelo
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 3. Salvar Experimento
```python
import os
import json

# Criar diretório do experimento
exp_dir = 'models/random_forest/experiment_001/'
os.makedirs(exp_dir, exist_ok=True)

# Salvar modelo e scaler
joblib.dump(model, f'{exp_dir}/model.joblib')
joblib.dump(scaler, f'{exp_dir}/scaler.joblib')

# Salvar métricas
metrics = {'auc_roc': 0.85, 'f2_score': 0.72, ...}
with open(f'{exp_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

### 4. Registrar no MLflow
```python
import mlflow

with mlflow.start_run(run_name="random_forest_exp_001"):
    mlflow.log_params({...})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
```

---

## 📚 Documentação Adicional

- **Notebooks**: Cada notebook contém documentação detalhada inline

---

---

## 📝 Licença

Este projeto foi desenvolvido como parte do Tech Challenge da Fase 1 de Machine Learning Engineering.

---

## 📞 Contato

Para dúvidas ou sugestões, entre em contato com os integrantes do grupo listados acima.

---

**Última atualização:** 11 de Abril de 2026  
**Versão:** 1.2

**Changelog:**
- **v1.2** (11/04/2026): Reestruturação completa do projeto com organização por tipo de modelo, convenções de nomenclatura padronizadas, documentação expandida
- **v1.1** (09/04/2026): Implementação dos modelos baseline (DummyClassifier e Regressão Logística)
- **v1.0** (09/04/2026): Setup inicial do projeto e EDA

