# Estrutura de Projeto - Tech Challenge FIAP Fase 1
## Previsão de Churn em Telecomunicações

---

## 📁 Organização de Pastas (Opção 1 - Simplificada)

```
tech-challenge-fase-1/
│
├── 📂 data/
│   ├── raw/                              # Dataset original (NUNCA MODIFICAR)
│   │   └── Telco-Customer-Churn.csv
│   │
│   └── processed/
│       └── logistic_regression/          # Dados processados para o modelo final
│           ├── X_train.csv               # Features de treino (padronizadas)
│           ├── X_test.csv                # Features de teste (padronizadas)
│           ├── y_train.csv               # Target de treino
│           ├── y_test.csv                # Target de teste
│           └── scaler.joblib             # Objeto StandardScaler (fit no treino)
│
├── 📂 models/
│   ├── logistic_regression/              # Modelo final treinado
│   │   ├── model.joblib                  # Modelo LogisticRegression treinado
│   │   ├── scaler.joblib                 # Scaler (cópia para facilitar carga)
│   │   ├── config.json                   # Hiperparâmetros do modelo
│   │   ├── metrics.json                  # Métricas técnicas e de negócio
│   │   ├── feature_importance.csv        # Coeficientes das features
│   │   └── feature_names.txt             # Nomes das features (order preservation)
│   │
│   └── baseline_comparison.csv           # Comparação entre diferentes modelos
│
├── 📂 notebooks/
│   ├── 01_prepare_data.ipynb             # EDA + Preparação de dados
│   └── 02_train_logistic_regression.ipynb # Treinamento do modelo final
│
├── 📂 reports/
│   └── model_summary.md                  # Sumário executivo dos resultados
│
├── 📂 src/                               # 🔑 PACOTE PRINCIPAL - Código Centralizado
│   ├── __init__.py                       # Importações públicas do pacote
│   │
│   ├── 📂 config/                        # 🔧 CONFIGURAÇÕES CENTRALIZADAS
│   │   ├── __init__.py
│   │   └── settings.py                   # Parâmetros globais
│   │
│   ├── 📂 data/                          # 📊 OPERAÇÕES COM DADOS
│   │   ├── __init__.py
│   │   ├── loader.py                     # Carregamento de datasets
│   │   └── processor.py                  # Transformação de dados
│   │
│   ├── 📂 models/                        # 🤖 TREINAMENTO E PERSISTÊNCIA
│   │   ├── __init__.py
│   │   └── trainer.py                    # Treinamento e salvamento
│   │
│   ├── 📂 evaluation/                    # 📈 MÉTRICAS E ANÁLISE
│   │   ├── __init__.py
│   │   ├── metrics.py                    # Cálculo de métricas
│   │   └── features.py                   # Feature importance
│   │
│   ├── 📂 visualization/                 # 📉 GERAÇÃO DE GRÁFICOS
│   │   ├── __init__.py
│   │   └── plots.py                      # Plotagens e visualizações
│   │
│   ├── 📂 utils/                         # 🛠️ UTILITÁRIOS GERAIS
│   │   ├── __init__.py
│   │   └── helpers.py                    # Funções auxiliares
│   │
│   ├── 📂 api/                           # 🌐 APLICAÇÃO FLASK REST
│   │   ├── __init__.py
│   │   ├── app.py                        # Configuração Flask
│   │   ├── routes.py                     # Endpoints da API
│   │   └── run.py                        # Script para rodar app
│   │
│   └── README.md                         # Documentação de src/
│
├── 📂 docs/
│   ├── ml_canvas.md                      # Business Model Canvas
│   └── telco_churn_mlp.ipynb             # Documentação técnica
│
├── 🚀 run_app.py                         # PONTO DE ENTRADA - Inicia Flask API
├── .gitignore
├── requirements.txt                      # Dependências do projeto
├── README.md                             # Documentação principal
├── ESTRUTURA_PROJETO.md                  # Este arquivo
└── setup.py                              # Package setup (opcional)
```

---

## 🎯 Separação de Responsabilidades

### **Notebook 1: Preparação de Dados** (`01_prepare_data.ipynb`)
**Responsabilidade:** Exploração e Preparação

- ✅ Carregamento dos dados brutos
- ✅ Análise Exploratória (EDA)
- ✅ Tratamento de valores faltantes
- ✅ Encoding de variáveis categóricas
- ✅ Split estratificado treino/teste
- ✅ Padronização com StandardScaler
- ✅ Exportação de dados processados

**Output:** Dados limpos e padronizados em `data/processed/logistic_regression/`

---

### **Notebook 2: Treinamento do Modelo** (`02_train_logistic_regression.ipynb`)
**Responsabilidade:** Modelagem e Avaliação

- ✅ Carregamento dos dados pré-processados
- ✅ Configuração e treinamento do LogisticRegression
- ✅ Avaliação com métricas técnicas
- ✅ Cálculo de métricas de negócio
- ✅ Análise de Feature Importance
- ✅ Visualizações (ROC, Precision-Recall, Confusion Matrix)
- ✅ Persistência do modelo e artefatos
- ✅ Registro de experimentos no MLflow

**Output:** Modelo treinado em `models/logistic_regression/`

---

## 📊 Estrutura Modularizada do `src/`

A pasta `src/` é organizada em **módulos temáticos**, cada um com responsabilidade clara:

---

### **📂 src/config/** - Configurações Centralizadas
**Propósito:** Centralizar todos os parâmetros do projeto em um único lugar.

**Arquivos:**
- `settings.py` - Parâmetros globais do projeto:
  - Caminhos de diretórios (data/, models/, etc)
  - Parâmetros do modelo (max_iter, C, solver, etc)
  - Configuração de split treino/teste
  - Métricas de negócio (custos TP, FP, FN, TN)
  - Metas de desempenho
  - Configuração MLflow

**Benefício:** Mudança fácil de parâmetros sem editar múltiplos arquivos.

```python
from src.config.settings import MODEL_CONFIG, BUSINESS_METRICS, DATA_PATHS
```

---

### **📂 src/data/** - Operações com Dados
**Propósito:** Gerenciar todo o ciclo de vida dos dados (carregamento, transformação, persistência).

**Arquivos:**
- `loader.py` - Funções de carregamento:
  - `load_raw_data()` - Carrega dataset bruto
  - `load_processed_data()` - Carrega dados já processados
  
- `processor.py` - Funções de transformação:
  - `handle_missing_values()` - Trata NaNs
  - `encode_features()` - Label e One-Hot Encoding
  - `split_and_scale_data()` - Split estratificado + StandardScaler
  - `save_processed_data()` - Persiste dados processados

**Benefício:** Reutilização de código entre notebooks/modelos.

```python
from src.data.loader import load_raw_data, load_processed_data
from src.data.processor import encode_features, split_and_scale_data
```

---

### **📂 src/models/** - Treinamento e Persistência
**Propósito:** Encapsular lógica de treinamento e salvamento de modelos.

**Arquivos:**
- `trainer.py` - Funções de treinamento:
  - `train_logistic_regression()` - Treina e retorna modelo
  - `save_model()` - Persiste modelo e artefatos
  - `load_model()` - Carrega modelo treinado

**Benefício:** Padronização do processo de treinamento e persistência.

```python
from src.models.trainer import train_logistic_regression, save_model, load_model
```

---

### **📂 src/evaluation/** - Métricas e Análise
**Propósito:** Calcular métricas técnicas e de negócio, analisar importância de features.

**Arquivos:**
- `metrics.py` - Cálculo de métricas:
  - `evaluate_model()` - Calcula Accuracy, Precision, Recall, F1, F2, AUC-ROC, PR-AUC
  - `calculate_business_value()` - Calcula ROI e valor total baseado em custos
  
- `features.py` - Análise de features:
  - `get_feature_importance()` - Extrai coeficientes do modelo
  - `rank_features()` - Ranking de features por importância

**Benefício:** Métricas consistentes em todos os notebooks e modelos.

```python
from src.evaluation.metrics import evaluate_model, calculate_business_value
from src.evaluation.features import get_feature_importance, rank_features
```

---

### **📂 src/visualization/** - Geração de Gráficos
**Propósito:** Centralizar funções de plotagem para reutilização.

**Arquivos:**
- `plots.py` - Funções de visualização:
  - `plot_roc_curve()` - Curva ROC
  - `plot_precision_recall()` - Curva Precision-Recall
  - `plot_confusion_matrix()` - Matriz de confusão
  - `plot_feature_importance()` - Ranking de features

**Benefício:** Visualizações consistentes e reutilizáveis.

```python
from src.visualization.plots import plot_roc_curve, plot_confusion_matrix, plot_feature_importance
```

---

### **📂 src/utils/** - Utilitários Gerais
**Propósito:** Funções auxiliares pequenas que não se encaixam em nenhuma categoria.

**Arquivos:**
- `helpers.py` - Funções auxiliares:
  - `create_directories()` - Cria pastas necessárias
  - `log_experiment()` - Registra experimentos
  - `save_json()`, `load_json()` - Manipulação de JSON

**Benefício:** Código limpo e reutilizável para tarefas comuns.

```python
from src.utils.helpers import create_directories, log_experiment, save_json
```

---

### **📂 src/api/** - Aplicação Flask REST
**Propósito:** Expor o modelo ML através de uma API REST para predições em produção.

**Arquivos:**
- `app.py` - Configuração e inicialização:
  - Configurações Flask (debug, JSON, etc)
  - Health check endpoints
  - Registro de blueprints
  - Error handlers
  
- `routes.py` - Definição de endpoints:
  - `GET /health` - Verifica saúde da API e modelo
  - `GET /api/v1/model/info` - Informações do modelo
  - `GET /api/v1/features` - Lista de features esperadas
  - `POST /api/v1/predict` - Predição individual
  - `POST /api/v1/predict/batch` - Predição em batch
  
- `run.py` - Script para rodar a aplicação

**Benefício:** API REST padronizada e produção-ready.

```python
from src.api.app import create_app

app = create_app(config_name='production')
app.run(host='0.0.0.0', port=5000)
```

---

## 🌐 Usando a API

### **Como Iniciar**
```bash
# Opção 1: Rodar via run_app.py na raiz (recomendado)
python run_app.py

# Opção 2: Rodar via src/api/run.py
python src/api/run.py
```

### **Exemplo: Predição Individual**
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.5, "feature_2": 1.0, ...}'
```

### **Exemplo: Predição em Batch**
```bash
curl -X POST http://localhost:5000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"feature_1": 0.5, "feature_2": 1.0, ...},
      {"feature_1": 0.3, "feature_2": 0.8, ...}
    ]
  }'
```

---

### **Passo 1: Preparação de Dados**
```bash
jupyter notebook notebooks/01_prepare_data.ipynb
# Gera: data/processed/logistic_regression/
```

### **Passo 2: Treinamento do Modelo**
```bash
jupyter notebook notebooks/02_train_logistic_regression.ipynb
# Gera: models/logistic_regression/ + MLflow runs
```

### **Passo 3: Visualizar Experimentos MLflow**
```bash
mlflow ui --backend-store-uri file:./mlruns
# Acessa: http://localhost:5000
```

---

## 📋 Importações Recomendadas por Módulo

### **Nos Notebooks de Dados**
```python
from src.data.loader import load_raw_data, load_processed_data
from src.data.processor import encode_features, split_and_scale_data, save_processed_data
from src.config.settings import DATA_PATHS, DATA_CONFIG
```

### **Nos Notebooks de Treinamento**
```python
from src.models.trainer import train_logistic_regression, save_model, load_model
from src.evaluation.metrics import evaluate_model, calculate_business_value
from src.evaluation.features import get_feature_importance, rank_features
from src.visualization.plots import plot_roc_curve, plot_confusion_matrix, plot_feature_importance
from src.config.settings import MODEL_CONFIG, BUSINESS_METRICS, DATA_PATHS
```

### **Em Scripts Utilitários**
```python
from src.utils.helpers import create_directories, save_json, load_json, log_experiment
from src.config.settings import MLFLOW_CONFIG
```

---

## 🔄 Organização Hierárquica de `src/` (Centralizado)

```
src/                                      # 🔑 PACOTE PRINCIPAL
├── __init__.py                           # Importações públicas
│
├── config/                               # 🔧 CONFIGURAÇÕES
│   ├── __init__.py
│   └── settings.py
│
├── data/                                 # 📊 DADOS
│   ├── __init__.py
│   ├── loader.py
│   └── processor.py
│
├── models/                               # 🤖 TREINAMENTO
│   ├── __init__.py
│   └── trainer.py
│
├── evaluation/                           # 📈 MÉTRICAS
│   ├── __init__.py
│   ├── metrics.py
│   └── features.py
│
├── visualization/                        # 📉 GRÁFICOS
│   ├── __init__.py
│   └── plots.py
│
├── utils/                                # 🛠️ UTILITÁRIOS
│   ├── __init__.py
│   └── helpers.py
│
├── api/                                  # 🌐 FLASK REST API
│   ├── __init__.py
│   ├── app.py
│   ├── routes.py
│   └── run.py
│
└── README.md                             # Documentação do src/
```

---

## 📈 Métricas Monitoradas

### **Técnicas**
- Accuracy: 0.7971
- Precision: 0.5489  
- Recall: 0.7807 ✅ (≥0.75)
- F1-Score: 0.6447
- F2-Score: 0.7044 ✅ (≥0.70)
- AUC-ROC: 0.8414 ✅ (≥0.80)
- PR-AUC: 0.6748 ✅ (≥0.65)

### **Negócio**
- Valor Total: +R$ 61.500
- Valor Médio por Cliente: R$ 43,65

---

## 🔄 Adicionando Novos Modelos

Para adicionar um **novo modelo** (ex: Random Forest), siga esse padrão:

1. **Nova pasta de dados:**
   ```
   data/processed/random_forest/
   ```

2. **Nova pasta de modelo:**
   ```
   models/random_forest/
   ```

3. **Novo notebook:**
   ```
   notebooks/03_train_random_forest.ipynb
   ```

**Importante:** Não use versões numéricas (v1, v2) nos nomes. Use nomes **descritivos finais**.

---

## 📝 Configurações Principais

### **Parâmetros do LogisticRegression**
```python
{
  "max_iter": 1000,
  "C": 1.0,
  "solver": "lbfgs",
  "class_weight": "balanced",
  "random_state": 42
}
```

### **Split Treino/Teste**
- Test Size: 20% (1.409 amostras)
- Train Size: 80% (5.634 amostras)
- Stratified: Sim (mantém proporção de churners)

### **Custos de Negócio**
- TP (Cliente retido): +R$ 400
- FP (Ação desnecessária): -R$ 50
- FN (Cliente perdido): -R$ 500
- TN (Sem custo): R$ 0

---

---

## ✍️ Boas Práticas

### **1️⃣ Sempre Centralizar em `src/config/settings.py`**
- Novos parâmetros? Adicione em `settings.py`
- Evite hardcoding de valores nos notebooks
- Facilita reprodutibilidade e rastreamento

### **2️⃣ Use Importações Relativas do `src/`**
```python
# ✅ BOM
from src.data.loader import load_raw_data
from src.models.trainer import train_logistic_regression

# ❌ EVITAR
import sys; sys.path.append('../src')
from config import MODEL_CONFIG  # Sem 'src.'
```

### **3️⃣ Não Misture Responsabilidades**
- Carregamento de dados → `src/data/loader.py`
- Processamento → `src/data/processor.py`
- Treinamento → `src/models/trainer.py`
- Métricas → `src/evaluation/metrics.py`
- Gráficos → `src/visualization/plots.py`

### **4️⃣ Cada Notebook tem um Propósito Único**
```
notebooks/
├── 01_prepare_data.ipynb          # EDA + Transformação
└── 02_train_logistic_regression.ipynb  # Treinamento + Avaliação
```

### **5️⃣ Adicione Docstrings em Tudo**
- Funções devem documentar: O quê, Args, Returns
- Facilita autocompletar no IDE e uso por outros

### **6️⃣ Use Logging/Print Amigável**
```python
print("✅ Dados processados")     # Sucesso
print("⚠️ Aviso: valor faltante")  # Aviso
print("❌ Erro crítico")            # Erro
```

---

## 📊 Data Preparation é Preparação, não EDA

1. **Foco em transformações**, não em análises profundas
2. **Exportar dados limpos** ao final
3. **EDA é exploratório** - fica nos notebooks, não em módulos `src/`

---

## 🔄 Modelos em Pastas Separadas (Não Versões)

**❌ EVITAR:**
- `v1_baseline`, `v2_smote`, `v3_rf`
- `logistic_regression_v1`, `logistic_regression_v2`

**✅ USAR:**
- `logistic_regression/`, `random_forest/`, `xgboost/`
- `naive_bayes/`, `svm/`, `gradient_boosting/`

---

## 📁 Como Adicionar um Novo Modelo

### **Exemplo: Treinar Random Forest**

1. **Criar folder de dados:**
```
data/processed/random_forest/
```

2. **Criar folder de modelo:**
```
models/random_forest/
```

3. **Novo notebook:**
```
notebooks/03_train_random_forest.ipynb
```

4. **Usar importações padrão:**
```python
from src.data.loader import load_raw_data
from src.data.processor import encode_features, split_and_scale_data
from src.config.settings import MODEL_CONFIG, BUSINESS_METRICS, DATA_PATHS
from src.evaluation.metrics import evaluate_model, calculate_business_value
```

---

## 🧪 Testando Importações e Executando

Para garantir que tudo está funcionando, rode no terminal:

```bash
# Teste todas as importações
python -c "from src.config.settings import MODEL_CONFIG; print('✅ Config OK')"
python -c "from src.data.loader import load_raw_data; print('✅ Data Loader OK')"
python -c "from src.models.trainer import train_logistic_regression; print('✅ Trainer OK')"
python -c "from src.evaluation.metrics import evaluate_model; print('✅ Evaluation OK')"
python -c "from src.visualization.plots import plot_roc_curve; print('✅ Visualization OK')"
python -c "from src.utils.helpers import create_directories; print('✅ Utils OK')"
python -c "from src.api.app import create_app; print('✅ API OK')"

# Iniciar API Flask
python run_app.py
# Ou
python src/api/run.py
```

---

## 📚 Documentação Completa

- **src/README.md** - Guia detalhado de módulos e endpoints
- **docs/ml_canvas.md** - Business Model Canvas
- **Dataset:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## ✅ Resumo: O que foi Centralizado

| Item | Antes | Depois |
|------|-------|--------|
| **Configurações** | `config.py` fora de src/ | `src/config/settings.py` ✅ |
| **Preprocessing** | `preprocessing.py` fora de src/ | `src/data/processor.py` ✅ |
| **Evaluation** | `evaluation.py` fora de src/ | `src/evaluation/` ✅ |
| **API** | Não existia | `src/api/` com Flask ✅ |
| **Entry Point** | Não existia | `run_app.py` na raiz ✅ |

---

**Última Atualização:** Abril 2026  
**Status:** Estrutura Completamente Organizada e Pronta para Flask API ✅
