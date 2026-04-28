# ✅ Reorganização Completa - Tech Challenge FIAP

## Resumo das Mudanças

### 🎯 Objetivo Alcançado
**Separação clara entre preparação de dados e aprendizado do modelo, sem versionamento numérico.**

---

## 📦 Estrutura de Pastas Criada

### **Novas Pastas**
```
✅ src/                  # Código reutilizável
✅ data/processed/logistic_regression/    # Dados processados
✅ models/logistic_regression/             # Modelo final
✅ reports/              # Relatórios e sumários
```

### **Arquivos de Configuração**
```
✅ src/config.py         # Configurações centralizadas
✅ src/preprocessing.py  # Funções de pré-processamento
✅ src/evaluation.py     # Funções de avaliação
✅ src/__init__.py       # Package initialization
```

---

## 📓 Notebooks Resultantes

### **1️⃣ Preparação de Dados**
**Nome:** `01_prepare_data.ipynb`

**Responsabilidade:** 
- Carregar dados brutos
- EDA (Exploração + Qualidade)
- Tratamento de faltantes
- Encoding de variáveis
- Split treino/teste estratificado
- **Exportar dados processados**

**Output:**
```
data/processed/logistic_regression/
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
└── scaler.joblib
```

---

### **2️⃣ Treinamento do Modelo**
**Nome:** `02_train_logistic_regression.ipynb`

**Responsabilidade:**
- **Carregar** dados pré-processados
- Treinar LogisticRegression
- Calcular métricas técnicas
- Calcular métricas de negócio
- Análise de Features
- **Salvar modelo** com artefatos
- Registrar no MLflow

**Output:**
```
models/logistic_regression/
├── model.joblib
├── scaler.joblib
├── config.json
├── metrics.json
└── feature_importance.csv
```

---

## 🔑 Principais Benefícios

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Separação** | ❌ Tudo em 1 notebook | ✅ 2 notebooks distintos |
| **Reutilização** | ❌ Código duplicado | ✅ Funções em `src/` |
| **Configuração** | ❌ Hard-coded | ✅ Centralizada em `config.py` |
| **Versionamento** | ❌ v1, v2, v3... | ✅ Nomes descritivos finais |
| **Manutenção** | ❌ Difícil encontrar código | ✅ Organização clara |
| **Escalabilidade** | ❌ Difícil adicionar modelos | ✅ Simples estender para RF, XGB |

---

## 🚀 Como Usar

### **Execução Sequencial Recomendada**

```
1. Executar: 01_prepare_data.ipynb
   ↓
   Gera dados em: data/processed/logistic_regression/
   
2. Executar: 02_train_logistic_regression.ipynb
   ↓
   Gera modelo em: models/logistic_regression/
   
3. (Opcional) Visualizar MLflow:
   mlflow ui --backend-store-uri file:./mlruns
```

---

## 📐 Exemplos de Uso

### **Usar funções de preprocessing**
```python
from src.preprocessing import load_raw_data, encode_features, split_and_scale_data

# Carregar
df = load_raw_data(RAW_DATASET)

# Processar
X, y, feature_names = encode_features(df)

# Split e Scale
data = split_and_scale_data(X, y)
```

### **Usar configurações globais**
```python
from src.config import MODEL_CONFIG, BUSINESS_METRICS, MODELS_DIR

# Parâmetros do modelo
print(MODEL_CONFIG)
# {'max_iter': 1000, 'C': 1.0, ...}

# Custos de negócio
print(BUSINESS_METRICS)
# {'cost_true_positive': 400, ...}

# Salvar em diretório certo
model_path = MODELS_DIR / 'model.joblib'
```

### **Usar função de avaliação**
```python
from src.evaluation import evaluate_model

metrics = evaluate_model(
    y_true=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    model_name="LogisticRegression",
    business_metrics=BUSINESS_METRICS,
    verbose=True
)

# Salvar métricas
import json
with open(METRICS_FILE, 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

## 📊 Modelo Atual - LogisticRegression

### **Parâmetros**
```python
{
  "max_iter": 1000,
  "C": 1.0,
  "solver": "lbfgs",
  "class_weight": "balanced",
  "random_state": 42
}
```

### **Desempenho (Esperado)**
| Métrica | Valor | Status |
|---------|-------|--------|
| AUC-ROC | 0.8414 | ✅ Atende meta (≥0.80) |
| PR-AUC | 0.6748 | ✅ Atende meta (≥0.65) |
| Recall | 0.7807 | ✅ Atende meta (≥0.75) |
| F2-Score | 0.7044 | ✅ Atende meta (≥0.70) |
| **Valor Total** | **+R$ 61.500** | ✅ Lucrativo |

---

## 🔄 Próximos Passos para Novos Modelos

Para adicionar **Random Forest**, por exemplo:

### **1. Criar estrutura de pastas**
```bash
mkdir -p data/processed/random_forest
mkdir -p models/random_forest
```

### **2. Criar novo notebook**
```
notebooks/03_train_random_forest.ipynb
```

### **3. Usar mesmo padrão de carregamento**
```python
from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.evaluation import evaluate_model

# Carregar dados pré-processados (reutilizar de 01_prepare_data)
X_train = pd.read_csv(f'{DATA_PROCESSED_DIR}/logistic_regression/X_train.csv')
# ...
```

### **4. Salvar no local correto**
```python
# Em vez de: models/v2_random_forest/
# Usar: models/random_forest/
```

---

## 📁 Checklist Final

- ✅ Estrutura de pastas organizada
- ✅ Código modularizado em `src/`
- ✅ Configurações centralizadas
- ✅ 2 notebooks separados (preparação + treinamento)
- ✅ Sem numeração de versões (v1, v2, ...)
- ✅ Documentação clara (ESTRUTURA_PROJETO.md)
- ✅ Pronto para adicionar novos modelos
- ✅ MLflow configurado
- ✅ Reprodutibilidade garantida (seed=42)

---

## 💡 Filosofia de Organização

> **"Um modelo = Uma pasta + Um notebook"**

Não importa quantos modelos você treina:
- LogisticRegression → `models/logistic_regression/`
- RandomForest → `models/random_forest/`
- XGBoost → `models/xgboost/`
- MLP → `models/neural_network/`

Todos **no mesmo nível**, sem versões numéricas.

---

## 📞 Referências Rápidas

| Arquivo | Propósito |
|---------|-----------|
| `src/config.py` | Configurações globais |
| `src/preprocessing.py` | Funções de pré-processamento |
| `src/evaluation.py` | Funções de avaliação |
| `notebooks/01_prepare_data.ipynb` | Preparação de dados |
| `notebooks/02_train_logistic_regression.ipynb` | Treinamento do modelo |
| `ESTRUTURA_PROJETO.md` | Documentação da estrutura |

---

**Status:** ✅ Reorganização Completa e Pronta para Uso

**Data:** Abril 2026
