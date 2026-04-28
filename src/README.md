# 🔬 Módulo `src/` - Código Reutilizável

Este diretório contém todo o código reutilizável do projeto, organizado por responsabilidade.

## 📂 Estrutura de Pastas

### **config/** - Configurações Centralizadas
Centraliza parâmetros do projeto (caminhos, hiperparâmetros, métricas de negócio).
- **settings.py** → Todos os parâmetros globais do projeto

```python
from src.config.settings import MODEL_CONFIG, BUSINESS_METRICS, DATA_PATHS
```

---

### **data/** - Operações com Dados
Carregamento, transformação e persistência de dados.
- **loader.py** → Carrega datasets brutos e processados
- **processor.py** → Codificação, scaling, split estratificado

```python
from src.data.loader import load_raw_data, load_processed_data
from src.data.processor import encode_features, split_and_scale_data, save_processed_data
```

---

### **models/** - Treinamento e Persistência
Encapsula a lógica de treinamento e salvamento de modelos.
- **trainer.py** → Treina, valida e persiste modelos

```python
from src.models.trainer import train_logistic_regression, save_model, load_model
```

---

### **evaluation/** - Métricas e Análise
Cálculo de métricas técnicas e de negócio, análise de features.
- **metrics.py** → Accuracy, Precision, Recall, F1, AUC, valor de negócio
- **features.py** → Importância de features, ranking

```python
from src.evaluation.metrics import evaluate_model, calculate_business_value
from src.evaluation.features import get_feature_importance, rank_features
```

---

### **visualization/** - Geração de Gráficos
Centraliza plotagens para reutilização.
- **plots.py** → ROC, Precision-Recall, Confusion Matrix, Feature Importance

```python
from src.visualization.plots import plot_roc_curve, plot_confusion_matrix, plot_feature_importance
```

---

### **utils/** - Utilitários Gerais
Funções auxiliares que não se encaixam em nenhuma categoria.
- **helpers.py** → Criar diretórios, logging, manipular JSON

```python
from src.utils.helpers import create_directories, save_json, load_json, log_experiment
```

---

### **api/** - 🌐 Aplicação Flask REST
Expõe o modelo ML através de uma API REST.
- **app.py** → Configuração e inicialização da Flask app
- **routes.py** → Definição de endpoints e blueprints
- **run.py** → Script para rodar a app

```python
from src.api.app import create_app

app = create_app()
app.run()
```

---

## 🌐 API REST - Endpoints

### **Health Check**
```
GET /health
GET /api/v1/health
```
Verifica se a API e o modelo estão operacionais.

**Response:**
```json
{
  "status": "healthy",
  "message": "✅ API e modelo operacionais"
}
```

---

### **Informações do Modelo**
```
GET /api/v1/model/info
```
Retorna detalhes do modelo treinado.

**Response:**
```json
{
  "model_type": "LogisticRegression",
  "n_features": 20,
  "n_classes": 2,
  "classes": [0, 1]
}
```

---

### **Lista de Features**
```
GET /api/v1/features
```
Retorna lista de features esperadas pela API.

**Response:**
```json
{
  "n_features": 20,
  "features": ["feature_1", "feature_2", ...]
}
```

---

### **Predição Individual**
```
POST /api/v1/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "feature_1": 0.5,
  "feature_2": 1.0,
  "feature_3": 0.3,
  ...
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability_no_churn": 0.35,
  "probability_churn": 0.65,
  "confidence": 0.65
}
```

---

### **Predição em Batch**
```
POST /api/v1/predict/batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "customers": [
    {"feature_1": 0.5, "feature_2": 1.0, ...},
    {"feature_1": 0.3, "feature_2": 0.8, ...}
  ]
}
```

**Response:**
```json
{
  "total": 2,
  "predictions": [
    {
      "customer_index": 0,
      "prediction": 1,
      "prediction_label": "Churn",
      "probability_no_churn": 0.35,
      "probability_churn": 0.65
    },
    {
      "customer_index": 1,
      "prediction": 0,
      "prediction_label": "Não Churn",
      "probability_no_churn": 0.75,
      "probability_churn": 0.25
    }
  ]
}
```

---

## 🚀 Como Rodar

### **1. Opção A: Rodar via run_app.py (Recomendado)**
```bash
cd /caminho/para/projeto
python run_app.py
```

### **2. Opção B: Rodar via src/api/run.py**
```bash
python src/api/run.py
```

### **3. Opção C: Rodar via Python direto**
```bash
python -c "from src.api.app import create_app; app = create_app(); app.run()"
```

---

## 📝 Exemplo Prático: Cliente Python

```python
# Importar tudo o que precisar no início
from src.config.settings import MODEL_CONFIG, BUSINESS_METRICS, DATA_PATHS
from src.data.loader import load_raw_data
from src.data.processor import encode_features, split_and_scale_data, save_processed_data
from src.models.trainer import train_logistic_regression, save_model
from src.evaluation.metrics import evaluate_model, calculate_business_value
from src.evaluation.features import get_feature_importance
from src.visualization.plots import plot_roc_curve, plot_confusion_matrix

# 1. Carregar dados
df = load_raw_data()

# 2. Processar
df_encoded = encode_features(df)
X, y = df_encoded.drop('Churn', axis=1), df_encoded['Churn']

# 3. Split e Scale
X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

# 4. Salvar dados processados
save_processed_data(X_train, X_test, y_train, y_test, scaler)

# 5. Treinar
model = train_logistic_regression(X_train, y_train)

# 6. Salvar modelo
save_model(model, scaler)

# 7. Avaliar
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = evaluate_model(y_test, y_pred, y_pred_proba)
business = calculate_business_value(y_test, y_pred)

# 8. Visualizar
plot_roc_curve(y_test, y_pred_proba)
plot_confusion_matrix(y_test, y_pred)
get_feature_importance(model, X_train.columns)
```

---

## ✅ Boas Práticas

1. **Sempre use `src/config/settings.py` para parâmetros**
   - Facilita rastreamento e reprodutibilidade

2. **Cada função deve ter docstring clara**
   - O quê, Args, Returns

3. **Use typing quando possível**
   - `def load_raw_data() -> pd.DataFrame:`

4. **Prints amigáveis com emojis**
   - ✅ Sucesso | ⚠️ Aviso | ❌ Erro

5. **Não misture responsabilidades**
   - Carregamento → `data/loader.py`
   - Transformação → `data/processor.py`
   - Treinamento → `models/trainer.py`
   - Avaliação → `evaluation/`
   - Visualização → `visualization/`
   - API REST → `api/`

---

## 🧪 Testando Importações

Para garantir que tudo está funcionando:

```bash
# Teste todas as importações
python -c "from src.config.settings import MODEL_CONFIG; print('✅ Config OK')"
python -c "from src.data.loader import load_raw_data; print('✅ Data Loader OK')"
python -c "from src.models.trainer import train_logistic_regression; print('✅ Trainer OK')"
python -c "from src.evaluation.metrics import evaluate_model; print('✅ Evaluation OK')"
python -c "from src.visualization.plots import plot_roc_curve; print('✅ Visualization OK')"
python -c "from src.utils.helpers import create_directories; print('✅ Utils OK')"
python -c "from src.api.app import create_app; print('✅ API OK')"
```

---

## 📚 Documentação Completa

Para mais detalhes sobre a arquitetura do projeto, veja:
- [ESTRUTURA_PROJETO.md](../ESTRUTURA_PROJETO.md) - Visão geral completa
- [README.md](../README.md) - Documentação principal do projeto

---

**Última Atualização:** Abril 2026  
**Status:** Estrutura Organizada com API Flask ✅
- [ESTRUTURA_PROJETO.md](../ESTRUTURA_PROJETO.md) - Visão geral completa
- [README.md](../README.md) - Documentação principal do projeto

---

**Última Atualização:** Abril 2026  
**Status:** Estrutura Organizada ✅
