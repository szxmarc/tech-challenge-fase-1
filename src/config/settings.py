"""
Arquivo de Settings - Configurações Globais do Projeto
=======================================================

Propósito:
----------
Definir TODAS as constantes e configurations do projeto.
Centralizar paths absolutos, parâmetros, e métricas de negócio.

Vantagens:
- 1 source of truth para configurações
- Fácil atualizar parâmetros sem mexer em notebooks
- Reprodutibilidade garantida
- Documentação clara de choices

Uso em notebooks:
    from src.config.settings import MODEL_CONFIG, BUSINESS_METRICS
    
    print(MODEL_CONFIG)  
    # {'max_iter': 1000, 'C': 1.0, 'solver': 'lbfgs', ...}
"""

import os
from pathlib import Path

# ============================================================================
# 1. PATHS (Caminhos de Diretórios)
# ============================================================================
# Propósito: Definir caminhos absolutos para que o código funcione em 
# qualquer máquina, em qualquer diretório de execução.

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "baseline"
MODELS_DIR = PROJECT_ROOT / "models" / "baseline" / "logistic_regression"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Criar diretórios se não existirem
for dir_path in [DATA_PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# 2. DATA FILES (Arquivos de Dados)
# ============================================================================
# Propósito: Centralizar nomes e caminhos de arquivos de dados.
# Facilita mudanças de nomes e localizações sem quebrar código.

# Dataset bruto (NUNCA modifique!)
RAW_DATASET = DATA_RAW_DIR / "Telco-Customer-Churn.csv"

# Dados processados (outputs do notebook 01)
X_TRAIN_FILE = DATA_PROCESSED_DIR / "X_train.csv"
X_TEST_FILE = DATA_PROCESSED_DIR / "X_test.csv"
Y_TRAIN_FILE = DATA_PROCESSED_DIR / "y_train.csv"
Y_TEST_FILE = DATA_PROCESSED_DIR / "y_test.csv"

# Scaler para padronização
SCALER_FILE = DATA_PROCESSED_DIR / "scaler.joblib"

# Feature names para reaproveitamento
FEATURE_NAMES_FILE = PROJECT_ROOT / "models" / "feature_names.txt"


# ============================================================================
# 3. MODEL FILES (Arquivos do Modelo Treinado)
# ============================================================================
# Propósito: Centralizar nomes de arquivos produzidos pelo treinamento.
# Facilita carga, avaliação e produção do modelo.

# Modelo treinado
MODEL_FILE = MODELS_DIR / "model.joblib"

# Configuração do modelo (hiperparâmetros)
CONFIG_FILE = MODELS_DIR / "config.json"

# Métricas técnicas e de negócio
METRICS_FILE = MODELS_DIR / "metrics.json"

# Feature importance (coeficientes)
FEATURE_IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"


# ============================================================================
# 4. MODEL HYPERPARAMETERS (Parâmetros do Modelo)
# ============================================================================
# Propósito: Definir e documentar hiperparâmetros em um só lugar.
# Facilita experimentação e comparação de modelos.
# 
# Racional de cada parâmetro:
# - max_iter=1000: Iterações suficientes para convergência
# - C=1.0: Força de regularização L2 (padrão balanceado)
# - solver='lbfgs': Algoritmo de otimização (bom para dataset pequeno)
# - class_weight='balanced': Penaliza classe minoritária (churn) para lidar com desbalanceamento
# - random_state=42: Reprodutibilidade (mesmos resultados toda vez)

MODEL_CONFIG = {
    "max_iter": 1000,
    "C": 1.0,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": 42
}

# Configuração de split treino/teste
# Propósito: Garantir que dados são divididos consistentemente.
# 
# Racional:
# - test_size=0.2: 20% para teste (1.409 amostras), 80% para treino (5.634 amostras)
# - random_state=42: Reprodutibilidade
# - stratify=True: Mantém proporção de churners em ambos conjuntos (essencial para dados desbalanceados)

TRAIN_TEST_SPLIT = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True
}


# ============================================================================
# 5. BUSINESS METRICS (Métricas de Negócio)
# ============================================================================
# Propósito: Quantificar impacto financeiro de cada decisão do modelo.
# Baseado em análise de negócio real com stakeholders.
#
# Racional:
# - TP (+R$400): Cliente foi retido com sucesso (receita salva)
# - FP (-R$50): Ação de retenção desnecessária (custo operacional)
# - FN (-R$500): Cliente foi perdido sem tentar (receita perdida = maior custo!)
# - TN (R$0): Não fazer nada para cliente que não churned (sem custo/benefício)
#
# Insight: FN é 10x mais caro que FP! → Priorizar RECALL

BUSINESS_METRICS = {
    "cost_true_positive": 400,       # +R$ por cliente retido
    "cost_false_positive": -50,      # -R$ por ação desnecessária
    "cost_false_negative": -500,     # -R$ por cliente perdido
    "cost_true_negative": 0          # R$0 sem custo
}


# ============================================================================
# 6. PERFORMANCE TARGETS (Metas de Desempenho)
# ============================================================================
# Propósito: Definir metas técnicas que o modelo deve atingir.
# Facilita avaliação de sucesso e comparação entre modelos.
#
# Racional:
# - AUC-ROC ≥ 0.80: Excelente capacidade de discriminação geral
# - PR-AUC ≥ 0.65: Bom desempenho focado na classe minoritária (churners)
# - Recall ≥ 0.75: Capturar pelo menos 75% dos clientes em risco
# - F2-Score ≥ 0.70: Métrica balanceada ponderando mais Recall (alinhada com negócio)

PERFORMANCE_TARGETS = {
    "auc_roc": 0.80,       # Meta conservadora mas rigorosa
    "pr_auc": 0.65,        # Para dados desbalanceados
    "recall": 0.75,        # Minimizar falsos negativos
    "f2_score": 0.70       # Métrica de negócio
}


# ============================================================================
# 7. MLFLOW CONFIGURATION (Rastreamento de Experimentos)
# ============================================================================
# Propósito: Configurar MLflow para rastrear todos os experimentos.
# Facilita reprodutibilidade e comparação entre versões de modelos.
#
# Uso:
#     mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
#     mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
#     with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
#         mlflow.log_params(MODEL_CONFIG)
#         # ... training code ...

MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "tech_challenge_churn_etapa1"
MLFLOW_RUN_NAME = "logistic_regression_final"

# Dicionário consolidado de configuração MLflow
MLFLOW_CONFIG = {
    "tracking_uri": str(MLFLOW_TRACKING_URI),
    "experiment_name": MLFLOW_EXPERIMENT_NAME,
    "run_name": MLFLOW_RUN_NAME,
}


# ============================================================================
# 8. DATA PATHS CONSOLIDATED (Dicionário com Todos os Caminhos)
# ============================================================================
# Propósito: Consolidar todos os caminhos em um dicionário único
# para facilitar acesso e documentação.

DATA_PATHS = {
    "project_root": str(PROJECT_ROOT),
    "data_raw": str(DATA_RAW_DIR),
    "data_processed": str(DATA_PROCESSED_DIR),
    "models": str(MODELS_DIR),
    "reports": str(REPORTS_DIR),
    "notebooks": str(NOTEBOOKS_DIR),
    "raw_dataset": str(RAW_DATASET),
    "x_train": str(X_TRAIN_FILE),
    "x_test": str(X_TEST_FILE),
    "y_train": str(Y_TRAIN_FILE),
    "y_test": str(Y_TEST_FILE),
    "scaler": str(SCALER_FILE),
    "model": str(MODEL_FILE),
    "config": str(CONFIG_FILE),
    "metrics": str(METRICS_FILE),
    "feature_names": str(FEATURE_NAMES_FILE),
    "feature_importance": str(FEATURE_IMPORTANCE_FILE),
}
