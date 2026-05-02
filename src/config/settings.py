"""
Arquivo de Settings - Configurações Globais do Projeto
=======================================================

Propósito:
----------
Definir TODAS as constantes e configurations do projeto.
Centralizar paths absolutos, parâmetros, e métricas de negócio.

Uso em notebooks:
    from src.config.settings import MODEL_CONFIG, BUSINESS_METRICS
"""

import os
from pathlib import Path

# ============================================================================
# 1. PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "baseline"
MODELS_DIR = PROJECT_ROOT / "models" / "baseline" / "mlp"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for dir_path in [DATA_PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# 2. DATA FILES
# ============================================================================

RAW_DATASET = DATA_RAW_DIR / "Telco-Customer-Churn.csv"

X_TRAIN_FILE = DATA_PROCESSED_DIR / "X_train.csv"
X_TEST_FILE = DATA_PROCESSED_DIR / "X_test.csv"
Y_TRAIN_FILE = DATA_PROCESSED_DIR / "y_train.csv"
Y_TEST_FILE = DATA_PROCESSED_DIR / "y_test.csv"

SCALER_FILE = DATA_PROCESSED_DIR / "scaler.joblib"
FEATURE_NAMES_FILE = PROJECT_ROOT / "models" / "feature_names.txt"

# ============================================================================
# 3. MODEL FILES
# ============================================================================

MODEL_FILE = MODELS_DIR / "model.pt"
CONFIG_FILE = MODELS_DIR / "config.json"
METRICS_FILE = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"

# ============================================================================
# 4. MODEL HYPERPARAMETERS (MLP PyTorch)
# ============================================================================
#
# - input_dim=31:           Número de features após pré-processamento
# - hidden_layers=[64, 32]: Arquitetura decrescente
# - dropout=0.3:            Regularização
# - learning_rate=1e-3:     Taxa de aprendizado Adam
# - epochs=50:              Épocas de treino
# - batch_size=32:          Mini-batch DataLoader
# - random_state=42:        Reprodutibilidade

MODEL_CONFIG = {
    "input_dim": 19,
    "hidden_layers": [128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "epochs": 80,
    "batch_size": 256,
    "random_state": 42,
    "patience": 10,  # early stopping
}

TRAIN_TEST_SPLIT = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True
}

# ============================================================================
# 5. BUSINESS METRICS
# ============================================================================

BUSINESS_METRICS = {
    "cost_true_positive": 400,
    "cost_false_positive": -50,
    "cost_false_negative": -500,
    "cost_true_negative": 0
}

# ============================================================================
# 6. PERFORMANCE TARGETS
# ============================================================================

PERFORMANCE_TARGETS = {
    "auc_roc": 0.80,
    "pr_auc": 0.65,
    "recall": 0.75,
    "f2_score": 0.70
}

# ============================================================================
# 7. MLFLOW CONFIGURATION
# ============================================================================

MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "tech_challenge_churn_etapa1"
MLFLOW_RUN_NAME = "mlp_pytorch_final"

MLFLOW_CONFIG = {
    "tracking_uri": str(MLFLOW_TRACKING_URI),
    "experiment_name": MLFLOW_EXPERIMENT_NAME,
    "run_name": MLFLOW_RUN_NAME,
}

# ============================================================================
# 8. DATA PATHS CONSOLIDATED
# ============================================================================

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