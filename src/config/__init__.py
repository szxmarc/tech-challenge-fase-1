"""
Módulo de Configuração do Projeto
==================================

Propósito:
----------
Centralizar TODAS as configurações do projeto em um único lugar.
Evita hard-coding, facilita manutenção e reprodutibilidade.

Inclui:
- Caminhos (paths) de diretórios
- Parâmetros de modelos
- Configurações de dados
- Métricas de negócio
- Targets de desempenho
- Configuração MLflow

Uso:
----
from src.config import MODEL_CONFIG, BUSINESS_METRICS, MODELS_DIR
"""

from .settings import (
    # Paths
    PROJECT_ROOT,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
    
    # Files
    RAW_DATASET,
    FEATURE_NAMES_FILE,
    SCALER_FILE,
    MODEL_FILE,
    CONFIG_FILE,
    METRICS_FILE,
    FEATURE_IMPORTANCE_FILE,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    
    # Model parameters
    MODEL_CONFIG,
    TRAIN_TEST_SPLIT,
    
    # Business metrics
    BUSINESS_METRICS,
    PERFORMANCE_TARGETS,
    
    # MLflow
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_RUN_NAME,
)

__all__ = [
    # Paths
    'PROJECT_ROOT',
    'DATA_RAW_DIR',
    'DATA_PROCESSED_DIR',
    'MODELS_DIR',
    'REPORTS_DIR',
    'NOTEBOOKS_DIR',
    
    # Files
    'RAW_DATASET',
    'FEATURE_NAMES_FILE',
    'SCALER_FILE',
    'MODEL_FILE',
    'CONFIG_FILE',
    'METRICS_FILE',
    'FEATURE_IMPORTANCE_FILE',
    'X_TRAIN_FILE',
    'X_TEST_FILE',
    'Y_TRAIN_FILE',
    'Y_TEST_FILE',
    
    # Model parameters
    'MODEL_CONFIG',
    'TRAIN_TEST_SPLIT',
    
    # Business metrics
    'BUSINESS_METRICS',
    'PERFORMANCE_TARGETS',
    
    # MLflow
    'MLFLOW_TRACKING_URI',
    'MLFLOW_EXPERIMENT_NAME',
    'MLFLOW_RUN_NAME',
]
