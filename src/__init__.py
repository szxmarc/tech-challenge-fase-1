"""
✨ Tech Challenge FIAP - Fase 1

Solução de Predição de Churn em Telecomunicações com Flask REST API.

Módulos Principais:
- config: Configurações centralizadas (settings.py)
- data: Carregamento e processamento de dados (loader.py, processor.py)
- models: Treinamento e persistência de modelos (trainer.py)
- evaluation: Métricas e análise de features (metrics.py, features.py)
- visualization: Gráficos e visualizações (plots.py)
- utils: Utilitários gerais (helpers.py)
- api: Aplicação Flask REST (app.py, routes.py)

Uso Rápido:
    # Usar como aplicação ML
    from src.data.loader import load_raw_data
    from src.models.trainer import train_logistic_regression
    
    # Iniciar API Flask
    from src.api.app import create_app
    app = create_app()
    app.run()
"""

__version__ = "1.0.0"
__author__ = "FIAP Tech Challenge"

# ============================================================================
# Importações Seguras (sem dependências externas)
# ============================================================================
from src.config.settings import MODEL_CONFIG, BUSINESS_METRICS, DATA_PATHS

__all__ = [
    # Config
    "MODEL_CONFIG",
    "BUSINESS_METRICS",
    "DATA_PATHS",
    # Importações opcionais (lazy) - importe conforme necessário:
    # from src.data.loader import load_raw_data, load_processed_data
    # from src.data.processor import encode_features, split_and_scale_data
    # from src.models.trainer import train_logistic_regression, save_model, load_model
    # from src.evaluation.metrics import evaluate_model, calculate_business_value
    # from src.evaluation.features import get_feature_importance
    # from src.api.app import create_app
]
