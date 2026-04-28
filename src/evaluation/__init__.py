"""
📈 MÓDULO EVALUATION - Métricas e Análise

Responsabilidade: Calcular métricas técnicas e de negócio,
analisar importância de features.

Arquivos:
- metrics.py: Cálculo de métricas (Accuracy, Precision, Recall, F1, AUC, etc)
- features.py: Análise de feature importance

Exemplo de uso:
    from src.evaluation.metrics import evaluate_model, calculate_business_value
    from src.evaluation.features import get_feature_importance, rank_features
"""

from .metrics import evaluate_model, calculate_business_value
from .features import get_feature_importance, rank_features

__all__ = [
    "evaluate_model",
    "calculate_business_value",
    "get_feature_importance",
    "rank_features",
]
