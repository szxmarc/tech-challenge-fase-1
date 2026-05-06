"""
Módulo de avaliação — métricas técnicas e de negócio.

Pacotes internos:
- metrics.py:  accuracy, precision, recall, F1, AUC-ROC, PR-AUC e valor financeiro
- features.py: importância e ranking de features
"""

from .features import get_feature_importance, rank_features
from .metrics import calculate_business_value, evaluate_model

__all__ = [
    "evaluate_model",
    "calculate_business_value",
    "get_feature_importance",
    "rank_features",
]
