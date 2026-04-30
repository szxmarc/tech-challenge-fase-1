"""
Módulo de visualização — geração de gráficos e análises visuais.

Responsabilidades:
  plots.py  Curva ROC, matriz de confusão e importância de features
"""

from .plots import plot_confusion_matrix, plot_feature_importance, plot_roc_curve

__all__ = ["plot_roc_curve", "plot_confusion_matrix", "plot_feature_importance"]
