"""
📉 MÓDULO VISUALIZATION - Geração de Gráficos

Responsabilidade: Centralizar funções de plotagem para reutilização
em diferentes notebooks e análises.

Arquivo Principal:
- plots.py: Funções de visualização (ROC, Precision-Recall, Confusion Matrix)

Exemplo de uso:
    from src.visualization.plots import plot_roc_curve, plot_confusion_matrix, plot_feature_importance
"""

from .plots import plot_roc_curve, plot_confusion_matrix, plot_feature_importance

__all__ = ["plot_roc_curve", "plot_confusion_matrix", "plot_feature_importance"]
