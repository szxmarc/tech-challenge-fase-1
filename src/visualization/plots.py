"""
📊 Funções de Visualização

Gráficos para análise de modelos (ROC, Precision-Recall, Confusion Matrix, etc).
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import numpy as np


def plot_roc_curve(y_true, y_pred_proba, ax=None):
    """
    Plota curva ROC.
    
    Args:
        y_true: Labels reais
        y_pred_proba: Probabilidades preditas
        ax: Axes do matplotlib (opcional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    return ax


def plot_confusion_matrix(y_true, y_pred, ax=None):
    """
    Plota matriz de confusão.
    
    Args:
        y_true: Labels reais
        y_pred: Predições
        ax: Axes do matplotlib (opcional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    return ax


def plot_feature_importance(feature_importance_df, top_n=10, ax=None):
    """
    Plota gráfico de importância das features.
    
    Args:
        feature_importance_df: DataFrame com features e importâncias
        top_n: Número de features a mostrar
        ax: Axes do matplotlib (opcional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    top_features = feature_importance_df.head(top_n)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    return ax
