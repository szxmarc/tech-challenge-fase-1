"""
📊 Cálculo de Métricas Técnicas e de Negócio

Funções para avaliar modelos com métricas técnicas
(Accuracy, Precision, Recall, F1, AUC) e métricas de negócio (ROI, valor total).
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from src.config.settings import BUSINESS_METRICS
import pandas as pd
import numpy as np


def evaluate_model(y_true, y_pred, y_pred_proba=None) -> dict:
    """
    Calcula métricas técnicas do modelo.
    
    Args:
        y_true: Labels reais
        y_pred: Predições (class)
        y_pred_proba: Probabilidades preditas (para AUC)
    
    Returns:
        Dicionário com métricas
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    
    if y_pred_proba is not None:
        y_pred_proba = np.array(y_pred_proba)
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]
        
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics["pr_auc"] = auc(recall, precision)
    
    return metrics


def calculate_business_value(y_true, y_pred) -> dict:
    """
    Calcula métricas de negócio baseadas em custos/ganhos.
    
    Args:
        y_true: Labels reais
        y_pred: Predições
    
    Returns:
        Dicionário com métricas de negócio
    """
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_value = (
        tp * BUSINESS_METRICS["TP"] +
        fp * BUSINESS_METRICS["FP"] +
        fn * BUSINESS_METRICS["FN"] +
        tn * BUSINESS_METRICS["TN"]
    )
    
    n_customers = len(y_true)
    avg_value_per_customer = total_value / n_customers
    
    return {
        "total_value": total_value,
        "avg_value_per_customer": avg_value_per_customer,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
