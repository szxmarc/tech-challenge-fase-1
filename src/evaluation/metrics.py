"""
Métricas técnicas e de negócio.

Funções para avaliar modelos com métricas técnicas (Accuracy, Precision, Recall,
F1, AUC-ROC, PR-AUC) e métricas de negócio (ROI baseado em custo por decisão).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.settings import BUSINESS_METRICS


def evaluate_model(y_true, y_pred, y_pred_proba=None) -> dict:
    """
    Calcula métricas técnicas do modelo.

    Args:
        y_true: Labels reais.
        y_pred: Predições de classe.
        y_pred_proba: Probabilidades preditas (necessário para AUC-ROC e PR-AUC).

    Returns:
        Dicionário com accuracy, precision, recall, f1 e, opcionalmente, auc_roc e pr_auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        probabilities = np.array(y_pred_proba)
        if probabilities.ndim > 1:
            probabilities = probabilities[:, 1]

        metrics["auc_roc"] = roc_auc_score(y_true, probabilities)

        precision_vals, recall_vals, _ = precision_recall_curve(y_true, probabilities)
        metrics["pr_auc"] = auc(recall_vals, precision_vals)

    return metrics


def calculate_business_value(y_true, y_pred) -> dict:
    """
    Calcula o valor financeiro gerado pelo modelo com base nos custos por decisão.

    Custos definidos em BUSINESS_METRICS (settings.py):
      - cost_true_positive:  receita salva por cliente retido corretamente
      - cost_false_positive: custo de ação desnecessária em cliente que não churnou
      - cost_false_negative: receita perdida por cliente que churnou sem ser identificado
      - cost_true_negative:  sem custo para previsão correta de não-churn

    Args:
        y_true: Labels reais.
        y_pred: Predições de classe.

    Returns:
        Dicionário com valor total, valor médio por cliente e contagem da matriz de confusão.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_value = (
        tp * BUSINESS_METRICS["cost_true_positive"]
        + fp * BUSINESS_METRICS["cost_false_positive"]
        + fn * BUSINESS_METRICS["cost_false_negative"]
        + tn * BUSINESS_METRICS["cost_true_negative"]
    )

    avg_value_per_customer = total_value / len(y_true)

    return {
        "total_value": total_value,
        "avg_value_per_customer": avg_value_per_customer,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
