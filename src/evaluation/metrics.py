"""
Métricas técnicas e de negócio.

Funções para avaliar modelos com métricas técnicas (Accuracy, Precision, Recall,
F1, AUC-ROC, PR-AUC) e métricas de negócio (ROI baseado em custo por decisão).
"""

from datetime import datetime

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


def compare_models(lr_model, mlp_model, X_test, y_test) -> dict:
    """
    Avalia ambos os modelos no conjunto de teste e retorna tabela comparativa.

    Args:
        lr_model: Modelo de Regressão Logística treinado.
        mlp_model: Modelo MLP treinado.
        X_test:   Features de teste (já escalonadas).
        y_test:   Target de teste.

    Returns:
        Dicionário com métricas técnicas, valor de negócio, vencedor por métrica
        e recomendação final baseada no valor de negócio total.
    """
    _METRIC_LABELS = ["accuracy", "precision", "recall", "f1Score", "aucRoc", "prAuc"]
    _INTERNAL_KEYS = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "pr_auc"]
    _CAMEL_MAP = dict(zip(_INTERNAL_KEYS, _METRIC_LABELS))

    results: dict = {}
    for name, model in [("logisticRegression", lr_model), ("mlp", mlp_model)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        raw = evaluate_model(y_test, y_pred, y_proba)
        biz = calculate_business_value(y_test, y_pred)

        results[name] = {
            "metrics": {
                _CAMEL_MAP[k]: round(float(raw[k]), 4)
                for k in _INTERNAL_KEYS
            },
            "businessValue": {
                "totalValue": int(biz["total_value"]),
                "avgValuePerCustomer": round(float(biz["avg_value_per_customer"]), 2),
                "truePositives": biz["true_positives"],
                "falsePositives": biz["false_positives"],
                "falseNegatives": biz["false_negatives"],
                "trueNegatives": biz["true_negatives"],
            },
        }

    winner: dict = {}
    for metric in _METRIC_LABELS:
        lr_val = results["logisticRegression"]["metrics"][metric]
        mlp_val = results["mlp"]["metrics"][metric]
        if lr_val > mlp_val:
            winner[metric] = "logisticRegression"
        elif mlp_val > lr_val:
            winner[metric] = "mlp"
        else:
            winner[metric] = "tie"

    lr_biz = results["logisticRegression"]["businessValue"]["totalValue"]
    mlp_biz = results["mlp"]["businessValue"]["totalValue"]
    recommendation = "mlp" if mlp_biz >= lr_biz else "logisticRegression"

    return {
        "trainedAt": datetime.now().isoformat(timespec="seconds"),
        "nTestSamples": int(len(y_test)),
        "models": results,
        "winner": winner,
        "recommendation": recommendation,
    }
