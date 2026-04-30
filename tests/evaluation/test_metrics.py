import numpy as np
import pytest

from src.evaluation.metrics import calculate_business_value, evaluate_model

Y_TRUE = np.array([0, 1, 1, 0, 1, 0, 1, 0])
Y_PRED = np.array([0, 1, 0, 0, 1, 1, 1, 0])
Y_PROBA = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.7, 0.3])


def test_evaluate_model_retorna_metricas_basicas():
    result = evaluate_model(Y_TRUE, Y_PRED)
    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1_score" in result


def test_evaluate_model_sem_proba_nao_tem_auc():
    result = evaluate_model(Y_TRUE, Y_PRED)
    assert "auc_roc" not in result
    assert "pr_auc" not in result


def test_evaluate_model_com_proba_tem_auc():
    result = evaluate_model(Y_TRUE, Y_PRED, y_pred_proba=Y_PROBA)
    assert "auc_roc" in result
    assert "pr_auc" in result


def test_evaluate_model_metricas_entre_0_e_1():
    result = evaluate_model(Y_TRUE, Y_PRED)
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        assert 0.0 <= result[key] <= 1.0


def test_evaluate_model_aceita_proba_2d():
    proba_2d = np.column_stack([1 - Y_PROBA, Y_PROBA])
    result = evaluate_model(Y_TRUE, Y_PRED, y_pred_proba=proba_2d)
    assert "auc_roc" in result


def test_calculate_business_value_retorna_chaves_esperadas():
    result = calculate_business_value(Y_TRUE, Y_PRED)
    for key in ["total_value", "avg_value_per_customer", "true_positives",
                "false_positives", "false_negatives", "true_negatives"]:
        assert key in result


def test_calculate_business_value_previsao_perfeita():
    y = np.array([1, 0, 1, 0])
    result = calculate_business_value(y, y)
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["true_positives"] == 2
    assert result["true_negatives"] == 2


def test_calculate_business_value_media_por_cliente():
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    result = calculate_business_value(y_true, y_pred)
    assert result["avg_value_per_customer"] == result["total_value"] / 2


def test_calculate_business_value_retorna_inteiros_na_matriz():
    result = calculate_business_value(Y_TRUE, Y_PRED)
    assert isinstance(result["true_positives"], int)
    assert isinstance(result["false_positives"], int)
