from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluation.metrics import calculate_business_value, compare_models, evaluate_model

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


# ── compare_models ────────────────────────────────────────────────────────────

@pytest.fixture
def two_models():
    """Par de modelos mock (LR e MLP) com predict e predict_proba."""
    def _make_model(pred, proba_churn):
        m = MagicMock()
        m.predict.return_value = np.array(pred)
        proba = np.column_stack([1 - np.array(proba_churn), np.array(proba_churn)])
        m.predict_proba.return_value = proba
        return m

    lr = _make_model(Y_PRED, Y_PROBA)
    mlp_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    mlp_proba = np.array([0.05, 0.95, 0.60, 0.15, 0.85, 0.45, 0.75, 0.25])
    mlp = _make_model(mlp_pred, mlp_proba)
    return lr, mlp


def test_compare_models_retorna_chaves_esperadas(two_models):
    lr, mlp = two_models
    result = compare_models(lr, mlp, Y_TRUE, Y_TRUE)
    for key in ["trainedAt", "nTestSamples", "models", "winner", "recommendation"]:
        assert key in result


def test_compare_models_contem_ambos_modelos(two_models):
    lr, mlp = two_models
    result = compare_models(lr, mlp, Y_TRUE, Y_TRUE)
    assert "logisticRegression" in result["models"]
    assert "mlp" in result["models"]


def test_compare_models_metricas_completas(two_models):
    lr, mlp = two_models
    result = compare_models(lr, mlp, Y_TRUE, Y_TRUE)
    for model_key in ["logisticRegression", "mlp"]:
        metrics = result["models"][model_key]["metrics"]
        for m in ["accuracy", "precision", "recall", "f1Score", "aucRoc", "prAuc"]:
            assert m in metrics
            assert 0.0 <= metrics[m] <= 1.0


def test_compare_models_business_value_completo(two_models):
    lr, mlp = two_models
    result = compare_models(lr, mlp, Y_TRUE, Y_TRUE)
    for model_key in ["logisticRegression", "mlp"]:
        biz = result["models"][model_key]["businessValue"]
        for k in ["totalValue", "avgValuePerCustomer", "truePositives",
                  "falsePositives", "falseNegatives", "trueNegatives"]:
            assert k in biz


def test_compare_models_winner_por_metrica(two_models):
    lr, mlp = two_models
    result = compare_models(lr, mlp, Y_TRUE, Y_TRUE)
    valid = {"logisticRegression", "mlp", "tie"}
    for metric, winner in result["winner"].items():
        assert winner in valid


def test_compare_models_recomendacao_valida(two_models):
    lr, mlp = two_models
    result = compare_models(lr, mlp, Y_TRUE, Y_TRUE)
    assert result["recommendation"] in {"logisticRegression", "mlp"}


def test_compare_models_n_test_samples(two_models):
    lr, mlp = two_models
    result = compare_models(lr, mlp, Y_TRUE, Y_TRUE)
    assert result["nTestSamples"] == len(Y_TRUE)
