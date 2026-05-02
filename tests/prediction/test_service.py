"""
Testes para src/prediction/service.py
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.prediction.service import predict_batch, predict_single


@pytest.fixture
def patched_deps(mock_model, mock_scaler, mock_imputer):
    """Injeta modelo, scaler e imputer mockados nas dependências do serviço."""
    with patch("src.prediction.service.get_model", return_value=(mock_model, mock_scaler, mock_imputer)):
        yield


# ── predict_single ────────────────────────────────────────────────────────────

def test_predict_single_retorna_chaves_esperadas(sample_input, patched_deps):
    result = predict_single(sample_input)
    assert "prediction" in result
    assert "predictionLabel" in result
    assert "probabilityChurn" in result
    assert "probabilityNoChurn" in result
    assert "confidence" in result


def test_predict_single_label_churn(sample_input, patched_deps):
    result = predict_single(sample_input)
    assert result["prediction"] == 1
    assert result["predictionLabel"] == "Churn"


def test_predict_single_label_nao_churn(sample_input, mock_scaler, mock_imputer):
    mock_model_nao_churn = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    mock_model_nao_churn.return_value = torch.tensor([[-2.0]])
    with patch("src.prediction.service.get_model", return_value=(mock_model_nao_churn, mock_scaler, mock_imputer)):
        result = predict_single(sample_input)
    assert result["prediction"] == 0


def test_predict_single_probabilidades_em_porcentagem(sample_input, patched_deps):
    result = predict_single(sample_input)
    assert result["probabilityChurn"].endswith("%")
    assert result["probabilityNoChurn"].endswith("%")
    assert result["confidence"].endswith("%")


def test_predict_single_probabilidades_somam_100(sample_input, patched_deps):
    result = predict_single(sample_input)
    churn = float(result["probabilityChurn"].rstrip("%"))
    no_churn = float(result["probabilityNoChurn"].rstrip("%"))
    assert abs(churn + no_churn - 100.0) < 0.2


# ── predict_batch ─────────────────────────────────────────────────────────────

def test_predict_batch_retorna_lista(sample_input, patched_deps):
    result = predict_batch([sample_input, sample_input])
    assert isinstance(result, list)
    assert len(result) == 2


def test_predict_batch_preserva_indice_do_cliente(sample_input, patched_deps):
    result = predict_batch([sample_input, sample_input])
    assert result[0]["customerIndex"] == 0
    assert result[1]["customerIndex"] == 1


def test_predict_batch_captura_erros_individuais(patched_deps):
    result = predict_batch([{"gender": "Male"}])
    assert "error" in result[0]
    assert result[0]["customerIndex"] == 0


def test_predict_batch_lista_vazia_retorna_lista_vazia(patched_deps):
    result = predict_batch([])
    assert result == []