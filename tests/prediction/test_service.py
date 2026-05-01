from unittest.mock import patch

import numpy as np
import pytest

from src.prediction.service import _normalize_gender, predict_batch, predict_single
from tests.conftest import MOCK_FEATURE_NAMES


@pytest.fixture
def patched_deps(mock_model, mock_scaler):
    """Injeta modelo e features mockados nas dependências do serviço."""
    with patch("src.prediction.service.get_model", return_value=(mock_model, mock_scaler)), \
         patch("src.prediction.service.get_feature_names", return_value=MOCK_FEATURE_NAMES):
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


def test_predict_single_label_nao_churn(sample_input, mock_model, mock_scaler):
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.80, 0.20]])
    with patch("src.prediction.service.get_model", return_value=(mock_model, mock_scaler)), \
         patch("src.prediction.service.get_feature_names", return_value=MOCK_FEATURE_NAMES):
        result = predict_single(sample_input)
    assert result["prediction"] == 0
    assert result["predictionLabel"] == "Não Churn"


def test_predict_single_probabilidades_em_porcentagem(sample_input, patched_deps):
    result = predict_single(sample_input)
    assert result["probabilityChurn"].endswith("%")
    assert result["probabilityNoChurn"].endswith("%")
    assert result["confidence"].endswith("%")


def test_predict_single_probabilidades_somam_100(sample_input, patched_deps):
    result = predict_single(sample_input)
    churn = float(result["probabilityChurn"].rstrip("%"))
    no_churn = float(result["probabilityNoChurn"].rstrip("%"))
    assert abs(churn + no_churn - 100.0) < 0.1


def test_predict_single_levanta_erro_features_ausentes(patched_deps):
    with pytest.raises(ValueError, match="Features ausentes"):
        predict_single({"gender": "male"})


def test_predict_single_levanta_erro_tipo_invalido(sample_input, patched_deps):
    sample_input["monthlyCharges"] = "invalido"
    with pytest.raises(ValueError, match="Tipo de dado inválido"):
        predict_single(sample_input)


def test_predict_single_aceita_booleanos(sample_input, patched_deps):
    sample_input["isSeniorCitizen"] = True
    sample_input["hasPartner"] = False
    result = predict_single(sample_input)
    assert "prediction" in result


def test_predict_single_aceita_inteiros(sample_input, patched_deps):
    sample_input["isSeniorCitizen"] = 0
    sample_input["hasPartner"] = 1
    result = predict_single(sample_input)
    assert "prediction" in result


def test_predict_single_aceita_gender_female(sample_input, patched_deps):
    sample_input["gender"] = "female"
    result = predict_single(sample_input)
    assert "prediction" in result


def test_normalize_gender_male_retorna_1():
    assert _normalize_gender({"gender": "male"})["gender"] == 1


def test_normalize_gender_female_retorna_0():
    assert _normalize_gender({"gender": "female"})["gender"] == 0


def test_normalize_gender_case_insensitive():
    assert _normalize_gender({"gender": "Male"})["gender"] == 1
    assert _normalize_gender({"gender": "FEMALE"})["gender"] == 0


def test_normalize_gender_invalido_levanta_erro():
    with pytest.raises(ValueError, match="Valor inválido para gender"):
        _normalize_gender({"gender": "other"})


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
    result = predict_batch([{"gender": "male"}])  # features ausentes
    assert "error" in result[0]
    assert result[0]["customerIndex"] == 0


def test_predict_batch_processa_parcialmente(sample_input, patched_deps):
    result = predict_batch([sample_input, {"gender": "male"}])
    assert "prediction" in result[0]
    assert "error" in result[1]


def test_predict_batch_lista_vazia_retorna_lista_vazia(patched_deps):
    result = predict_batch([])
    assert result == []
