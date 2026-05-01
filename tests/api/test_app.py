from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import _resolve_required_api_names, _save_selected_api_names, _train_on_startup, create_app
from src.prediction.feature_mapping import MODEL_FEATURE_NAMES


def test_create_app_retorna_instancia_fastapi():
    assert isinstance(create_app(), FastAPI)


def test_app_tem_titulo_configurado(app):
    assert app.title == "Churn Prediction API"


def test_app_tem_versao_configurada(app):
    assert app.version == "1.0.0"


def test_openapi_schema_disponivel(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200


def test_health_raiz_retorna_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_raiz_retorna_status_healthy(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"


def test_rota_inexistente_retorna_404(client):
    response = client.get("/rota-que-nao-existe")
    assert response.status_code == 404


def test_handler_404_retorna_json_com_erro(client):
    data = client.get("/rota-que-nao-existe").json()
    assert "error" in data
    assert data["status"] == 404


def test_startup_chama_train_on_startup():
    """Garante que o lifespan dispara _train_on_startup ao iniciar o app."""
    with patch("src.api.app._train_on_startup") as mock_train:
        with TestClient(create_app()):
            pass
    mock_train.assert_called_once()


def test_train_on_startup_executa_pipeline_completo():
    """Verifica que _train_on_startup chama todas as etapas do pipeline."""
    import pandas as pd

    scaler_mock = object()
    encoded_data = {col: [0.0, 1.0] for col in MODEL_FEATURE_NAMES}
    encoded_data["Churn"] = [0, 1]
    engineered_data = dict(encoded_data)

    with patch("src.data.loader.load_raw_data") as mock_load, \
         patch("src.data.processor.encode_features") as mock_encode, \
         patch("src.data.processor.engineer_features") as mock_engineer, \
         patch("src.data.processor.select_features") as mock_select, \
         patch("src.data.processor.split_and_scale_data") as mock_split, \
         patch("src.data.processor.save_processed_data") as mock_save_data, \
         patch("src.models.trainer.train_logistic_regression") as mock_lr, \
         patch("src.models.trainer.save_model") as mock_save_lr, \
         patch("src.models.trainer.train_mlp") as mock_mlp, \
         patch("src.models.trainer.save_mlp_model") as mock_save_mlp, \
         patch("src.evaluation.metrics.compare_models") as mock_compare, \
         patch("src.api.app._save_selected_api_names") as mock_save_names:
        mock_load.return_value = pd.DataFrame({"Churn": [0, 1]})
        mock_encode.return_value = pd.DataFrame(encoded_data)
        mock_engineer.return_value = pd.DataFrame(engineered_data)
        mock_select.return_value = MODEL_FEATURE_NAMES
        mock_split.return_value = (
            pd.DataFrame({col: [0.0] for col in MODEL_FEATURE_NAMES}),
            pd.DataFrame({col: [1.0] for col in MODEL_FEATURE_NAMES}),
            pd.Series([1]),
            pd.Series([0]),
            scaler_mock,
        )
        mock_compare.return_value = {
            "trainedAt": "2024-01-01T12:00:00",
            "nTestSamples": 1,
            "models": {
                "logisticRegression": {"metrics": {"f1Score": 0.70}, "businessValue": {"totalValue": 300}},
                "mlp": {"metrics": {"f1Score": 0.73}, "businessValue": {"totalValue": 340}},
            },
            "winner": {},
            "recommendation": "mlp",
        }
        with patch("builtins.open", create=True), \
             patch("pathlib.Path.write_text"), \
             patch("pathlib.Path.mkdir"):
            _train_on_startup()

    mock_load.assert_called_once()
    mock_encode.assert_called_once()
    mock_engineer.assert_called_once()
    mock_select.assert_called_once()
    mock_split.assert_called_once()
    mock_save_data.assert_called_once()
    mock_lr.assert_called_once()
    mock_save_lr.assert_called_once()
    mock_mlp.assert_called_once()
    mock_save_mlp.assert_called_once()
    mock_compare.assert_called_once()
    mock_save_names.assert_called_once()


def test_resolve_required_api_names_features_diretas():
    """Features diretas (sem derivadas) retornam seus nomes camelCase."""
    result = _resolve_required_api_names(["TotalCharges", "tenure", "MonthlyCharges"])
    assert "totalCharges" in result
    assert "tenureMonths" in result
    assert "monthlyCharges" in result


def test_resolve_required_api_names_feature_derivada_expande_fontes():
    """Feature derivada é expandida para suas features-fonte."""
    result = _resolve_required_api_names(["is_monthly"])
    assert "contractOneYear" in result
    assert "contractTwoYear" in result


def test_resolve_required_api_names_deduplicacao():
    """Features-fonte compartilhadas entre derivadas aparecem uma só vez."""
    result = _resolve_required_api_names(["service_count", "has_no_services"])
    assert result.count("onlineSecurityActive") == 1
