"""
Testes para src/api/routes.py
"""

from unittest.mock import patch

import pytest

import src.api.dependencies as deps

MOCK_RESULT = {
    "prediction": 1,
    "predictionLabel": "Churn",
    "probabilityNoChurn": "32.0%",
    "probabilityChurn": "68.0%",
    "confidence": "68.0%",
}


@pytest.fixture(autouse=True)
def patch_cache(mock_model, mock_scaler, mock_imputer):
    """Injeta modelo mockado no cache de dependências."""
    deps._model = mock_model
    deps._scaler = mock_scaler
    deps._imputer = mock_imputer


# ── /api/v1/health ────────────────────────────────────────────────────────────

def test_health_retorna_200(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_retorna_status_healthy(client):
    data = client.get("/api/v1/health").json()
    assert data["status"] == "healthy"


def test_health_retorna_503_quando_modelo_falha(client):
    deps._model = None
    deps._scaler = None
    deps._imputer = None
    with patch("src.api.dependencies._load_model", side_effect=FileNotFoundError):
        response = client.get("/api/v1/health")
    assert response.status_code == 503


# ── /api/v1/model/info ────────────────────────────────────────────────────────

def test_model_info_retorna_200(client):
    assert client.get("/api/v1/model/info").status_code == 200


def test_model_info_contem_campos_esperados(client):
    data = client.get("/api/v1/model/info").json()
    assert "modelType" in data
    assert "nFeatures" in data
    assert "nClasses" in data
    assert "classes" in data


def test_model_info_n_classes_correto(client):
    data = client.get("/api/v1/model/info").json()
    assert data["nClasses"] == 2
    assert data["classes"] == [0, 1]


# ── /api/v1/features ──────────────────────────────────────────────────────────

def test_features_retorna_200(client):
    assert client.get("/api/v1/features").status_code == 200


def test_features_retorna_19_features(client):
    data = client.get("/api/v1/features").json()
    assert data["nFeatures"] == 19


# ── /api/v1/predict ───────────────────────────────────────────────────────────

def test_predict_retorna_200_com_input_valido(client, sample_input):
    with patch("src.api.routes.predict_single", return_value=MOCK_RESULT):
        response = client.post("/api/v1/predict", json=sample_input)
    assert response.status_code == 200


def test_predict_retorna_422_sem_json(client):
    response = client.post("/api/v1/predict", data="texto")
    assert response.status_code == 422


def test_predict_retorna_422_com_body_incompleto(client):
    response = client.post("/api/v1/predict", json={"gender": "Male"})
    assert response.status_code == 422


def test_predict_retorna_422_com_valor_invalido(client, sample_input):
    sample_input["gender"] = "Alien"
    response = client.post("/api/v1/predict", json=sample_input)
    assert response.status_code == 422


def test_predict_retorna_400_quando_service_levanta_erro(client, sample_input):
    with patch("src.api.routes.predict_single", side_effect=ValueError("erro")):
        response = client.post("/api/v1/predict", json=sample_input)
    assert response.status_code == 400


def test_predict_resposta_contem_campos_esperados(client, sample_input):
    with patch("src.api.routes.predict_single", return_value=MOCK_RESULT):
        data = client.post("/api/v1/predict", json=sample_input).json()
    for key in ["prediction", "predictionLabel", "probabilityChurn", "probabilityNoChurn", "confidence"]:
        assert key in data


def test_predict_retorna_500_em_erro_interno(client, sample_input):
    with patch("src.api.routes.predict_single", side_effect=RuntimeError("falha")):
        response = client.post("/api/v1/predict", json=sample_input)
    assert response.status_code == 500


# ── /api/v1/predict/batch ─────────────────────────────────────────────────────

def test_predict_batch_retorna_200(client, sample_input):
    batch_result = [{**MOCK_RESULT, "customerIndex": 0}]
    with patch("src.api.routes.predict_batch", return_value=batch_result):
        response = client.post("/api/v1/predict/batch", json={"customers": [sample_input]})
    assert response.status_code == 200


def test_predict_batch_retorna_422_sem_chave_customers(client):
    response = client.post("/api/v1/predict/batch", json={"data": []})
    assert response.status_code == 422


def test_predict_batch_total_corresponde_ao_input(client, sample_input):
    batch_result = [{**MOCK_RESULT, "customerIndex": i} for i in range(2)]
    with patch("src.api.routes.predict_batch", return_value=batch_result):
        data = client.post(
            "/api/v1/predict/batch",
            json={"customers": [sample_input, sample_input]},
        ).json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2