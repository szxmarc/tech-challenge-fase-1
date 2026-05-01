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
def patch_cache(mock_model, mock_scaler):
    """Injeta modelo e features mockados no cache de dependências."""
    from tests.conftest import MOCK_FEATURE_NAMES
    deps._model = mock_model
    deps._scaler = mock_scaler
    deps._feature_names = MOCK_FEATURE_NAMES


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
    with patch("src.api.dependencies.joblib.load", side_effect=FileNotFoundError):
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


def test_model_info_n_features_correto(client):
    from tests.conftest import MOCK_FEATURE_NAMES
    data = client.get("/api/v1/model/info").json()
    assert data["nFeatures"] == len(MOCK_FEATURE_NAMES)


def test_model_info_retorna_500_em_erro(client):
    with patch("src.api.routes.get_model", side_effect=RuntimeError("falha")):
        response = client.get("/api/v1/model/info")
    assert response.status_code == 500


# ── /api/v1/features ──────────────────────────────────────────────────────────

def test_features_retorna_200(client):
    assert client.get("/api/v1/features").status_code == 200


def test_features_retorna_lista_correta(client):
    from tests.conftest import MOCK_FEATURE_NAMES
    data = client.get("/api/v1/features").json()
    assert data["nFeatures"] == len(MOCK_FEATURE_NAMES)
    assert data["features"] == MOCK_FEATURE_NAMES


def test_features_retorna_500_em_erro(client):
    with patch("src.api.routes.get_feature_names", side_effect=RuntimeError("falha")):
        response = client.get("/api/v1/features")
    assert response.status_code == 500


# ── /api/v1/model/comparison ─────────────────────────────────────────────────

MOCK_COMPARISON = {
    "trainedAt": "2024-01-01T12:00:00",
    "nTestSamples": 1409,
    "models": {
        "logisticRegression": {
            "metrics": {"accuracy": 0.78, "precision": 0.62, "recall": 0.81,
                        "f1Score": 0.70, "aucRoc": 0.85, "prAuc": 0.72},
            "businessValue": {"totalValue": 300000, "avgValuePerCustomer": 212.9,
                              "truePositives": 280, "falsePositives": 50,
                              "falseNegatives": 20, "trueNegatives": 1059},
        },
        "mlp": {
            "metrics": {"accuracy": 0.82, "precision": 0.68, "recall": 0.79,
                        "f1Score": 0.73, "aucRoc": 0.88, "prAuc": 0.76},
            "businessValue": {"totalValue": 340000, "avgValuePerCustomer": 241.3,
                              "truePositives": 290, "falsePositives": 40,
                              "falseNegatives": 10, "trueNegatives": 1069},
        },
    },
    "winner": {"accuracy": "mlp", "precision": "mlp", "recall": "logisticRegression",
               "f1Score": "mlp", "aucRoc": "mlp", "prAuc": "mlp"},
    "recommendation": "mlp",
}


def test_comparison_retorna_404_sem_arquivo(client, tmp_path):
    with patch("src.api.routes.DATA_PATHS", {"comparison": str(tmp_path / "nao_existe.json")}):
        response = client.get("/api/v1/model/comparison")
    assert response.status_code == 404


def test_comparison_retorna_200_com_arquivo(client, tmp_path):
    import json
    comparison_file = tmp_path / "comparison.json"
    comparison_file.write_text(json.dumps(MOCK_COMPARISON), encoding="utf-8")
    with patch("src.api.routes.DATA_PATHS", {"comparison": str(comparison_file)}):
        response = client.get("/api/v1/model/comparison")
    assert response.status_code == 200


def test_comparison_contem_chaves_esperadas(client, tmp_path):
    import json
    comparison_file = tmp_path / "comparison.json"
    comparison_file.write_text(json.dumps(MOCK_COMPARISON), encoding="utf-8")
    with patch("src.api.routes.DATA_PATHS", {"comparison": str(comparison_file)}):
        data = client.get("/api/v1/model/comparison").json()
    for key in ["trainedAt", "nTestSamples", "models", "winner", "recommendation"]:
        assert key in data


def test_comparison_contem_ambos_modelos(client, tmp_path):
    import json
    comparison_file = tmp_path / "comparison.json"
    comparison_file.write_text(json.dumps(MOCK_COMPARISON), encoding="utf-8")
    with patch("src.api.routes.DATA_PATHS", {"comparison": str(comparison_file)}):
        data = client.get("/api/v1/model/comparison").json()
    assert "logisticRegression" in data["models"]
    assert "mlp" in data["models"]


# ── /api/v1/predict ───────────────────────────────────────────────────────────

def test_predict_retorna_200_com_input_valido(client, sample_input):
    with patch("src.api.routes.predict_single", return_value=MOCK_RESULT):
        response = client.post("/api/v1/predict", json=sample_input)
    assert response.status_code == 200


def test_predict_retorna_422_sem_json(client):
    response = client.post("/api/v1/predict", data="texto")
    assert response.status_code == 422


def test_predict_retorna_422_com_body_incompleto(client):
    response = client.post("/api/v1/predict", json={"gender": "male"})
    assert response.status_code == 422


def test_predict_retorna_400_quando_service_levanta_erro(client, sample_input):
    with patch("src.api.routes.predict_single", side_effect=ValueError("Features ausentes")):
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


def test_predict_batch_retorna_422_customers_nao_e_lista(client):
    response = client.post("/api/v1/predict/batch", json={"customers": "invalido"})
    assert response.status_code == 422


def test_predict_batch_retorna_422_sem_json(client):
    response = client.post("/api/v1/predict/batch", data="texto")
    assert response.status_code == 422


def test_predict_batch_retorna_500_em_erro_interno(client, sample_input):
    with patch("src.api.routes.predict_batch", side_effect=RuntimeError("falha")):
        response = client.post("/api/v1/predict/batch", json={"customers": [sample_input]})
    assert response.status_code == 500


def test_predict_batch_total_corresponde_ao_input(client, sample_input):
    batch_result = [{**MOCK_RESULT, "customerIndex": i} for i in range(2)]
    with patch("src.api.routes.predict_batch", return_value=batch_result):
        data = client.post(
            "/api/v1/predict/batch",
            json={"customers": [sample_input, sample_input]},
        ).json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2
