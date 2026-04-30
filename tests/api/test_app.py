from fastapi import FastAPI

from src.api.app import create_app


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
