"""
Controllers HTTP da API.

Cada função de rota tem responsabilidade única: parse do request,
delegação ao serviço de predição e formatação da resposta HTTP.
Nenhuma lógica de negócio ou acesso ao modelo reside aqui.

Endpoints registrados em /api/v1:
  GET  /health         Verifica saúde da API e disponibilidade do modelo
  GET  /model/info     Metadados do modelo carregado
  GET  /features       Lista de features esperadas (nomes camelCase)
  POST /predict        Predição de churn para um único cliente
  POST /predict/batch  Predição de churn para múltiplos clientes
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.dependencies import get_feature_names, get_model
from src.api.schemas import BatchPredictRequest, CustomerFeatures
from src.config.settings import DATA_PATHS
from src.prediction.service import predict_batch, predict_single

api_router = APIRouter()
logger = logging.getLogger(__name__)


@api_router.get("/health")
def health():
    try:
        get_model()
        return {"status": "healthy", "message": "API e modelo operacionais"}
    except Exception as exc:
        return JSONResponse({"status": "unhealthy", "error": str(exc)}, status_code=503)


@api_router.get("/model/info")
def model_info():
    try:
        model, _ = get_model()
        feature_names = get_feature_names()
        return {
            "modelType": type(model).__name__,
            "nFeatures": len(feature_names),
            "nClasses": len(model.classes_),
            "classes": model.classes_.tolist(),
        }
    except Exception as exc:
        logger.error("Erro ao obter info do modelo: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@api_router.get("/features")
def features():
    try:
        feature_names = get_feature_names()
        return {"nFeatures": len(feature_names), "features": feature_names}
    except Exception as exc:
        logger.error("Erro ao listar features: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@api_router.get("/model/comparison")
def model_comparison():
    comparison_path = Path(DATA_PATHS["comparison"])
    if not comparison_path.exists():
        return JSONResponse(
            {"error": "Comparação não disponível. Aguarde o treinamento inicial."},
            status_code=404,
        )
    try:
        return json.loads(comparison_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Erro ao ler comparação: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@api_router.post("/predict")
def predict(body: CustomerFeatures):
    try:
        result = predict_single(body.model_dump())
        return result
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        logger.error("Erro na predição: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@api_router.post("/predict/batch")
def predict_batch_endpoint(body: BatchPredictRequest):
    try:
        customers = [c.model_dump() for c in body.customers]
        predictions = predict_batch(customers)
        return {"total": len(body.customers), "predictions": predictions}
    except Exception as exc:
        logger.error("Erro na predição em batch: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
