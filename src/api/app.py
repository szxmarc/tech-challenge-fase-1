"""
Factory da aplicação FastAPI.

Cria e configura a instância da aplicação, inclui o router da API
e define o handler de erro para rotas não encontradas.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Churn Prediction API",
        version="1.0.0",
        description="API de predição de churn para clientes Telco - Tech Challenge Fase 1",
    )

    @app.get("/health")
    def health():
        return {"status": "healthy", "message": "API operacional"}

    from .routes import api_router
    app.include_router(api_router, prefix="/api/v1")

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            {"error": "Endpoint não encontrado", "status": 404},
            status_code=404,
        )

    logging.getLogger(__name__).info("Aplicação iniciada")
    return app
