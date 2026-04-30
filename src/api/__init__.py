"""
Módulo API — aplicação FastAPI com endpoints REST.

Pacotes internos:
- app.py:          factory function e setup da aplicação FastAPI
- routes.py:       controllers HTTP (endpoints)
- dependencies.py: carregamento e cache de modelo e features
- schemas.py:      modelos Pydantic para validação de request/response
"""

from .app import create_app
from .routes import api_router

__all__ = ["create_app", "api_router"]
