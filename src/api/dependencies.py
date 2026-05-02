"""
Dependências compartilhadas da API.

Centraliza o carregamento e cache do modelo ChurnMLP, scaler e imputer.
"""

import logging

from src.models.trainer import load_model as _load_model

logger = logging.getLogger(__name__)

_model = None
_scaler = None
_imputer = None


def get_model():
    """
    Retorna o modelo ChurnMLP, scaler e imputer, carregando do disco na primeira chamada.

    Returns:
        Tupla (modelo ChurnMLP em eval(), scaler, imputer).
    """
    global _model, _scaler, _imputer

    if _model is None:
        _model, _scaler, _imputer = _load_model()
        logger.info("Modelo MLP carregado com sucesso.")

    return _model, _scaler, _imputer