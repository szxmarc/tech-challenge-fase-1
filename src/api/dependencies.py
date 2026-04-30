"""
Dependências compartilhadas da API.

Centraliza o carregamento e cache de recursos pesados reutilizados entre
requisições: modelo treinado, scaler e lista de features (camelCase).

Os nomes de features retornados por get_feature_names() seguem o contrato
público da API (camelCase). A conversão para os nomes internos do modelo
é responsabilidade de src.prediction.feature_mapping.
"""

import logging
from pathlib import Path

import joblib

from src.config.settings import DATA_PATHS

logger = logging.getLogger(__name__)

_model = None
_scaler = None
_feature_names: list[str] | None = None


def get_model():
    """
    Retorna o modelo e o scaler, carregando do disco na primeira chamada.

    Returns:
        Tupla (modelo LogisticRegression, scaler StandardScaler).
    """
    global _model, _scaler

    if _model is None:
        model_path = Path(DATA_PATHS["models"])
        _model = joblib.load(model_path / "model.joblib")
        _scaler = joblib.load(model_path / "scaler.joblib")
        logger.info("Modelo carregado: %s", model_path)

    return _model, _scaler


def get_feature_names() -> list[str]:
    """
    Retorna a lista de features em camelCase esperadas pela API.

    Os nomes são lidos de feature_names.txt e representam o contrato público
    da API — não os nomes internos do modelo sklearn.

    Returns:
        Lista ordenada de nomes de features em camelCase.
    """
    global _feature_names

    if _feature_names is None:
        feature_file = Path(DATA_PATHS["feature_names"])
        with open(feature_file, "r", encoding="utf-8") as f:
            _feature_names = [line.strip() for line in f if line.strip()]
        logger.info("%d features carregadas", len(_feature_names))

    return _feature_names
