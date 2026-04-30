"""
Módulo de predição — serviço de inferência e mapeamento de features.

Responsabilidades:
  service.py         Orquestra o fluxo completo de predição (validação,
                     normalização, conversão e inferência)
  feature_mapping.py Define o mapeamento entre os nomes camelCase da API
                     e os nomes originais usados pelo modelo sklearn

Fluxo de dados:
  API (camelCase) → feature_mapping → modelo sklearn (nomes originais)
"""

from .feature_mapping import API_FEATURE_NAMES, FEATURE_NAME_MAPPING, MODEL_FEATURE_NAMES
from .service import predict_batch, predict_single

__all__ = [
    "predict_single",
    "predict_batch",
    "FEATURE_NAME_MAPPING",
    "API_FEATURE_NAMES",
    "MODEL_FEATURE_NAMES",
]
