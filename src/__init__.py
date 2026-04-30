"""
Tech Challenge FIAP — Fase 1
Solução de predição de churn em telecomunicações com API REST Flask.

Convenção de nomenclatura:
  - Contrato público da API: camelCase (ex: tenureMonths, hasPartner)
  - Nomes internos do modelo sklearn: mantidos conforme gerados pelo pandas
  - A conversão entre os dois domínios é feita em src.prediction.feature_mapping

Módulos:
  config        Configurações e constantes centralizadas (settings.py)
  data          Carregamento e processamento de dados (loader, processor)
  models        Treinamento e persistência do modelo (trainer)
  evaluation    Métricas técnicas e de negócio (metrics, features)
  prediction    Serviço de inferência e mapeamento de features (service, feature_mapping)
  visualization Geração de gráficos (plots)
  utils         Utilitários gerais (helpers)
  api           Aplicação Flask REST (app, routes, dependencies)
"""

__version__ = "1.0.0"
__author__ = "FIAP Tech Challenge"

from src.config.settings import BUSINESS_METRICS, DATA_PATHS, MODEL_CONFIG

__all__ = [
    "MODEL_CONFIG",
    "BUSINESS_METRICS",
    "DATA_PATHS",
]
