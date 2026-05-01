"""
Módulo de dados — carregamento e processamento.

Responsabilidades:
  loader.py    Leitura do dataset bruto e dos dados processados
  processor.py Encoding de variáveis categóricas, split e escalonamento
"""

from .loader import load_raw_data
from .processor import (
    DERIVED_FEATURE_SOURCES,
    encode_features,
    engineer_features,
    save_processed_data,
    select_features,
    split_and_scale_data,
)

__all__ = [
    "load_raw_data",
    "encode_features",
    "engineer_features",
    "DERIVED_FEATURE_SOURCES",
    "select_features",
    "split_and_scale_data",
    "save_processed_data",
]
