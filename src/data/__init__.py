"""
Módulo de dados — carregamento e processamento.

Responsabilidades:
  loader.py    Leitura do dataset bruto e dos dados processados
  processor.py Encoding de variáveis categóricas, split e escalonamento
"""

from .loader import load_raw_data
from .processor import encode_features, save_processed_data, split_and_scale_data

__all__ = [
    "load_raw_data",
    "encode_features",
    "split_and_scale_data",
    "save_processed_data",
]
