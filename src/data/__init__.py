"""
Módulo de Dados (Data)
======================

Propósito:
----------
Agrupar todas as funções relacionadas a CARREGAMENTO e MANIPULAÇÃO de dados.
Centraliza lógica de pré-processamento para reutilização entre notebooks.

Submódulos:
- loader.py: Carregamento e leitura de dados
- preprocessor.py: Transformações e limpeza

Uso:
----
from src.data import load_raw_data, handle_missing_values, encode_features
"""

from .loader import load_raw_data
from .preprocessor import (
    handle_missing_values,
    encode_features,
    split_and_scale_data,
    save_processed_data,
)

__all__ = [
    'load_raw_data',
    'handle_missing_values',
    'encode_features',
    'split_and_scale_data',
    'save_processed_data',
]
