"""
Testes para src/api/dependencies.py
"""

from unittest.mock import patch

import pytest

import src.api.dependencies as deps


def test_get_model_carrega_do_disco(mock_model, mock_scaler, mock_imputer):
    with patch("src.api.dependencies._load_model", return_value=(mock_model, mock_scaler, mock_imputer)):
        model, scaler, imputer = deps.get_model()

    assert model is mock_model
    assert scaler is mock_scaler
    assert imputer is mock_imputer


def test_get_model_usa_cache_na_segunda_chamada(mock_model, mock_scaler, mock_imputer):
    with patch("src.api.dependencies._load_model", return_value=(mock_model, mock_scaler, mock_imputer)) as mock_load:
        deps.get_model()
        deps.get_model()

    assert mock_load.call_count == 1


def test_get_model_retorna_tupla_de_3(mock_model, mock_scaler, mock_imputer):
    with patch("src.api.dependencies._load_model", return_value=(mock_model, mock_scaler, mock_imputer)):
        result = deps.get_model()

    assert isinstance(result, tuple)
    assert len(result) == 3