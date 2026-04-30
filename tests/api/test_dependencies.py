from unittest.mock import patch

import numpy as np
import pytest

import src.api.dependencies as deps


def test_get_model_carrega_do_disco(tmp_path, mock_model, mock_scaler):
    with patch("src.api.dependencies.joblib.load", side_effect=[mock_model, mock_scaler]), \
         patch("src.api.dependencies.DATA_PATHS", {"models": str(tmp_path)}):
        model, scaler = deps.get_model()

    assert model is mock_model
    assert scaler is mock_scaler


def test_get_model_usa_cache_na_segunda_chamada(tmp_path, mock_model, mock_scaler):
    with patch("src.api.dependencies.joblib.load", side_effect=[mock_model, mock_scaler]) as mock_load, \
         patch("src.api.dependencies.DATA_PATHS", {"models": str(tmp_path)}):
        deps.get_model()
        deps.get_model()

    assert mock_load.call_count == 2  # carregou model + scaler apenas uma vez


def test_get_model_retorna_tupla(tmp_path, mock_model, mock_scaler):
    with patch("src.api.dependencies.joblib.load", side_effect=[mock_model, mock_scaler]), \
         patch("src.api.dependencies.DATA_PATHS", {"models": str(tmp_path)}):
        result = deps.get_model()

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_get_feature_names_carrega_do_arquivo(tmp_path):
    feature_file = tmp_path / "feature_names.txt"
    feature_file.write_text("gender\nisSeniorCitizen\nhasPartner\n")

    with patch("src.api.dependencies.DATA_PATHS", {"feature_names": str(feature_file)}):
        names = deps.get_feature_names()

    assert names == ["gender", "isSeniorCitizen", "hasPartner"]


def test_get_feature_names_ignora_linhas_vazias(tmp_path):
    feature_file = tmp_path / "feature_names.txt"
    feature_file.write_text("gender\n\nisSeniorCitizen\n\n")

    with patch("src.api.dependencies.DATA_PATHS", {"feature_names": str(feature_file)}):
        names = deps.get_feature_names()

    assert len(names) == 2
    assert "" not in names


def test_get_feature_names_usa_cache_na_segunda_chamada(tmp_path):
    feature_file = tmp_path / "feature_names.txt"
    feature_file.write_text("gender\n")

    with patch("src.api.dependencies.DATA_PATHS", {"feature_names": str(feature_file)}):
        first = deps.get_feature_names()
        second = deps.get_feature_names()

    assert first is second
