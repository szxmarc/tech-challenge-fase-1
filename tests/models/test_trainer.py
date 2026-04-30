from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.trainer import load_model, save_model, train_logistic_regression


@pytest.fixture
def dados_simples():
    X = pd.DataFrame({"f1": [1, 0, 1, 0, 1, 0], "f2": [0, 1, 1, 0, 0, 1]})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    return X, y


def test_train_retorna_logistic_regression(dados_simples):
    X, y = dados_simples
    model = train_logistic_regression(X, y)
    assert isinstance(model, LogisticRegression)


def test_train_modelo_esta_ajustado(dados_simples):
    X, y = dados_simples
    model = train_logistic_regression(X, y)
    assert hasattr(model, "coef_")


@pytest.fixture
def modelo_e_scaler_reais(dados_simples):
    from sklearn.preprocessing import StandardScaler
    X, y = dados_simples
    model = train_logistic_regression(X, y)
    scaler = StandardScaler().fit(X)
    return model, scaler


def test_save_model_cria_arquivo_model(tmp_path, modelo_e_scaler_reais):
    model, scaler = modelo_e_scaler_reais
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        save_model(model, scaler)
    assert (tmp_path / "model.joblib").exists()


def test_save_model_cria_arquivo_scaler(tmp_path, modelo_e_scaler_reais):
    model, scaler = modelo_e_scaler_reais
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        save_model(model, scaler)
    assert (tmp_path / "scaler.joblib").exists()


def test_save_model_cria_arquivo_config(tmp_path, modelo_e_scaler_reais):
    model, scaler = modelo_e_scaler_reais
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        save_model(model, scaler)
    assert (tmp_path / "config.json").exists()


def test_load_model_retorna_tupla(tmp_path, mock_model, mock_scaler):
    (tmp_path / "model.joblib").touch()
    (tmp_path / "scaler.joblib").touch()
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}), \
         patch("src.models.trainer.joblib.load", side_effect=[mock_model, mock_scaler]):
        model, scaler = load_model()
    assert model is mock_model
    assert scaler is mock_scaler


def test_load_model_levanta_erro_sem_model_joblib(tmp_path):
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
            load_model()


def test_load_model_levanta_erro_sem_scaler_joblib(tmp_path):
    (tmp_path / "model.joblib").touch()
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        with pytest.raises(FileNotFoundError, match="Scaler não encontrado"):
            load_model()
