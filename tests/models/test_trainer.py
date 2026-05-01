from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.models.trainer import (
    load_mlp_model,
    load_model,
    save_mlp_model,
    save_model,
    train_logistic_regression,
    train_mlp,
)


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


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

@pytest.fixture
def dados_mlp():
    """Dataset maior para suportar early_stopping (validação estratificada)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.integers(0, 2, 40), "f2": rng.integers(0, 2, 40)})
    y = pd.Series([i % 2 for i in range(40)])
    return X, y


def test_train_mlp_retorna_mlp_classifier(dados_mlp):
    X, y = dados_mlp
    model = train_mlp(X, y)
    assert isinstance(model, MLPClassifier)


def test_train_mlp_modelo_esta_ajustado(dados_mlp):
    X, y = dados_mlp
    model = train_mlp(X, y)
    assert hasattr(model, "coefs_")


def test_train_mlp_aplica_sample_weight(dados_mlp):
    """Garante que o treino usa pesos balanceados por classe."""
    from unittest.mock import patch, MagicMock
    X, y = dados_mlp
    mock_model = MagicMock(spec=MLPClassifier)
    with patch("src.models.trainer.MLPClassifier", return_value=mock_model):
        train_mlp(X, y)
    call_kwargs = mock_model.fit.call_args
    assert call_kwargs is not None
    assert "sample_weight" in call_kwargs.kwargs or len(call_kwargs.args) >= 3


@pytest.fixture
def mlp_e_scaler_reais(dados_mlp):
    from sklearn.preprocessing import StandardScaler
    X, y = dados_mlp
    model = train_mlp(X, y)
    scaler = StandardScaler().fit(X)
    return model, scaler


def test_save_mlp_model_cria_arquivo_model(tmp_path, mlp_e_scaler_reais):
    model, scaler = mlp_e_scaler_reais
    with patch("src.models.trainer.DATA_PATHS", {"mlp_models": str(tmp_path)}):
        save_mlp_model(model, scaler)
    assert (tmp_path / "model.joblib").exists()


def test_save_mlp_model_cria_arquivo_scaler(tmp_path, mlp_e_scaler_reais):
    model, scaler = mlp_e_scaler_reais
    with patch("src.models.trainer.DATA_PATHS", {"mlp_models": str(tmp_path)}):
        save_mlp_model(model, scaler)
    assert (tmp_path / "scaler.joblib").exists()


def test_save_mlp_model_cria_arquivo_config(tmp_path, mlp_e_scaler_reais):
    model, scaler = mlp_e_scaler_reais
    with patch("src.models.trainer.DATA_PATHS", {"mlp_models": str(tmp_path)}):
        save_mlp_model(model, scaler)
    assert (tmp_path / "config.json").exists()


def test_load_mlp_model_retorna_tupla(tmp_path, mock_model, mock_scaler):
    (tmp_path / "model.joblib").touch()
    (tmp_path / "scaler.joblib").touch()
    with patch("src.models.trainer.DATA_PATHS", {"mlp_models": str(tmp_path)}), \
         patch("src.models.trainer.joblib.load", side_effect=[mock_model, mock_scaler]):
        model, scaler = load_mlp_model()
    assert model is mock_model
    assert scaler is mock_scaler


def test_load_mlp_model_levanta_erro_sem_model_joblib(tmp_path):
    with patch("src.models.trainer.DATA_PATHS", {"mlp_models": str(tmp_path)}):
        with pytest.raises(FileNotFoundError, match="Modelo MLP não encontrado"):
            load_mlp_model()


def test_load_mlp_model_levanta_erro_sem_scaler_joblib(tmp_path):
    (tmp_path / "model.joblib").touch()
    with patch("src.models.trainer.DATA_PATHS", {"mlp_models": str(tmp_path)}):
        with pytest.raises(FileNotFoundError, match="Scaler não encontrado"):
            load_mlp_model()
