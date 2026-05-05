from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.mlp_torch import PyTorchMLPWrapper
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
# MLP PyTorch
# ---------------------------------------------------------------------------

@pytest.fixture
def dados_mlp():
    """Dataset com 40 amostras para suportar validation split e BatchNorm."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.integers(0, 2, 40), "f2": rng.integers(0, 2, 40)})
    y = pd.Series([i % 2 for i in range(40)])
    return X, y


def test_train_mlp_retorna_pytorch_wrapper(dados_mlp):
    X, y = dados_mlp
    model = train_mlp(X, y)
    assert isinstance(model, PyTorchMLPWrapper)


def test_train_mlp_possui_interface_sklearn(dados_mlp):
    X, y = dados_mlp
    model = train_mlp(X, y)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "classes_")
    assert hasattr(model, "feature_names_in_")


def test_train_mlp_prediz_apos_treino(dados_mlp):
    X, y = dados_mlp
    model = train_mlp(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_train_mlp_trata_desbalanceamento(dados_mlp):
    """Garante que o treino funciona com classes desbalanceadas via pos_weight."""
    X, _ = dados_mlp
    y_unbalanced = pd.Series([1 if i < 8 else 0 for i in range(40)])
    model = train_mlp(X, y_unbalanced)
    assert isinstance(model, PyTorchMLPWrapper)


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
