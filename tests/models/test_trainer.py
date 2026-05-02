"""
Testes para src/models/trainer.py — ChurnMLP, train_mlp, save_model, load_model.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.models.trainer import ChurnMLP, load_model, save_model, train_mlp


@pytest.fixture
def dados_simples():
    X = np.random.rand(50, 19).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    return X, y


@pytest.fixture
def modelo_treinado(dados_simples):
    X, y = dados_simples
    return train_mlp(X, pd.Series(y))


@pytest.fixture
def scaler_real(dados_simples):
    X, _ = dados_simples
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture
def imputer_real(dados_simples):
    X, _ = dados_simples
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X)
    return imputer


# ── ChurnMLP ──────────────────────────────────────────────────────────────────

def test_churn_mlp_retorna_tensor():
    model = ChurnMLP(input_dim=19, hidden_layers=[64, 32], dropout=0.3)
    x = torch.rand(4, 19)
    out = model(x)
    assert isinstance(out, torch.Tensor)


def test_churn_mlp_shape_saida():
    model = ChurnMLP(input_dim=19, hidden_layers=[64, 32], dropout=0.3)
    x = torch.rand(8, 19)
    out = model(x)
    assert out.shape == (8, 1)


def test_churn_mlp_tem_batchnorm():
    model = ChurnMLP(input_dim=19, hidden_layers=[64, 32], dropout=0.3)
    has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules())
    assert has_bn


# ── train_mlp ─────────────────────────────────────────────────────────────────

def test_train_mlp_retorna_churn_mlp(dados_simples):
    X, y = dados_simples
    model = train_mlp(X, pd.Series(y))
    assert isinstance(model, ChurnMLP)


def test_train_mlp_modelo_em_eval(dados_simples):
    X, y = dados_simples
    model = train_mlp(X, pd.Series(y))
    assert not model.training


# ── save_model ────────────────────────────────────────────────────────────────

def test_save_model_cria_model_pt(tmp_path, modelo_treinado, scaler_real, imputer_real):
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        save_model(modelo_treinado, scaler_real, imputer_real)
    assert (tmp_path / "model.pt").exists()


def test_save_model_cria_scaler_joblib(tmp_path, modelo_treinado, scaler_real, imputer_real):
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        save_model(modelo_treinado, scaler_real, imputer_real)
    assert (tmp_path / "scaler.joblib").exists()


def test_save_model_cria_imputer_joblib(tmp_path, modelo_treinado, scaler_real, imputer_real):
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        save_model(modelo_treinado, scaler_real, imputer_real)
    assert (tmp_path / "imputer.joblib").exists()


def test_save_model_cria_config_json(tmp_path, modelo_treinado, scaler_real, imputer_real):
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        save_model(modelo_treinado, scaler_real, imputer_real)
    assert (tmp_path / "config.json").exists()


# ── load_model ────────────────────────────────────────────────────────────────

def test_load_model_levanta_erro_sem_model_pt(tmp_path):
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
            load_model()


def test_load_model_levanta_erro_sem_scaler(tmp_path):
    (tmp_path / "model.pt").touch()
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        with pytest.raises(FileNotFoundError, match="Scaler não encontrado"):
            load_model()


def test_load_model_levanta_erro_sem_imputer(tmp_path):
    (tmp_path / "model.pt").touch()
    (tmp_path / "scaler.joblib").touch()
    with patch("src.models.trainer.DATA_PATHS", {"models": str(tmp_path)}):
        with pytest.raises(FileNotFoundError, match="Imputer não encontrado"):
            load_model()