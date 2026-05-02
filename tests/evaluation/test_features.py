"""
Testes para src/evaluation/features.py — feature importance para MLP.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.features import get_feature_importance_mlp, rank_features
from src.models.trainer import ChurnMLP


@pytest.fixture
def feature_names():
    return [f"f{i}" for i in range(19)]


@pytest.fixture
def modelo_mlp(feature_names):
    model = ChurnMLP(input_dim=len(feature_names), hidden_layers=[32, 16], dropout=0.0)
    model.eval()
    return model


@pytest.fixture
def X_test(feature_names):
    return np.random.rand(50, len(feature_names)).astype(np.float32)


@pytest.fixture
def y_test():
    return np.random.randint(0, 2, 50)


def test_get_feature_importance_retorna_dataframe(modelo_mlp, X_test, y_test, feature_names):
    result = get_feature_importance_mlp(modelo_mlp, X_test, y_test, feature_names)
    assert isinstance(result, pd.DataFrame)


def test_get_feature_importance_contem_colunas_esperadas(modelo_mlp, X_test, y_test, feature_names):
    result = get_feature_importance_mlp(modelo_mlp, X_test, y_test, feature_names)
    assert "feature" in result.columns
    assert "importance" in result.columns


def test_get_feature_importance_tem_todas_features(modelo_mlp, X_test, y_test, feature_names):
    result = get_feature_importance_mlp(modelo_mlp, X_test, y_test, feature_names)
    assert len(result) == len(feature_names)


def test_get_feature_importance_ordenado_decrescente(modelo_mlp, X_test, y_test, feature_names):
    result = get_feature_importance_mlp(modelo_mlp, X_test, y_test, feature_names)
    assert result["importance"].is_monotonic_decreasing


def test_rank_features_retorna_top_n(modelo_mlp, X_test, y_test, feature_names):
    df = get_feature_importance_mlp(modelo_mlp, X_test, y_test, feature_names)
    top = rank_features(df, top_n=5)
    assert len(top) == 5


def test_rank_features_retorna_todas_se_n_maior(modelo_mlp, X_test, y_test, feature_names):
    df = get_feature_importance_mlp(modelo_mlp, X_test, y_test, feature_names)
    top = rank_features(df, top_n=100)
    assert len(top) == len(feature_names)