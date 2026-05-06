from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.evaluation.features import get_feature_importance, rank_features


@pytest.fixture
def feature_names():
    return ["f1", "f2", "f3", "f4", "f5"]


@pytest.fixture
def modelo_linear(feature_names):
    model = MagicMock()
    model.coef_ = np.array([[0.5, -0.3, 0.8, 0.1, -0.6]])
    return model


def test_get_feature_importance_retorna_dataframe(modelo_linear, feature_names):
    result = get_feature_importance(modelo_linear, feature_names)
    assert isinstance(result, pd.DataFrame)


def test_get_feature_importance_contem_colunas_esperadas(modelo_linear, feature_names):
    result = get_feature_importance(modelo_linear, feature_names)
    assert "feature" in result.columns
    assert "importance" in result.columns
    assert "coefficient" in result.columns


def test_get_feature_importance_ordenado_decrescente(modelo_linear, feature_names):
    result = get_feature_importance(modelo_linear, feature_names)
    assert result["importance"].is_monotonic_decreasing


def test_get_feature_importance_usa_valor_absoluto(modelo_linear, feature_names):
    result = get_feature_importance(modelo_linear, feature_names)
    assert (result["importance"] >= 0).all()


def test_get_feature_importance_levanta_erro_sem_coef(feature_names):
    model = MagicMock(spec=[])  # sem atributo coef_
    with pytest.raises(ValueError, match="coef_"):
        get_feature_importance(model, feature_names)


def test_rank_features_retorna_top_n(modelo_linear, feature_names):
    df = get_feature_importance(modelo_linear, feature_names)
    top = rank_features(df, top_n=3)
    assert len(top) == 3


def test_rank_features_retorna_todas_se_n_maior(modelo_linear, feature_names):
    df = get_feature_importance(modelo_linear, feature_names)
    top = rank_features(df, top_n=100)
    assert len(top) == len(feature_names)


def test_rank_features_mantem_ordenacao(modelo_linear, feature_names):
    df = get_feature_importance(modelo_linear, feature_names)
    top = rank_features(df, top_n=3)
    assert top.iloc[0]["importance"] >= top.iloc[1]["importance"]
