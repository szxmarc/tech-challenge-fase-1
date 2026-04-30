from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.processor import encode_features, save_processed_data, split_and_scale_data


@pytest.fixture
def dataframe_bruto():
    return pd.DataFrame({
        "customerID": ["001", "002", "003", "004", "005", "006"],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "tenure": [12, 24, 6, 36, 48, 3],
        "MonthlyCharges": [50.0, 70.0, 45.0, 80.0, 60.0, 90.0],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "No"],
        "Churn": ["Yes", "No", "Yes", "No", "Yes", "No"],
    })


def test_encode_features_remove_customer_id(dataframe_bruto):
    result = encode_features(dataframe_bruto)
    assert "customerID" not in result.columns


def test_encode_features_codifica_churn_como_inteiro(dataframe_bruto):
    result = encode_features(dataframe_bruto)
    assert result["Churn"].dtype in [int, np.int64, np.int32]


def test_encode_features_aplica_get_dummies(dataframe_bruto):
    result = encode_features(dataframe_bruto)
    colunas_categoricas_originais = {"gender", "InternetService"}
    assert not colunas_categoricas_originais.issubset(result.columns)


def test_encode_features_sem_churn_nao_falha():
    df = pd.DataFrame({"gender": ["Male", "Female"], "tenure": [12, 24]})
    result = encode_features(df)
    assert "customerID" not in result.columns


def test_split_and_scale_data_proporcao_treino_teste():
    X = pd.DataFrame(np.random.rand(100, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(np.random.randint(0, 2, 100))
    X_train, X_test, y_train, y_test, _ = split_and_scale_data(X, y)
    assert len(X_train) == 80
    assert len(X_test) == 20


def test_split_and_scale_data_retorna_scaler():
    X = pd.DataFrame(np.random.rand(100, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(np.random.randint(0, 2, 100))
    _, _, _, _, scaler = split_and_scale_data(X, y)
    assert isinstance(scaler, StandardScaler)


def test_split_and_scale_data_retorna_dataframe():
    X = pd.DataFrame(np.random.rand(100, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(np.random.randint(0, 2, 100))
    X_train, X_test, _, _, _ = split_and_scale_data(X, y)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)


def test_save_processed_data_cria_csvs(tmp_path):
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])
    scaler = StandardScaler()

    with patch("src.data.processor.DATA_PATHS", {"data_processed": str(tmp_path)}):
        save_processed_data(X, X, y, y, scaler)

    assert (tmp_path / "X_train.csv").exists()
    assert (tmp_path / "X_test.csv").exists()
    assert (tmp_path / "y_train.csv").exists()
    assert (tmp_path / "y_test.csv").exists()


def test_save_processed_data_cria_scaler_joblib(tmp_path):
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])
    scaler = StandardScaler()

    with patch("src.data.processor.DATA_PATHS", {"data_processed": str(tmp_path)}):
        save_processed_data(X, X, y, y, scaler)

    assert (tmp_path / "scaler.joblib").exists()
