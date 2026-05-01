from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.processor import DERIVED_FEATURE_SOURCES, encode_features, engineer_features, save_processed_data, select_features, split_and_scale_data


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


@pytest.fixture
def dataframe_encodado():
    """DataFrame já encodado com colunas no padrão pós-encode_features."""
    return pd.DataFrame({
        "tenure": [12, 0, 24],
        "MonthlyCharges": [50.0, 70.0, 45.0],
        "TotalCharges": [600.0, 0.0, 1080.0],
        "SeniorCitizen": [0, 1, 0],
        "Partner": [1, 0, 0],
        "Contract_One year": [0, 0, 1],
        "Contract_Two year": [0, 0, 0],
        "OnlineSecurity_Yes": [1, 0, 0],
        "OnlineBackup_Yes": [0, 0, 1],
        "DeviceProtection_Yes": [0, 0, 0],
        "TechSupport_Yes": [0, 0, 0],
        "StreamingTV_Yes": [0, 0, 0],
        "StreamingMovies_Yes": [0, 0, 0],
    })


def test_engineer_features_adiciona_charges_per_month(dataframe_encodado):
    result = engineer_features(dataframe_encodado)
    assert "charges_per_month" in result.columns
    assert result.loc[0, "charges_per_month"] == pytest.approx(600.0 / 12)
    assert result.loc[1, "charges_per_month"] == pytest.approx(70.0)


def test_engineer_features_adiciona_is_monthly(dataframe_encodado):
    result = engineer_features(dataframe_encodado)
    assert "is_monthly" in result.columns
    assert result.loc[0, "is_monthly"] == 1
    assert result.loc[2, "is_monthly"] == 0


def test_engineer_features_adiciona_service_count(dataframe_encodado):
    result = engineer_features(dataframe_encodado)
    assert "service_count" in result.columns
    assert result.loc[0, "service_count"] == 1
    assert result.loc[1, "service_count"] == 0


def test_engineer_features_adiciona_has_no_services(dataframe_encodado):
    result = engineer_features(dataframe_encodado)
    assert "has_no_services" in result.columns
    assert result.loc[1, "has_no_services"] == 1
    assert result.loc[0, "has_no_services"] == 0


def test_engineer_features_adiciona_is_senior_alone(dataframe_encodado):
    result = engineer_features(dataframe_encodado)
    assert "is_senior_alone" in result.columns
    assert result.loc[0, "is_senior_alone"] == 0
    assert result.loc[1, "is_senior_alone"] == 1


def test_engineer_features_adiciona_charges_x_monthly(dataframe_encodado):
    result = engineer_features(dataframe_encodado)
    assert "charges_x_monthly" in result.columns
    assert result.loc[0, "charges_x_monthly"] == pytest.approx(50.0)
    assert result.loc[2, "charges_x_monthly"] == pytest.approx(0.0)


def test_engineer_features_sem_colunas_nao_falha():
    df = pd.DataFrame({"tenure": [5], "MonthlyCharges": [30.0]})
    result = engineer_features(df)
    assert "charges_per_month" not in result.columns
    assert "is_monthly" not in result.columns


def test_derived_feature_sources_contem_todas_derivadas():
    expected = {"charges_per_month", "is_monthly", "service_count",
                "has_no_services", "is_senior_alone", "charges_x_monthly"}
    assert expected == set(DERIVED_FEATURE_SOURCES.keys())


def test_select_features_retorna_lista(dataframe_bruto):
    result = encode_features(dataframe_bruto)
    X = result.drop(columns=["Churn"])
    y = result["Churn"]
    selected = select_features(X, y, threshold=0.90)
    assert isinstance(selected, list)
    assert len(selected) >= 1
    assert len(selected) <= len(X.columns)


def test_select_features_nomes_validos(dataframe_bruto):
    result = encode_features(dataframe_bruto)
    X = result.drop(columns=["Churn"])
    y = result["Churn"]
    selected = select_features(X, y, threshold=0.90)
    assert all(col in X.columns for col in selected)


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
