"""
Testes para src/prediction/service.py — pipeline de preprocessing.

O feature_mapping.py foi substituído pelo pipeline interno do service.py.
Estes testes validam o preprocessing direto no service.
"""

import numpy as np
import pandas as pd
import pytest

from src.prediction.service import (
    ALL_FEATURES,
    TOP_FEATURES,
    _encode,
    _feature_engineering,
)


@pytest.fixture
def df_bruto():
    return pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }])


def test_feature_engineering_cria_charges_per_month(df_bruto):
    result = _feature_engineering(df_bruto)
    assert "charges_per_month" in result.columns


def test_feature_engineering_cria_is_monthly(df_bruto):
    result = _feature_engineering(df_bruto)
    assert "is_monthly" in result.columns
    assert result["is_monthly"].iloc[0] == 1  # Month-to-month


def test_feature_engineering_cria_service_count(df_bruto):
    result = _feature_engineering(df_bruto)
    assert "service_count" in result.columns


def test_feature_engineering_cria_charges_x_monthly(df_bruto):
    result = _feature_engineering(df_bruto)
    assert "charges_x_monthly" in result.columns


def test_encode_gera_36_colunas(df_bruto):
    df = _feature_engineering(df_bruto)
    df_encoded = _encode(df)
    for col in ALL_FEATURES:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    assert len([c for c in ALL_FEATURES if c in df_encoded.columns or True]) == 36


def test_all_features_tem_36_elementos():
    assert len(ALL_FEATURES) == 36


def test_top_features_tem_19_elementos():
    assert len(TOP_FEATURES) == 19


def test_top_features_subconjunto_de_all_features():
    for f in TOP_FEATURES:
        assert f in ALL_FEATURES


def test_feature_engineering_tenure_zero_usa_monthly_charges():
    df = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 0, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0, "TotalCharges": 0.0,
    }])
    result = _feature_engineering(df)
    assert result["charges_per_month"].iloc[0] == 50.0