"""
Fixtures compartilhadas entre todos os testes.
"""

import numpy as np
import pytest

import src.api.dependencies as deps
from src.api.app import create_app

# Nomes das features em camelCase (espelho do feature_names.txt)
MOCK_FEATURE_NAMES = [
    "gender", "isSeniorCitizen", "hasPartner", "hasDependents", "tenureMonths",
    "hasPhoneService", "hasPaperlessBilling", "monthlyCharges", "totalCharges",
    "multipleLinesNoPhone", "multipleLinesActive", "internetFiberOptic", "internetNone",
    "onlineSecurityNoInternet", "onlineSecurityActive", "onlineBackupNoInternet",
    "onlineBackupActive", "deviceProtectionNoInternet", "deviceProtectionActive",
    "techSupportNoInternet", "techSupportActive", "streamingTvNoInternet",
    "streamingTvActive", "streamingMoviesNoInternet", "streamingMoviesActive",
    "contractOneYear", "contractTwoYear", "paymentCreditCardAutomatic",
    "paymentElectronicCheck", "paymentMailedCheck",
]


@pytest.fixture(autouse=True)
def reset_dependency_cache():
    """Limpa o cache de modelo e features antes de cada teste."""
    deps._model = None
    deps._scaler = None
    deps._feature_names = None
    yield
    deps._model = None
    deps._scaler = None
    deps._feature_names = None


@pytest.fixture
def mock_model():
    from unittest.mock import MagicMock
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.32, 0.68]])
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([[0.1] * 30])
    return model


@pytest.fixture
def mock_scaler():
    from unittest.mock import MagicMock
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 30))
    return scaler


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def sample_input():
    return {
        "gender": 1,
        "isSeniorCitizen": False,
        "hasPartner": True,
        "hasDependents": False,
        "tenureMonths": 12,
        "hasPhoneService": True,
        "hasPaperlessBilling": True,
        "monthlyCharges": 65.5,
        "totalCharges": 786.0,
        "multipleLinesNoPhone": False,
        "multipleLinesActive": True,
        "internetFiberOptic": True,
        "internetNone": False,
        "onlineSecurityNoInternet": False,
        "onlineSecurityActive": False,
        "onlineBackupNoInternet": False,
        "onlineBackupActive": True,
        "deviceProtectionNoInternet": False,
        "deviceProtectionActive": False,
        "techSupportNoInternet": False,
        "techSupportActive": False,
        "streamingTvNoInternet": False,
        "streamingTvActive": True,
        "streamingMoviesNoInternet": False,
        "streamingMoviesActive": True,
        "contractOneYear": False,
        "contractTwoYear": False,
        "paymentCreditCardAutomatic": False,
        "paymentElectronicCheck": True,
        "paymentMailedCheck": False,
    }
