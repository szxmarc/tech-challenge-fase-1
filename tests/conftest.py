"""
Fixtures compartilhadas entre todos os testes.
"""

import numpy as np
import pytest
import torch

import src.api.dependencies as deps
from src.api.app import create_app


@pytest.fixture(autouse=True)
def reset_dependency_cache():
    """Limpa o cache de modelo e scaler antes de cada teste."""
    deps._model = None
    deps._scaler = None
    deps._imputer = None
    yield
    deps._model = None
    deps._scaler = None
    deps._imputer = None


@pytest.fixture
def mock_model():
    from unittest.mock import MagicMock
    model = MagicMock()
    # Simula ChurnMLP — retorna tensor com logit positivo (churn)
    model.return_value = torch.tensor([[0.8]])
    model.__class__.__name__ = "ChurnMLP"
    return model


@pytest.fixture
def mock_scaler():
    from unittest.mock import MagicMock
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 36))
    return scaler


@pytest.fixture
def mock_imputer():
    from unittest.mock import MagicMock
    imputer = MagicMock()
    imputer.transform.return_value = np.zeros((1, 36))
    return imputer


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def sample_input():
    """Input com dados brutos do cliente — novo contrato da API."""
    return {
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
    }