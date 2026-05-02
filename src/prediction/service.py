"""
Serviço de predição de churn.

Replica o pipeline exato do notebook:
  1. Feature engineering
  2. One-hot encoding → 36 colunas (mesma ordem do treino)
  3. Imputar NaN com mediana (SimpleImputer)
  4. Escalar as 36 colunas com o scaler treinado
  5. Selecionar as 19 TOP_FEATURES
  6. Inferência com ChurnMLP (PyTorch)
"""

import numpy as np
import pandas as pd
import torch

from src.api.dependencies import get_model

# Ordem exata das 36 colunas após get_dummies no notebook
ALL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_month",
    "is_monthly",
    "service_count",
    "has_no_services",
    "is_senior_alone",
    "charges_x_monthly",
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

# 19 features selecionadas pelo Random Forest (ordem do treino da MLP)
TOP_FEATURES = [
    "charges_x_monthly",
    "tenure",
    "TotalCharges",
    "is_monthly",
    "charges_per_month",
    "MonthlyCharges",
    "Contract_Two year",
    "InternetService_Fiber optic",
    "PaymentMethod_Electronic check",
    "service_count",
    "OnlineSecurity_Yes",
    "TechSupport_Yes",
    "PaperlessBilling_Yes",
    "gender_Male",
    "Contract_One year",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineBackup_Yes",
    "StreamingTV_No internet service",
]

CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Replica o feature_engineering do notebook."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df["charges_per_month"] = np.where(
        df["tenure"] == 0,
        df["MonthlyCharges"],
        df["TotalCharges"] / df["tenure"],
    )

    df["is_monthly"] = (df["Contract"] == "Month-to-month").astype(int)

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    df["has_no_services"] = (df["service_count"] == 0).astype(int)

    df["is_senior_alone"] = (
        (df["SeniorCitizen"] == 1) & (df["Partner"] == "No")
    ).astype(int)

    df["charges_x_monthly"] = df["MonthlyCharges"] * df["is_monthly"]

    return df


# Categorias exatas de cada coluna — mesmas do dataset de treino
CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ],
}

def _encode(df: pd.DataFrame) -> pd.DataFrame:
    for col, cats in CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return pd.get_dummies(df, columns=list(CATEGORIES.keys()), drop_first=True)


def _preprocess(input_data: dict, imputer, scaler) -> np.ndarray:
    """
    Pipeline completo:
    feature engineering → get_dummies → imputer → scaler → seleciona TOP_FEATURES
    """
    df = pd.DataFrame([input_data])
    df = _feature_engineering(df)
    df = _encode(df)

    # Garante todas as 36 colunas na ordem correta
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[ALL_FEATURES].astype(float)

    # Imputer → Scaler
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    # Seleciona as 19 TOP_FEATURES
    top_indices = [ALL_FEATURES.index(f) for f in TOP_FEATURES]
    return X_scaled[:, top_indices]


def _build_prediction_result(y_pred: int, proba_no_churn: float, proba_churn: float) -> dict:
    """Monta o dicionário de resposta."""
    return {
        "prediction": int(y_pred),
        "predictionLabel": "Churn" if y_pred == 1 else "Nao Churn",
        "probabilityNoChurn": f"{proba_no_churn * 100:.1f}%",
        "probabilityChurn": f"{proba_churn * 100:.1f}%",
        "confidence": f"{max(proba_no_churn, proba_churn) * 100:.1f}%",
    }


def predict_single(input_data: dict) -> dict:
    """
    Realiza predição de churn para um único cliente.

    Args:
        input_data: Dicionário com os campos brutos do cliente.

    Returns:
        Dicionário com predição, label e probabilidades.
    """
    model, scaler, imputer = get_model()

    X_top = _preprocess(input_data, imputer, scaler)

    with torch.no_grad():
        logits = model(torch.tensor(X_top, dtype=torch.float32))
        proba_churn = torch.sigmoid(logits).item()

    proba_no_churn = 1 - proba_churn
    y_pred = int(proba_churn >= 0.5)

    return _build_prediction_result(y_pred, proba_no_churn, proba_churn)


def predict_batch(customers: list[dict]) -> list[dict]:
    """
    Realiza predição de churn para uma lista de clientes.

    Args:
        customers: Lista de dicionários com campos brutos por cliente.

    Returns:
        Lista de resultados com o índice original de cada cliente preservado.
    """
    results = []

    for index, customer in enumerate(customers):
        try:
            result = predict_single(customer)
            result["customerIndex"] = index
        except Exception as exc:
            result = {"customerIndex": index, "error": str(exc)}

        results.append(result)

    return results