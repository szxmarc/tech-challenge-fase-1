"""
Serviço de predição de churn.

Responsável por toda a lógica de negócio do fluxo de inferência:
  1. Validar o input recebido pela API (nomes camelCase)
  2. Converter para os nomes originais esperados pelo modelo
  3. Escalonar as features com o scaler treinado
  4. Executar a inferência e retornar o resultado formatado

O isolamento desta camada garante que as rotas HTTP não conheçam
detalhes do modelo, e que o modelo não conheça o contrato da API.
"""

import pandas as pd

from src.api.dependencies import get_feature_names, get_model
from src.prediction.feature_mapping import MODEL_FEATURE_NAMES, to_model_input


def _validate_input(input_data: dict) -> list[str]:
    """Retorna lista de features camelCase ausentes no input recebido."""
    expected_features = get_feature_names()
    return [feature for feature in expected_features if feature not in input_data]


def _normalize_booleans(input_data: dict) -> dict:
    """Converte valores booleanos para inteiros (True → 1, False → 0)."""
    return {
        key: int(value) if isinstance(value, bool) else value
        for key, value in input_data.items()
    }


def _build_prediction_result(y_pred: int, y_pred_proba) -> dict:
    """Monta o dicionário de resposta a partir da predição bruta do modelo."""
    return {
        "prediction": int(y_pred),
        "predictionLabel": "Churn" if y_pred == 1 else "Não Churn",
        "probabilityNoChurn": f"{y_pred_proba[0] * 100:.1f}%",
        "probabilityChurn": f"{y_pred_proba[1] * 100:.1f}%",
        "confidence": f"{max(y_pred_proba) * 100:.1f}%",
    }


def predict_single(input_data: dict) -> dict:
    """
    Realiza predição de churn para um único cliente.

    O input deve usar os nomes camelCase do contrato da API
    (definidos em feature_mapping.py). A conversão para os nomes
    originais do modelo é feita internamente.

    Args:
        input_data: Dicionário com as features do cliente em camelCase.

    Returns:
        Dicionário com predição, label e probabilidades formatadas em %.

    Raises:
        ValueError: Se alguma feature obrigatória estiver ausente.
        ValueError: Se algum valor não puder ser convertido para float.
    """
    missing_features = _validate_input(input_data)
    if missing_features:
        raise ValueError(f"Features ausentes: {missing_features}")

    # Normaliza booleanos e converte chaves camelCase → nomes originais do modelo
    normalized = _normalize_booleans(input_data)
    model_input = to_model_input(normalized)

    model, scaler = get_model()

    try:
        X = pd.DataFrame([model_input])[MODEL_FEATURE_NAMES].astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Tipo de dado inválido: {exc}") from exc

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)[0]
    y_pred_proba = model.predict_proba(X_scaled)[0]

    return _build_prediction_result(y_pred, y_pred_proba)


def predict_batch(customers: list[dict]) -> list[dict]:
    """
    Realiza predição de churn para uma lista de clientes.

    Cada cliente é processado individualmente via predict_single.
    Erros por cliente são capturados e retornados no campo "error"
    sem interromper o processamento dos demais.

    Args:
        customers: Lista de dicionários com features em camelCase por cliente.

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
