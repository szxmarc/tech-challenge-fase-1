"""
Mapeamento de features entre o contrato público da API e o modelo treinado.

A API expõe nomes em camelCase padronizados (contrato público). Internamente,
o modelo sklearn foi treinado com os nomes originais gerados pelo pandas
(get_dummies + label encoding). Este módulo faz a ponte entre os dois mundos.

Padrões de nomenclatura camelCase adotados:
  has<Atributo>     → atributo booleano do cliente (hasPartner, hasDependents)
  is<Caracteristica> → característica descritiva (isSeniorCitizen)
  <tipo>Active      → variante "Yes" do one-hot encoding
  <tipo>NoInternet  → variante "sem internet disponível" do one-hot encoding
  <tipo>NoPhone     → variante "sem telefone disponível" do one-hot encoding
  <tipo>None        → variante "serviço não contratado" do one-hot encoding
  <grandeza>Unit    → variável numérica com unidade explícita (tenureMonths)
"""

# Mapeamento camelCase (API) → nome original do modelo (sklearn/pandas).
# A ORDEM das chaves é relevante: define a sequência de colunas entregue
# ao scaler e ao modelo, que foi a mesma usada no treinamento.
FEATURE_NAME_MAPPING: dict[str, str] = {
    "gender":                       "gender",
    "isSeniorCitizen":              "SeniorCitizen",
    "hasPartner":                   "Partner",
    "hasDependents":                "Dependents",
    "tenureMonths":                 "tenure",
    "hasPhoneService":              "PhoneService",
    "hasPaperlessBilling":          "PaperlessBilling",
    "monthlyCharges":               "MonthlyCharges",
    "totalCharges":                 "TotalCharges",
    "multipleLinesNoPhone":         "MultipleLines_No phone service",
    "multipleLinesActive":          "MultipleLines_Yes",
    "internetFiberOptic":           "InternetService_Fiber optic",
    "internetNone":                 "InternetService_No",
    "onlineSecurityNoInternet":     "OnlineSecurity_No internet service",
    "onlineSecurityActive":         "OnlineSecurity_Yes",
    "onlineBackupNoInternet":       "OnlineBackup_No internet service",
    "onlineBackupActive":           "OnlineBackup_Yes",
    "deviceProtectionNoInternet":   "DeviceProtection_No internet service",
    "deviceProtectionActive":       "DeviceProtection_Yes",
    "techSupportNoInternet":        "TechSupport_No internet service",
    "techSupportActive":            "TechSupport_Yes",
    "streamingTvNoInternet":        "StreamingTV_No internet service",
    "streamingTvActive":            "StreamingTV_Yes",
    "streamingMoviesNoInternet":    "StreamingMovies_No internet service",
    "streamingMoviesActive":        "StreamingMovies_Yes",
    "contractOneYear":              "Contract_One year",
    "contractTwoYear":              "Contract_Two year",
    "paymentCreditCardAutomatic":   "PaymentMethod_Credit card (automatic)",
    "paymentElectronicCheck":       "PaymentMethod_Electronic check",
    "paymentMailedCheck":           "PaymentMethod_Mailed check",
}

# Lista de nomes camelCase na ordem correta — usada para validação de input.
API_FEATURE_NAMES: list[str] = list(FEATURE_NAME_MAPPING.keys())

# Lista de nomes originais na ordem correta — usada para criar o DataFrame
# que será entregue ao scaler e ao modelo.
MODEL_FEATURE_NAMES: list[str] = list(FEATURE_NAME_MAPPING.values())


def to_model_input(api_input: dict) -> dict:
    """
    Converte um dicionário com chaves camelCase para os nomes originais do modelo.

    Args:
        api_input: Dicionário recebido da API com chaves camelCase.

    Returns:
        Dicionário com as mesmas chaves renomeadas para o padrão do modelo.
    """
    return {
        FEATURE_NAME_MAPPING[api_key]: value
        for api_key, value in api_input.items()
        if api_key in FEATURE_NAME_MAPPING
    }
