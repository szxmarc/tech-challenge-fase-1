"""
Processamento e transformação de dados.

Responsável por encoding de variáveis categóricas, split estratificado
e escalonamento. Usado exclusivamente em notebooks de treinamento —
não é chamado pela API em tempo de execução.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config.settings import DATA_PATHS, TRAIN_TEST_SPLIT

# Colunas binárias (Yes/No ou Male/Female): recebem LabelEncoder e mantêm o
# próprio nome, produzindo valores 0/1 que correspondem ao contrato da API.
_BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Colunas de origem de cada feature derivada (nomes pós-encoding do modelo).
DERIVED_FEATURE_SOURCES: dict[str, list[str]] = {
    "charges_per_month": ["TotalCharges", "tenure", "MonthlyCharges"],
    "is_monthly":        ["Contract_One year", "Contract_Two year"],
    "service_count":     ["OnlineSecurity_Yes", "OnlineBackup_Yes", "DeviceProtection_Yes",
                          "TechSupport_Yes", "StreamingTV_Yes", "StreamingMovies_Yes"],
    "has_no_services":   ["OnlineSecurity_Yes", "OnlineBackup_Yes", "DeviceProtection_Yes",
                          "TechSupport_Yes", "StreamingTV_Yes", "StreamingMovies_Yes"],
    "is_senior_alone":   ["SeniorCitizen", "Partner"],
    "charges_x_monthly": ["MonthlyCharges", "Contract_One year", "Contract_Two year"],
}

_SERVICE_COLS = [
    "OnlineSecurity_Yes", "OnlineBackup_Yes", "DeviceProtection_Yes",
    "TechSupport_Yes", "StreamingTV_Yes", "StreamingMovies_Yes",
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza encoding de variáveis categóricas do dataset Telco.

    Estratégia:
    - Colunas binárias (Yes/No, Male/Female): LabelEncoder → 0/1, mantém nome original.
    - Colunas multi-classe: pd.get_dummies com drop_first=True, removendo a
      categoria base de cada variável (elimina multicolinearidade).
    - TotalCharges: convertida para float antes do encoding (no CSV bruto
      algumas linhas têm espaço em branco, tornando a coluna object).

    Args:
        df: DataFrame com os dados brutos.

    Returns:
        DataFrame com todas as colunas numéricas, pronto para split e scaling.
    """
    df_encoded = df.copy()

    if "customerID" in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=["customerID"])

    if "TotalCharges" in df_encoded.columns:
        df_encoded["TotalCharges"] = (
            pd.to_numeric(df_encoded["TotalCharges"], errors="coerce").fillna(0.0)
        )

    if "Churn" in df_encoded.columns:
        le = LabelEncoder()
        df_encoded["Churn"] = le.fit_transform(df_encoded["Churn"])

    for col in _BINARY_COLS:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

    df_encoded = pd.get_dummies(df_encoded, drop_first=True)

    return df_encoded


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas a partir do DataFrame já encodado.

    Pode ser chamado tanto no pipeline de treino (DataFrame completo, 30 colunas)
    quanto na inferência (apenas as colunas originais presentes no input).
    Colunas ausentes fazem a derivada correspondente ser silenciosamente omitida.

    Features derivadas:
      charges_per_month  TotalCharges / tenure (ou MonthlyCharges se tenure == 0)
      is_monthly         1 se contrato mensal (categoria base do get_dummies)
      service_count      soma dos serviços de internet ativos
      has_no_services    1 se nenhum serviço de internet contratado
      is_senior_alone    SeniorCitizen sem parceiro
      charges_x_monthly  MonthlyCharges × is_monthly
    """
    df = df.copy()

    if {"TotalCharges", "tenure", "MonthlyCharges"}.issubset(df.columns):
        df["charges_per_month"] = df.apply(
            lambda r: r["TotalCharges"] / r["tenure"] if r["tenure"] > 0 else r["MonthlyCharges"],
            axis=1,
        )

    if {"Contract_One year", "Contract_Two year"}.issubset(df.columns):
        df["is_monthly"] = 1 - df["Contract_One year"] - df["Contract_Two year"]

    available_service = [c for c in _SERVICE_COLS if c in df.columns]
    if available_service:
        df["service_count"] = df[available_service].sum(axis=1)
        df["has_no_services"] = (df["service_count"] == 0).astype(int)

    if {"SeniorCitizen", "Partner"}.issubset(df.columns):
        df["is_senior_alone"] = df["SeniorCitizen"] * (1 - df["Partner"])

    if {"MonthlyCharges", "is_monthly"}.issubset(df.columns):
        df["charges_x_monthly"] = df["MonthlyCharges"] * df["is_monthly"]

    return df


def select_features(X: pd.DataFrame, y: pd.Series, threshold: float = 0.90) -> list[str]:
    """
    Seleciona features usando importância acumulada do Random Forest.

    Retorna o menor conjunto de features (em ordem decrescente de importância)
    que cobre ao menos `threshold` da importância total.

    Args:
        X: DataFrame com todas as features.
        y: Series do target.
        threshold: Fração mínima de importância acumulada (padrão 90%).

    Returns:
        Lista de nomes de features selecionadas.
    """
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    cumulative = importances.cumsum()

    n = int((cumulative < threshold).sum()) + 1
    n = min(n, len(importances))

    return importances.index[:n].tolist()


def split_and_scale_data(X: pd.DataFrame, y: pd.Series):
    """
    Realiza split estratificado e padronização com StandardScaler.

    O scaler é ajustado somente nos dados de treino para evitar
    data leakage nos dados de teste.

    Args:
        X: DataFrame de features.
        y: Series do target.

    Returns:
        Tupla (X_train, X_test, y_train, y_test, scaler).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT["test_size"],
        random_state=TRAIN_TEST_SPLIT["random_state"],
        stratify=y if TRAIN_TEST_SPLIT["stratify"] else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train, X_test, y_train, y_test, scaler


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
) -> None:
    """
    Persiste os dados processados em disco.

    O destino é definido por DATA_PATHS["data_processed"] (settings.py),
    garantindo consistência com o restante do projeto.

    Args:
        X_train: Features de treino escalonadas.
        X_test:  Features de teste escalonadas.
        y_train: Target de treino.
        y_test:  Target de teste.
        scaler:  StandardScaler ajustado no treino.
    """
    from pathlib import Path

    output_path = Path(DATA_PATHS["data_processed"])
    output_path.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)
    joblib.dump(scaler, output_path / "scaler.joblib")
