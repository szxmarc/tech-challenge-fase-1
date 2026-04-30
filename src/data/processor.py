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


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza encoding de variáveis categóricas do dataset Telco.

    Aplica LabelEncoder na coluna alvo (Churn) e get_dummies nas
    demais colunas categóricas. Remove a coluna customerID pois
    não é uma feature preditiva.

    Args:
        df: DataFrame com os dados brutos.

    Returns:
        DataFrame com todas as colunas numéricas, pronto para split e scaling.
    """
    df_encoded = df.copy()

    if "customerID" in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=["customerID"])

    if "Churn" in df_encoded.columns:
        le = LabelEncoder()
        df_encoded["Churn"] = le.fit_transform(df_encoded["Churn"])

    df_encoded = pd.get_dummies(df_encoded, drop_first=False)

    return df_encoded


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
