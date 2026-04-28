"""
⚙️ Transformação e Processamento de Dados

Funções para preparar dados: encoding, scaling, split estratificado, etc.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.config.settings import DATA_CONFIG, BUSINESS_METRICS
import joblib


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza encoding de variáveis categóricas.
    
    Args:
        df: DataFrame com dados brutos
    
    Returns:
        DataFrame com features codificadas
    """
    df_encoded = df.copy()
    
    # Exemplo: codificar target
    if "Churn" in df_encoded.columns:
        le = LabelEncoder()
        df_encoded["Churn"] = le.fit_transform(df_encoded["Churn"])
    
    print(f"✅ Features codificadas")
    return df_encoded


def split_and_scale_data(X, y):
    """
    Realiza split estratificado e padronização com StandardScaler.
    
    Args:
        X: Features
        y: Target
    
    Returns:
        Tupla (X_train, X_test, y_train, y_test, scaler)
    """
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=y if DATA_CONFIG["stratified"] else None
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"✅ Dados split e escalados: {X_train.shape[0]} (treino), {X_test.shape[0]} (teste)")
    return X_train, X_test, y_train, y_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test, scaler, model_name: str = "logistic_regression"):
    """
    Persiste dados processados em disco.
    
    Args:
        X_train, X_test, y_train, y_test: Dados split
        scaler: Objeto StandardScaler
        model_name: Nome da pasta do modelo
    """
    from pathlib import Path
    from src.config.settings import DATA_PATHS
    
    output_path = Path(DATA_PATHS["project_root"]) / "data" / "processed" / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)
    joblib.dump(scaler, output_path / "scaler.joblib")
    
    print(f"✅ Dados processados salvos em: {output_path}")
