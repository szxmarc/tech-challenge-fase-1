"""
🏋️ Treinamento de Modelos

Funções para treinar e persistir modelos de machine learning.
"""

import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from src.config.settings import MODEL_CONFIG, DATA_PATHS
import json


def train_logistic_regression(X_train, y_train):
    """
    Treina um modelo LogisticRegression com os parâmetros configurados.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
    
    Returns:
        Modelo treinado
    """
    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train, y_train.values.ravel())
    print(f"✅ Modelo LogisticRegression treinado")
    return model


def save_model(model, scaler, model_name: str = "logistic_regression"):
    """
    Persiste modelo e scaler em disco.
    
    Args:
        model: Modelo treinado
        scaler: Objeto StandardScaler
        model_name: Nome da pasta do modelo
    """
    output_path = Path(DATA_PATHS["project_root"]) / "models" / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, output_path / "model.joblib")
    joblib.dump(scaler, output_path / "scaler.joblib")
    
    with open(output_path / "config.json", "w") as f:
        json.dump(MODEL_CONFIG, f, indent=2)
    
    print(f"✅ Modelo salvo em: {output_path}")


def load_model(model_name: str = "logistic_regression"):
    """
    Carrega modelo treinado do disco.
    
    Args:
        model_name: Nome da pasta do modelo
    
    Returns:
        Tupla (modelo, scaler)
    """
    model_path = Path(DATA_PATHS["project_root"]) / "models" / model_name
    model = joblib.load(model_path / "model.joblib")
    scaler = joblib.load(model_path / "scaler.joblib")
    
    print(f"✅ Modelo carregado: {model_name}")
    return model, scaler
