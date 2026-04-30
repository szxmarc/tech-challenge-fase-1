"""
Treinamento e persistência de modelos.

Funções para treinar, salvar e carregar o modelo de regressão logística.
Os caminhos de armazenamento são lidos de DATA_PATHS (settings.py).
"""

import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from src.config.settings import DATA_PATHS, MODEL_CONFIG


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Treina um LogisticRegression com os hiperparâmetros definidos em MODEL_CONFIG.

    Args:
        X_train: Features de treino (array ou DataFrame).
        y_train: Target de treino (array ou Series).

    Returns:
        Modelo treinado.
    """
    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train, y_train.values.ravel())
    return model


def save_model(model: LogisticRegression, scaler) -> None:
    """
    Persiste modelo, scaler e configuração de hiperparâmetros em disco.

    Args:
        model: Modelo treinado.
        scaler: Objeto StandardScaler ajustado.
    """
    output_path = Path(DATA_PATHS["models"])
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path / "model.joblib")
    joblib.dump(scaler, output_path / "scaler.joblib")

    with open(output_path / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(MODEL_CONFIG, config_file, indent=2)


def load_model() -> tuple[LogisticRegression, object]:
    """
    Carrega modelo e scaler do disco.

    Returns:
        Tupla (modelo, scaler).

    Raises:
        FileNotFoundError: Se os arquivos de modelo não forem encontrados.
    """
    model_path = Path(DATA_PATHS["models"])
    model_file = model_path / "model.joblib"
    scaler_file = model_path / "scaler.joblib"

    if not model_file.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_file}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler não encontrado em: {scaler_file}")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler
