"""
Treinamento e persistência de modelos.

Funções para treinar, salvar e carregar modelos (Regressão Logística como
baseline e MLP como modelo de produção). Os caminhos de armazenamento são
lidos de DATA_PATHS (settings.py).
"""

import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config.settings import DATA_PATHS, MODEL_CONFIG, MLP_CONFIG


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Treina um LogisticRegression (baseline) com os hiperparâmetros de MODEL_CONFIG.

    Args:
        X_train: Features de treino.
        y_train: Target de treino.

    Returns:
        Modelo baseline treinado.
    """
    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_mlp(X_train, y_train) -> MLPClassifier:
    """
    Treina o MLP (modelo de produção) com os hiperparâmetros de MLP_CONFIG.

    Usa sample_weight balanceado para compensar o desbalanceamento de classes
    (~26% churners vs ~74% não-churners), equivalente ao class_weight='balanced'
    da Regressão Logística.

    Args:
        X_train: Features de treino (já escalonadas).
        y_train: Target de treino.

    Returns:
        Modelo MLP treinado.
    """
    y = y_train.values.ravel()
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    model = MLPClassifier(**MLP_CONFIG)
    model.fit(X_train, y, sample_weight=sample_weights)
    return model


def save_model(model: LogisticRegression, scaler) -> None:
    """
    Persiste o modelo baseline (LR), scaler e config em disco.

    Args:
        model: Modelo LogisticRegression treinado.
        scaler: StandardScaler ajustado.
    """
    output_path = Path(DATA_PATHS["models"])
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path / "model.joblib")
    joblib.dump(scaler, output_path / "scaler.joblib")

    with open(output_path / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(MODEL_CONFIG, config_file, indent=2)


def save_mlp_model(model: MLPClassifier, scaler) -> None:
    """
    Persiste o modelo MLP de produção, scaler e config em disco.

    Args:
        model: Modelo MLPClassifier treinado.
        scaler: StandardScaler ajustado (mesmo usado no baseline).
    """
    output_path = Path(DATA_PATHS["mlp_models"])
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path / "model.joblib")
    joblib.dump(scaler, output_path / "scaler.joblib")

    mlp_config_serializable = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in MLP_CONFIG.items()
    }
    with open(output_path / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(mlp_config_serializable, config_file, indent=2)


def load_model() -> tuple[LogisticRegression, object]:
    """
    Carrega o modelo baseline (LR) e scaler do disco.

    Returns:
        Tupla (modelo LogisticRegression, scaler).

    Raises:
        FileNotFoundError: Se os arquivos não forem encontrados.
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


def load_mlp_model() -> tuple[MLPClassifier, object]:
    """
    Carrega o modelo MLP de produção e scaler do disco.

    Returns:
        Tupla (modelo MLPClassifier, scaler).

    Raises:
        FileNotFoundError: Se os arquivos não forem encontrados.
    """
    model_path = Path(DATA_PATHS["mlp_models"])
    model_file = model_path / "model.joblib"
    scaler_file = model_path / "scaler.joblib"

    if not model_file.exists():
        raise FileNotFoundError(f"Modelo MLP não encontrado em: {model_file}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler não encontrado em: {scaler_file}")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler
