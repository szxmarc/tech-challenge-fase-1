"""
Treinamento e persistência de modelos.

Funções para treinar, salvar e carregar modelos (Regressão Logística como
baseline e MLP PyTorch como modelo de produção). Os caminhos de armazenamento são
lidos de DATA_PATHS (settings.py).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import DATA_PATHS, MLP_CONFIG, MODEL_CONFIG
from src.models.mlp_torch import ChurnMLP, PyTorchMLPWrapper


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


def train_mlp(X_train, y_train) -> PyTorchMLPWrapper:
    """
    Treina o MLP PyTorch (modelo de produção) com os hiperparâmetros de MLP_CONFIG.

    Desbalanceamento de classes tratado via pos_weight no BCEWithLogitsLoss,
    equivalente ao class_weight='balanced' da Regressão Logística.
    Early stopping monitora a val_loss em 10% do conjunto de treino.

    Args:
        X_train: Features de treino (já escalonadas).
        y_train: Target de treino.

    Returns:
        PyTorchMLPWrapper treinado com interface sklearn-compatível.
    """
    config = MLP_CONFIG
    X_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    y_np = y_train.values.ravel() if hasattr(y_train, "values") else np.asarray(y_train).ravel()
    feature_names = (
        list(X_train.columns) if hasattr(X_train, "columns")
        else [str(i) for i in range(X_np.shape[1])]
    )

    # pos_weight para compensar o desbalanceamento (~26% churners vs ~74% não-churners)
    n_neg = int((y_np == 0).sum())
    n_pos = int((y_np == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    # Separação de validação para early stopping
    X_t, X_v, y_t, y_v = train_test_split(
        X_np, y_np, test_size=0.1, random_state=42, stratify=y_np
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_t, dtype=torch.float32),
            torch.tensor(y_t, dtype=torch.float32),
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    X_val = torch.tensor(X_v, dtype=torch.float32)
    y_val = torch.tensor(y_v, dtype=torch.float32)

    net = ChurnMLP(
        input_size=X_np.shape[1],
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict | None = None

    for _epoch in range(config["epochs"]):
        net.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(net(X_batch).squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = criterion(net(X_val).squeeze(-1), y_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    return PyTorchMLPWrapper(net, feature_names, threshold=config["threshold"])


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


def save_mlp_model(model: PyTorchMLPWrapper, scaler) -> None:
    """
    Persiste o modelo MLP PyTorch de produção, scaler e config em disco.

    Args:
        model: PyTorchMLPWrapper treinado.
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


def load_mlp_model() -> tuple[PyTorchMLPWrapper, object]:
    """
    Carrega o modelo MLP PyTorch de produção e scaler do disco.

    Returns:
        Tupla (PyTorchMLPWrapper, scaler).

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
