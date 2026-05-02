"""
Treinamento e persistência do modelo MLP PyTorch.

Funções para treinar, salvar e carregar o modelo ChurnMLP.
Os caminhos de armazenamento são lidos de DATA_PATHS (settings.py).
"""

import json
import logging
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import DATA_PATHS, MODEL_CONFIG

logger = logging.getLogger(__name__)


class ChurnMLP(nn.Module):
    """
    Multi-Layer Perceptron para classificação binária de churn.

    Arquitetura: Linear → BatchNorm1d → ReLU → Dropout (repetido por camada oculta)
    seguido de uma Linear final com 1 saída (logit).
    A Sigmoid é aplicada na inferência, não aqui.

    Args:
        input_dim:     Número de features de entrada.
        hidden_layers: Lista com o número de neurônios de cada camada oculta.
        dropout:       Taxa de dropout aplicada após cada ativação ReLU.
    """

    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float):
        super().__init__()

        layers: list[nn.Module] = []
        in_features = input_dim

        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_mlp(X_train, y_train) -> ChurnMLP:
    """
    Treina um ChurnMLP com os hiperparâmetros definidos em MODEL_CONFIG.

    Implementa early stopping baseado em val_loss, ReduceLROnPlateau
    e weight_decay no otimizador Adam para regularização.

    Args:
        X_train: Features de treino (array ou DataFrame).
        y_train: Target de treino (array ou Series).

    Returns:
        Modelo ChurnMLP treinado com melhores pesos, em modo eval().
    """
    torch.manual_seed(MODEL_CONFIG["random_state"])

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_array = y_train.values.ravel() if hasattr(y_train, "values") else y_train
    y_tensor = torch.tensor(y_array, dtype=torch.float32).unsqueeze(1)

    n_pos = float(y_tensor.sum())
    n_neg = float(len(y_tensor) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    # Split treino/validação interno (80/20) para early stopping
    n_total = len(X_tensor)
    n_train = int(n_total * 0.8)
    X_tr, X_val = X_tensor[:n_train], X_tensor[n_train:]
    y_tr, y_val = y_tensor[:n_train], y_tensor[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=False,
    )

    model = ChurnMLP(
        input_dim=MODEL_CONFIG["input_dim"],
        hidden_layers=MODEL_CONFIG["hidden_layers"],
        dropout=MODEL_CONFIG["dropout"],
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MODEL_CONFIG["learning_rate"],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    patience = MODEL_CONFIG["patience"]

    for epoch in range(1, MODEL_CONFIG["epochs"] + 1):
        # Treino
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # Validação
        model.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss_acc += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss = val_loss_acc / len(X_val)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping na época %d (patience=%d)", epoch, patience)
                break

    # Carrega os melhores pesos
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model.eval()


def save_model(model: ChurnMLP, scaler, imputer) -> None:
    """
    Persiste modelo (state dict), scaler, imputer e configuração em disco.

    Args:
        model:   Modelo ChurnMLP treinado.
        scaler:  Objeto StandardScaler ajustado.
        imputer: Objeto SimpleImputer ajustado.
    """
    output_path = Path(DATA_PATHS["models"])
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pt")
    joblib.dump(scaler, output_path / "scaler.joblib")
    joblib.dump(imputer, output_path / "imputer.joblib")

    with open(output_path / "config.json", "w", encoding="utf-8") as config_file:
        json.dump({**MODEL_CONFIG}, config_file, indent=2)


def load_model() -> tuple[ChurnMLP, object, object]:
    """
    Carrega modelo MLP, scaler e imputer do disco.

    Returns:
        Tupla (modelo ChurnMLP em eval(), scaler, imputer).

    Raises:
        FileNotFoundError: Se os arquivos de modelo não forem encontrados.
    """
    model_path = Path(DATA_PATHS["models"])
    model_file = model_path / "model.pt"
    scaler_file = model_path / "scaler.joblib"
    imputer_file = model_path / "imputer.joblib"

    if not model_file.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_file}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler não encontrado em: {scaler_file}")
    if not imputer_file.exists():
        raise FileNotFoundError(f"Imputer não encontrado em: {imputer_file}")

    model = ChurnMLP(
        input_dim=MODEL_CONFIG["input_dim"],
        hidden_layers=MODEL_CONFIG["hidden_layers"],
        dropout=MODEL_CONFIG["dropout"],
    )
    model.load_state_dict(torch.load(model_file, weights_only=True))

    scaler = joblib.load(scaler_file)
    imputer = joblib.load(imputer_file)

    return model.eval(), scaler, imputer