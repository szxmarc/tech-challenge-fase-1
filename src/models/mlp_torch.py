"""
Arquitetura PyTorch e wrapper sklearn-compatível para o MLP de churn.
"""

import numpy as np
import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """MLP binário para predição de churn."""

    def __init__(self, input_size: int, hidden_sizes: tuple = (64, 32), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchMLPWrapper:
    """Wrapper sklearn-compatível para ChurnMLP.

    Expõe predict(), predict_proba(), classes_ e feature_names_in_
    para integração transparente com o pipeline existente (metrics, routes, service).
    """

    def __init__(self, model: ChurnMLP, feature_names: list[str], threshold: float = 0.5):
        self.model = model
        self.feature_names_in_ = np.array(feature_names)
        self.classes_ = np.array([0, 1])
        self.threshold = threshold

    def _to_tensor(self, X) -> torch.Tensor:
        if hasattr(X, "values"):
            X = X.values
        return torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_tensor(X)).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
        if probs.ndim == 0:
            probs = probs.reshape(1)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
