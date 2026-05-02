"""
Módulo de predição — serviço de inferência.

Responsabilidades:
  service.py: Orquestra o fluxo completo de predição com pipeline MLP PyTorch.
  Pipeline: feature engineering → encoding → imputer → scaler → inferência
"""

from .service import predict_batch, predict_single

__all__ = [
    "predict_single",
    "predict_batch",
]