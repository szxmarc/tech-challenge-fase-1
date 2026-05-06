"""
Módulo de modelos — treinamento e persistência.

Pacotes internos:
- trainer.py: treinar, salvar e carregar o modelo de regressão logística
"""

from .trainer import load_model, save_model, train_logistic_regression

__all__ = ["train_logistic_regression", "save_model", "load_model"]
