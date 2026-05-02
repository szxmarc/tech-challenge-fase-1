"""
Módulo de modelos — arquitetura, treinamento e persistência.

Pacotes internos:
- trainer.py: ChurnMLP, train_mlp, save_model, load_model
"""

from .trainer import ChurnMLP, load_model, save_model, train_mlp

__all__ = ["ChurnMLP", "train_mlp", "save_model", "load_model"]