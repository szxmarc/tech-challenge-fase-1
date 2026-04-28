"""
🤖 MÓDULO MODELS - Treinamento e Persistência

Responsabilidade: Encapsular lógica de treinamento, validação e
persistência de modelos.

Arquivo Principal:
- trainer.py: Funções de treinamento e salvamento de modelos

Exemplo de uso:
    from src.models.trainer import train_logistic_regression, save_model, load_model
"""

from .trainer import train_logistic_regression, save_model, load_model

__all__ = ["train_logistic_regression", "save_model", "load_model"]
