"""
🔧 Funções Auxiliares Gerais

Utilitários para criar diretórios, logging, manipulação de JSON, etc.
"""

import json
from pathlib import Path
from datetime import datetime


def create_directories(paths):
    """
    Cria diretórios necessários se não existirem.
    
    Args:
        paths: String ou lista de strings com caminhos
    """
    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Diretório criado/verificado: {path}")


def save_json(data: dict, filepath: str):
    """
    Salva dicionário em arquivo JSON.
    
    Args:
        data: Dicionário a salvar
        filepath: Caminho do arquivo
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ JSON salvo: {filepath}")


def load_json(filepath: str) -> dict:
    """
    Carrega dicionário de arquivo JSON.
    
    Args:
        filepath: Caminho do arquivo
    
    Returns:
        Dicionário carregado
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def log_experiment(model_name: str, metrics: dict, params: dict = None):
    """
    Registra informações do experimento em um arquivo de log.
    
    Args:
        model_name: Nome do modelo
        metrics: Dicionário com métricas
        params: Dicionário com parâmetros (opcional)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "timestamp": timestamp,
        "model": model_name,
        "metrics": metrics,
        "params": params or {}
    }
    
    print(f"📝 Log: {timestamp} - {model_name}")
