"""
Utilitários gerais.

Funções auxiliares para manipulação de arquivos e registro de experimentos.
Sem dependências do domínio do projeto — podem ser reutilizadas em qualquer módulo.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def create_directories(paths) -> None:
    """
    Cria diretórios recursivamente se não existirem.

    Args:
        paths: Caminho único (str/Path) ou lista de caminhos.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info("Diretório criado/verificado: %s", path)


def save_json(data: dict, filepath: str | Path) -> None:
    """
    Serializa um dicionário para arquivo JSON.

    Args:
        data:     Dicionário a serializar.
        filepath: Caminho de destino do arquivo.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("JSON salvo: %s", filepath)


def load_json(filepath: str | Path) -> dict:
    """
    Deserializa um arquivo JSON para dicionário.

    Args:
        filepath: Caminho do arquivo JSON.

    Returns:
        Dicionário com o conteúdo do arquivo.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def log_experiment(model_name: str, metrics: dict, params: dict | None = None) -> None:
    """
    Registra os resultados de um experimento em arquivo JSON.

    O arquivo é salvo em logs/experiments.jsonl (um registro por linha),
    permitindo acumular múltiplos experimentos sem sobrescrever.

    Args:
        model_name: Identificador do modelo treinado.
        metrics:    Dicionário com métricas de avaliação.
        params:     Dicionário com hiperparâmetros utilizados (opcional).
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "metrics": metrics,
        "params": params or {},
    }

    log_file = Path("logs") / "experiments.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    logger.info("Experimento registrado: %s — %s", model_name, log_file)
