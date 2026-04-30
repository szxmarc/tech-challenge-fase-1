"""
Carregamento de dados.

Funções para leitura do dataset bruto e dos dados processados.
Usado em notebooks de treinamento — não é chamado pela API em tempo de execução.
"""

import logging

import joblib
import pandas as pd

from src.config.settings import (
    RAW_DATASET,
    SCALER_FILE,
    X_TEST_FILE,
    X_TRAIN_FILE,
    Y_TEST_FILE,
    Y_TRAIN_FILE,
)

logger = logging.getLogger(__name__)


def load_raw_data(filepath=None) -> pd.DataFrame:
    """
    Carrega o dataset bruto a partir de um CSV.

    Args:
        filepath: Caminho alternativo para o arquivo. Se None, usa RAW_DATASET de settings.

    Returns:
        DataFrame com os dados brutos.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    from pathlib import Path

    target = Path(filepath) if filepath else RAW_DATASET

    if not target.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {target}")

    df = pd.read_csv(target)
    logger.info("Dataset carregado: %d linhas × %d colunas", df.shape[0], df.shape[1])
    return df


def load_processed_data() -> tuple:
    """
    Carrega os dados já processados (output de save_processed_data).

    Returns:
        Tupla (X_train, X_test, y_train, y_test, scaler).

    Raises:
        FileNotFoundError: Se algum dos arquivos processados não existir.
    """
    for path in [X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, SCALER_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Arquivo processado não encontrado: {path}")

    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).iloc[:, 0]
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]
    scaler = joblib.load(SCALER_FILE)

    logger.info("Dados processados carregados — treino: %s, teste: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test, scaler
