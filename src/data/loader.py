"""
Módulo de Carregamento de Dados (Data Loader)
==============================================

Propósito:
----------
Funções para CARREGAR dados de várias fontes (CSV, JSON, etc.).
Centraliza lógica de leitura com tratamento de erros e validações.

Funções:
- load_raw_data(): Carrega dataset bruto do arquivo CSV
- load_processed_data(): Carrega dados já processados (treino/teste/scaler)

Razão de existência:
- Evita código duplicado de leitura em vários notebooks
- Facilita mudar fonte de dados sem alterar múltiplos arquivos
- Padroniza validações (ex: verificar se arquivo existe)

Exemplo de uso em notebook:
    from src.data import load_raw_data
    df = load_raw_data()
    print(df.shape)  # (7043, 21)
"""

import pandas as pd
from pathlib import Path
from src.config import RAW_DATASET, X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, SCALER_FILE
import joblib


def load_raw_data(filepath=None):
    """
    Carrega o dataset bruto.
    
    Parâmetros:
    -----------
    filepath : str or Path, optional
        Caminho para arquivo CSV. Se None, usa RAW_DATASET de config.
    
    Retornos:
    ---------
    pd.DataFrame
        Dataset carregado com shape (7043, 21)
    
    Raises:
    -------
    FileNotFoundError
        Se arquivo não existe
    
    Exemplo:
        df = load_raw_data()
        df = load_raw_data("../data/raw/custom_data.csv")
    """
    if filepath is None:
        filepath = RAW_DATASET
    else:
        filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {filepath}")
    
    print(f"Carregando dataset de: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    
    return df


def load_processed_data():
    """
    Carrega dados JÁ PROCESSADOS (output de 01_prepare_data.ipynb).
    
    Carrega:
    - X_train, X_test, y_train, y_test (já padronizados)
    - scaler (StandardScaler fit no treino)
    
    Uso em 02_train_logistic_regression.ipynb:
        X_train, X_test, y_train, y_test, scaler = load_processed_data()
    
    Retornos:
    ---------
    tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("Carregando dados processados...")
    
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).iloc[:, 0]
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]
    scaler = joblib.load(SCALER_FILE)
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler
