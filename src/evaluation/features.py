"""
🔍 Análise de Feature Importance

Funções para extrair e rankear a importância das features.
"""

import numpy as np
import pandas as pd


def get_feature_importance(model, feature_names):
    """
    Extrai importância das features do modelo.
    
    Para modelos lineares (ex: LogisticRegression), usa os coeficientes.
    
    Args:
        model: Modelo treinado
        feature_names: Lista de nomes das features
    
    Returns:
        DataFrame com features e seus coeficientes
    """
    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        raise ValueError("Modelo não possui atributo 'coef_'")
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(importances),
        'coefficient': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df


def rank_features(feature_importance_df, top_n=10) -> pd.DataFrame:
    """
    Retorna as Top N features mais importantes.
    
    Args:
        feature_importance_df: DataFrame com importâncias
        top_n: Número de features a retornar
    
    Returns:
        DataFrame com top N features
    """
    return feature_importance_df.head(top_n)
