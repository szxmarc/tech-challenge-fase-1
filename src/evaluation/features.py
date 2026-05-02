"""
Análise de Feature Importance para MLP PyTorch.

Usa permutation importance via sklearn, compatível com qualquer modelo
que implemente a interface de predição.
"""

import numpy as np
import pandas as pd
import torch


def get_feature_importance_mlp(
    model, X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """
    Calcula importância de features por permutação para o ChurnMLP.

    Para cada feature, permuta seus valores e mede a queda no AUC-ROC.
    Features com maior queda são mais importantes.

    Args:
        model:         Modelo ChurnMLP treinado em eval().
        X:             Array de features (já escalado).
        y:             Array de labels reais.
        feature_names: Lista de nomes das features.

    Returns:
        DataFrame com features ordenadas por importância decrescente.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        baseline_proba = torch.sigmoid(model(X_tensor)).numpy().ravel()
    baseline_auc = roc_auc_score(y, baseline_proba)

    importances = []
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        X_perm_tensor = torch.tensor(X_permuted, dtype=torch.float32)

        with torch.no_grad():
            perm_proba = torch.sigmoid(model(X_perm_tensor)).numpy().ravel()

        perm_auc = roc_auc_score(y, perm_proba)
        importances.append(baseline_auc - perm_auc)

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return feature_importance_df


def rank_features(feature_importance_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Retorna as Top N features mais importantes.

    Args:
        feature_importance_df: DataFrame com importâncias.
        top_n: Número de features a retornar.

    Returns:
        DataFrame com top N features.
    """
    return feature_importance_df.head(top_n)