"""
Factory da aplicação FastAPI.

Cria e configura a instância da aplicação, inclui o router da API
e define o handler de erro para rotas não encontradas.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _save_selected_api_names(api_names: list[str]) -> None:
    """Persiste os nomes camelCase das features originais necessárias em feature_names.txt."""
    from pathlib import Path

    from src.config.settings import DATA_PATHS
    Path(DATA_PATHS["feature_names"]).write_text("\n".join(api_names))


def _resolve_required_api_names(selected_model_names: list[str]) -> list[str]:
    """
    Determina os nomes camelCase das features originais que a API deve receber.

    Para features diretas: converte via MODEL_TO_API_MAPPING.
    Para features derivadas: expande para as features-fonte originais.
    Retorna a lista deduplcada na ordem do FEATURE_NAME_MAPPING.
    """
    from src.data.processor import DERIVED_FEATURE_SOURCES
    from src.prediction.feature_mapping import FEATURE_NAME_MAPPING, MODEL_TO_API_MAPPING

    required_model_names: set[str] = set()
    for name in selected_model_names:
        if name in MODEL_TO_API_MAPPING:
            required_model_names.add(name)
        elif name in DERIVED_FEATURE_SOURCES:
            required_model_names.update(DERIVED_FEATURE_SOURCES[name])

    seen: set[str] = set()
    api_names: list[str] = []
    for api_name, model_name in FEATURE_NAME_MAPPING.items():
        if model_name in required_model_names and api_name not in seen:
            api_names.append(api_name)
            seen.add(api_name)

    return api_names


def _train_on_startup() -> None:
    """Pipeline de treino executado ao iniciar a aplicação.

    Ordem:
      1. Carrega e pré-processa os dados brutos
      2. Cria features derivadas (engenharia de features)
      3. Seleciona features via Random Forest (cobertura ≥ 90% de importância)
      4. Treina Regressão Logística (baseline) nas features selecionadas
      5. Treina MLP PyTorch (produção) nas features selecionadas
      6. Compara os modelos no conjunto de teste e persiste comparison.json
      7. Registra experimento no MLflow (parâmetros + métricas de ambos os modelos)
      8. Persiste feature_names.txt com os nomes camelCase das features originais necessárias
    """
    import json
    from pathlib import Path

    from src.config.settings import (
        DATA_PATHS,
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI,
        MLP_CONFIG,
        MODEL_CONFIG,
    )
    from src.data.loader import load_raw_data
    from src.data.processor import (
        encode_features,
        engineer_features,
        save_processed_data,
        select_features,
        split_and_scale_data,
    )
    from src.evaluation.metrics import compare_models
    from src.models.trainer import save_mlp_model, save_model, train_logistic_regression, train_mlp

    logger.info("Carregando dados brutos...")
    df = load_raw_data()

    logger.info("Pré-processando features...")
    df_encoded = encode_features(df)

    logger.info("Calculando features derivadas...")
    df_engineered = engineer_features(df_encoded)
    X_full = df_engineered.drop(columns=["Churn"])
    y = df_engineered["Churn"]

    logger.info("Selecionando features (Random Forest, cobertura ≥ 90%)...")
    selected_model_names = select_features(X_full, y)
    required_api_names = _resolve_required_api_names(selected_model_names)
    logger.info("%d features selecionadas de %d: %s", len(selected_model_names), len(X_full.columns), selected_model_names)

    X = X_full[selected_model_names]
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)

    logger.info("Treinando baseline (Regressão Logística)...")
    lr_model = train_logistic_regression(X_train, y_train)
    save_model(lr_model, scaler)

    logger.info("Treinando modelo de produção (MLP PyTorch)...")
    mlp_model = train_mlp(X_train, y_train)
    save_mlp_model(mlp_model, scaler)

    logger.info("Comparando modelos no conjunto de teste...")
    comparison = compare_models(lr_model, mlp_model, X_test, y_test)
    comparison_path = Path(DATA_PATHS["comparison"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Comparação concluída — recomendação: %s | MLP F1=%.4f, LR F1=%.4f",
        comparison["recommendation"],
        comparison["models"]["mlp"]["metrics"]["f1Score"],
        comparison["models"]["logisticRegression"]["metrics"]["f1Score"],
    )

    # Rastreamento MLflow — não-bloqueante: falha silenciosa se o backend estiver indisponível
    try:
        import mlflow.pytorch
        import mlflow.sklearn

        mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="churn_baseline_vs_mlp_pytorch") as run:
            mlflow.log_params({f"lr_{k}": str(v) for k, v in MODEL_CONFIG.items()})
            mlflow.log_params({f"mlp_{k}": str(v) for k, v in MLP_CONFIG.items()})
            logger.info("MLflow | parâmetros registrados | run_id=%s", run.info.run_id)

            for model_name, model_data in comparison["models"].items():
                for metric, value in model_data["metrics"].items():
                    mlflow.log_metric(f"{model_name}_{metric}", value)
            logger.info("MLflow | métricas de teste registradas (LR + MLP)")

            mlflow.sklearn.log_model(lr_model, "logistic_regression")
            logger.info("MLflow | artefato registrado: logistic_regression")

            mlflow.pytorch.log_model(mlp_model.model, "mlp_pytorch")
            logger.info("MLflow | artefato registrado: mlp_pytorch")

            mlflow.log_artifact(str(comparison_path), artifact_path="reports")
            logger.info("MLflow | artefato registrado: reports/model_comparison.json")

            mlflow.set_tag("recommendation", comparison["recommendation"])
            logger.info(
                "MLflow | experimento concluído | run_id=%s recomendação=%s",
                run.info.run_id,
                comparison["recommendation"],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow tracking indisponível, continuando sem rastreamento: %s", exc)

    _save_selected_api_names(required_api_names)
    logger.info("Modelos treinados e salvos com sucesso.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _train_on_startup()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Churn Prediction API",
        version="1.0.0",
        description="API de predição de churn para clientes Telco - Tech Challenge Fase 1",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {"status": "healthy", "message": "API operacional"}

    from .routes import api_router
    app.include_router(api_router, prefix="/api/v1")

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            {"error": "Endpoint não encontrado", "status": 404},
            status_code=404,
        )

    return app
