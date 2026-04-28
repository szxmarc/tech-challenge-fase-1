"""
📍 Rotas da API

Define todos os endpoints da aplicação Flask usando blueprints.

Endpoints:
- POST /api/v1/predict - Predição de churn
- GET /api/v1/model/info - Informações do modelo
- GET /api/v1/features - Lista de features
"""

from flask import Blueprint, request, jsonify
from src.models.trainer import load_model
from src.data.loader import load_processed_data
from src.evaluation.metrics import evaluate_model
from src.config.settings import DATA_PATHS
import pandas as pd
import logging

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# ============================================================================
# Variáveis Globais (Cache)
# ============================================================================
_model = None
_scaler = None
_feature_names = None


def get_model():
    """Carrega o modelo treinado (com cache)."""
    global _model, _scaler
    
    if _model is None:
        try:
            _model, _scaler = load_model()
            logger.info("✅ Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    return _model, _scaler


def get_feature_names():
    """Obtém nomes das features."""
    global _feature_names
    
    if _feature_names is None:
        try:
            from pathlib import Path
            feature_file = Path(DATA_PATHS["project_root"]) / "models" / "logistic_regression" / "feature_names.txt"
            with open(feature_file, 'r') as f:
                _feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"✅ {len(_feature_names)} features carregadas")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar features: {e}")
            raise
    
    return _feature_names


# ============================================================================
# ENDPOINT 1: Health Check
# ============================================================================
@api_bp.route('/health', methods=['GET'])
def health():
    """
    Verifica se a API está operacional.
    
    Returns:
        JSON com status da API
    """
    try:
        get_model()  # Tenta carregar o modelo
        return jsonify({
            'status': 'healthy',
            'message': '✅ API e modelo operacionais'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


# ============================================================================
# ENDPOINT 2: Informações do Modelo
# ============================================================================
@api_bp.route('/model/info', methods=['GET'])
def model_info():
    """
    Retorna informações sobre o modelo treinado.
    
    Returns:
        JSON com detalhes do modelo (tipo, features, etc)
    """
    try:
        model, _ = get_model()
        feature_names = get_feature_names()
        
        return jsonify({
            'model_type': 'LogisticRegression',
            'n_features': len(feature_names),
            'n_classes': model.n_classes_,
            'classes': model.classes_.tolist(),
            'model_path': DATA_PATHS.get('models', 'N/A'),
        }), 200
    except Exception as e:
        logger.error(f"❌ Erro ao obter info do modelo: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ENDPOINT 3: Lista de Features
# ============================================================================
@api_bp.route('/features', methods=['GET'])
def features():
    """
    Retorna lista de features esperadas pela API.
    
    Returns:
        JSON com nomes das features
    """
    try:
        feature_names = get_feature_names()
        return jsonify({
            'n_features': len(feature_names),
            'features': feature_names,
        }), 200
    except Exception as e:
        logger.error(f"❌ Erro ao listar features: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ENDPOINT 4: Predição de Churn
# ============================================================================
@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Realiza predição de churn para um cliente.
    
    Request Body:
        JSON com features do cliente
        
    Example:
        {
            "feature_1": 0.5,
            "feature_2": 1.0,
            ...
        }
    
    Returns:
        JSON com predição (classe) e probabilidade
    """
    try:
        # Validar request
        if not request.json:
            return jsonify({'error': 'Body deve ser JSON'}), 400
        
        # Carregar modelo
        model, scaler = get_model()
        feature_names = get_feature_names()
        
        # Converter input em DataFrame
        input_data = request.json
        
        # Validar features
        missing_features = [f for f in feature_names if f not in input_data]
        if missing_features:
            return jsonify({
                'error': f'Features faltando: {missing_features}'
            }), 400
        
        # Criar DataFrame com ordem correta
        X = pd.DataFrame([input_data])[feature_names]
        
        # Validar tipos/valores
        try:
            X = X.astype(float)
        except ValueError as e:
            return jsonify({'error': f'Tipo de dados inválido: {str(e)}'}), 400
        
        # Escalar features
        X_scaled = scaler.transform(X)
        
        # Predição
        y_pred = model.predict(X_scaled)[0]
        y_pred_proba = model.predict_proba(X_scaled)[0]
        
        # Resposta
        return jsonify({
            'prediction': int(y_pred),
            'prediction_label': 'Churn' if y_pred == 1 else 'Não Churn',
            'probability_no_churn': float(y_pred_proba[0]),
            'probability_churn': float(y_pred_proba[1]),
            'confidence': float(max(y_pred_proba)),
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ENDPOINT 5: Predição em Batch
# ============================================================================
@api_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Realiza predição em batch para múltiplos clientes.
    
    Request Body:
        JSON com lista de objetos cliente
        
    Example:
        {
            "customers": [
                {"feature_1": 0.5, "feature_2": 1.0, ...},
                {"feature_1": 0.3, "feature_2": 0.8, ...}
            ]
        }
    
    Returns:
        JSON com lista de predições
    """
    try:
        if not request.json or 'customers' not in request.json:
            return jsonify({'error': 'Body deve conter "customers"'}), 400
        
        customers = request.json['customers']
        
        if not isinstance(customers, list):
            return jsonify({'error': '"customers" deve ser uma lista'}), 400
        
        # Carregar modelo
        model, scaler = get_model()
        feature_names = get_feature_names()
        
        # Processar cada cliente
        predictions = []
        for i, customer in enumerate(customers):
            try:
                # Validar features
                missing_features = [f for f in feature_names if f not in customer]
                if missing_features:
                    predictions.append({
                        'customer_index': i,
                        'error': f'Features faltando: {missing_features}'
                    })
                    continue
                
                # Criar DataFrame
                X = pd.DataFrame([customer])[feature_names]
                X = X.astype(float)
                
                # Escalar
                X_scaled = scaler.transform(X)
                
                # Predição
                y_pred = model.predict(X_scaled)[0]
                y_pred_proba = model.predict_proba(X_scaled)[0]
                
                predictions.append({
                    'customer_index': i,
                    'prediction': int(y_pred),
                    'prediction_label': 'Churn' if y_pred == 1 else 'Não Churn',
                    'probability_no_churn': float(y_pred_proba[0]),
                    'probability_churn': float(y_pred_proba[1]),
                })
            except Exception as e:
                predictions.append({
                    'customer_index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'total': len(customers),
            'predictions': predictions,
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Erro na predição em batch: {e}")
        return jsonify({'error': str(e)}), 500
