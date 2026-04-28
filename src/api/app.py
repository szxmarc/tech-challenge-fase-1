"""
🚀 Configuração e Inicialização da Aplicação Flask

Cria a instância da Flask app com todas as configurações
e registra os blueprints da API.
"""

from flask import Flask, jsonify
from src.config.settings import MLFLOW_CONFIG, DATA_PATHS
import logging


def create_app(config_name='development'):
    """
    Cria e configura a aplicação Flask.
    
    Args:
        config_name: Nome da configuração ('development', 'testing', 'production')
    
    Returns:
        Aplicação Flask configurada
    """
    app = Flask(__name__)
    
    # ============================================================================
    # Configurações
    # ============================================================================
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    if config_name == 'development':
        app.config['DEBUG'] = True
    elif config_name == 'production':
        app.config['DEBUG'] = False
    
    # ============================================================================
    # Logging
    # ============================================================================
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    app.logger.info(f"🚀 Flask app iniciada em modo: {config_name}")
    
    # ============================================================================
    # Health Check
    # ============================================================================
    @app.route('/health', methods=['GET'])
    def health():
        """Verifica se a API está viva."""
        return jsonify({
            'status': 'healthy',
            'message': '✅ API está funcionando'
        }), 200
    
    # ============================================================================
    # Registrar Blueprints
    # ============================================================================
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    # ============================================================================
    # Error Handlers
    # ============================================================================
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint não encontrado',
            'status': 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Erro interno do servidor',
            'status': 500
        }), 500
    
    app.logger.info("✅ Blueprints registrados")
    
    return app
