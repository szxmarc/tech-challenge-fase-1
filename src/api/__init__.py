"""
🌐 MÓDULO API - Aplicação Flask

Responsabilidade: Expor modelos ML através de uma API REST.

Arquivos:
- app.py: Configuração e inicialização da Flask app
- routes.py: Definição de endpoints e blueprints

Exemplo de uso:
    from src.api.app import create_app
    
    app = create_app()
    if __name__ == "__main__":
        app.run(debug=True)
"""

from .app import create_app
from .routes import api_bp

__all__ = ["create_app", "api_bp"]
