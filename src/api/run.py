#!/usr/bin/env python
"""
🚀 Ponto de Entrada da Aplicação Flask

Execute este arquivo para iniciar o servidor:
    python run.py

Ou se usar o script na raiz:
    python run_app.py
"""

from src.api.app import create_app

if __name__ == "__main__":
    app = create_app(config_name='development')
    app.run(host='0.0.0.0', port=5000, debug=True)
