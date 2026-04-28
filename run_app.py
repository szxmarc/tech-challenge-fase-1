#!/usr/bin/env python
"""
🚀 Ponto de Entrada da Aplicação Flask

Execute este arquivo para iniciar o servidor REST API:
    python run_app.py

A API estará disponível em:
    http://localhost:5000
    
Documentação de endpoints:
    GET  /health              - Verifica saúde da API
    GET  /api/v1/health       - Verifica saúde do modelo
    GET  /api/v1/model/info   - Informações do modelo
    GET  /api/v1/features     - Lista de features esperadas
    POST /api/v1/predict      - Predição individual
    POST /api/v1/predict/batch - Predição em batch
"""

from src.api.app import create_app
import os

if __name__ == "__main__":
    # Configuração
    config_name = os.getenv('FLASK_ENV', 'development')
    
    print("=" * 70)
    print("🚀 Iniciando Aplicação Flask - Tech Challenge FIAP")
    print("=" * 70)
    print(f"📍 Ambiente: {config_name}")
    print(f"🔗 URL: http://localhost:5000")
    print(f"📚 Docs: http://localhost:5000/api/v1/features")
    print("=" * 70)
    print()
    
    # Criar e rodar app
    app = create_app(config_name=config_name)
    app.run(host='0.0.0.0', port=5000, debug=(config_name == 'development'))
