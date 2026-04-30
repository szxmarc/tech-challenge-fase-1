#!/usr/bin/env python
"""
Ponto de entrada da aplicação.

Execute para iniciar o servidor REST:
    python run_app.py

A API estará disponível em http://localhost:5000
Documentação interativa em http://localhost:5000/docs
"""

import uvicorn

from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("run_app:app", host="0.0.0.0", port=5000, reload=True)
