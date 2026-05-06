.PHONY: help install install-notebooks run test coverage lint lint-fix clean mlflow

# Alvo padrão: exibe os comandos disponíveis
help:
	@echo "Comandos disponíveis:"
	@echo ""
	@echo "  make install            Instala dependências principais + dev"
	@echo "  make install-notebooks  Instala dependências para notebooks (matplotlib, evidently, etc.)"
	@echo "  make run                Inicia o servidor FastAPI em http://localhost:8000"
	@echo "  make test               Executa a suíte de testes com pytest"
	@echo "  make coverage           Executa testes e abre o relatório HTML de cobertura"
	@echo "  make lint               Verifica estilo e erros com ruff"
	@echo "  make lint-fix           Corrige automaticamente problemas detectados pelo ruff"
	@echo "  make clean              Remove artefatos temporários e de build"
	@echo "  make mlflow             Inicia o MLflow Tracking UI em http://localhost:5000"

# Instala dependências principais e de desenvolvimento
# Instala dependências principais e de desenvolvimento
install:
	python -m pip install ".[dev]"

install-notebooks:
	python -m pip install ".[dev,notebooks]"
# Inicia o servidor FastAPI (com reload automático)
run:
	python run_app.py

# Executa a suíte completa de testes (configuração em pyproject.toml)
test:
	pytest

# Executa testes e abre o relatório de cobertura HTML no navegador
coverage:
	pytest
	python -m webbrowser reports/coverage/index.html

# Verifica código com ruff (sem alterar arquivos)
lint:
	ruff check src/ tests/

# Corrige automaticamente os problemas reportados pelo ruff
lint-fix:
	ruff check --fix src/ tests/

# Remove arquivos temporários, caches e artefatos de build
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .coverage htmlcov/ reports/coverage/
	rm -rf dist/ build/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/

# Inicia o servidor de experimentos do MLflow
mlflow:
	mlflow ui --port 5000
