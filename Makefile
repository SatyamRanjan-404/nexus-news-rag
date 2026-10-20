# Nexus News RAG — Makefile
# Commands: make test, make eval, make run, make lint, make build

.PHONY: help install test test-unit test-integration lint format run-api run-ui build clean

## Default: show help
help:
	@echo "Nexus News RAG — Available commands:"
	@echo ""
	@echo "  make install       Install all dependencies (editable)"
	@echo "  make install-dev   Install dev dependencies (pytest, ruff, pre-commit)"
	@echo "  make run-api       Start the FastAPI backend"
	@echo "  make run-ui        Start the Streamlit frontend"
	@echo "  make run           Start both (requires tmux)"
	@echo "  make test          Run all unit tests"
	@echo "  make test-unit     Run unit tests only (fast, no API keys)"
	@echo "  make eval          Run the full evaluation suite"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Run ruff formatter"
	@echo "  make build         Build Docker images"
	@echo "  make clean         Remove caches and build artifacts"

## Install
install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

## Running
run-api:
	uvicorn nexus.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run src/nexus/ui/streamlit_app.py --server.port 8501

## Testing
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v -m "not slow" --tb=short

test-cov:
	pytest tests/ --cov=nexus --cov-report=term-missing --cov-report=html

## Evaluation (Phase 7-8)
eval:
	python scripts/run_evaluation.py

eval-retrieval:
	python scripts/eval_retrieval.py

eval-generation:
	python scripts/eval_generation.py

## Code quality
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

format-check:
	ruff format --check src/ tests/

## Docker
build:
	docker-compose build

up:
	docker-compose up

up-build:
	docker-compose up --build

## Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
