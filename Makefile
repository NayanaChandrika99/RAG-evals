# Makefile for RAG Ops Framework

.PHONY: setup install ingest run eval up down clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies (uses conda env 'rag-ops')"
	@echo "  make ingest     - Run document ingestion"
	@echo "  make run        - Start Streamlit application"
	@echo "  make eval       - Run evaluation"
	@echo "  make up         - Start Docker stack"
	@echo "  make down       - Stop Docker containers"
	@echo "  make clean      - Remove cache files"
	@echo ""
	@echo "Note: Using conda environment 'rag-ops' (Python 3.12)"
	@echo "Activate with: conda activate rag-ops"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo ""
	@echo "Dependencies installed successfully!"

ingest:
	@echo "Running document ingestion..."
	python scripts/ingest.py
	@echo "Ingestion complete!"

run:
	@echo "Starting Streamlit application..."
	streamlit run app.py

eval:
	@echo "Running evaluation..."
	python scripts/run_evaluation.py
	@echo "Evaluation complete!"

up:
	@echo "Starting Docker stack..."
	docker-compose up -d
	@echo "Docker stack running!"
	@echo "  - Streamlit UI: http://localhost:8501"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000"

down:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "Docker stack stopped!"

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cache files cleaned!"
