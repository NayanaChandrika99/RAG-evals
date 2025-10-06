       # RAG Ops Framework

A production MLOps framework for Retrieval-Augmented Generation (RAG) systems with automated quality assurance and observability.

<!-- Testing CI/CD pipeline -->

       ## Overview

       This framework demonstrates end-to-end AI engineering practices for RAG systems, including:
       - Automated evaluation with RAGAS metrics
       - CI/CD quality gates to prevent regressions
       - Full observability stack with LangSmith, Prometheus, and Grafana
       - Containerized deployment with Docker

       ## Architecture

       **Hybrid Approach:**
       - **Embeddings**: Local `all-mpnet-base-v2` model (sentence-transformers)
       - **Vector Store**: FAISS for similarity search
       - **Generation**: OpenAI GPT-3.5-turbo
       - **Evaluation**: RAGAS framework with GPT-3.5-turbo as judge

       ```
       ┌─────────────┐      ┌──────────────┐      ┌─────────────┐
       │   User      │─────>│  Streamlit   │─────>│ RAG Chain   │
       │  Question   │<─────│     UI       │<─────│  (LCEL)     │
       └─────────────┘      └──────────────┘      └─────────────┘
                                   │                       │
                                   │                       ├─> Local Embeddings
                                   │                       ├─> FAISS Retrieval
                                   │                       └─> GPT-3.5 Generation
                                   │
                            ┌──────▼───────┐
                            │ Observability │
                            ├──────────────┤
                            │ • LangSmith   │
                            │ • Prometheus  │
                            │ • Grafana     │
                            └───────────────┘
       ```

       ## Quick Start

       ```bash
       # Install dependencies
       pip install -r requirements.txt

       # Configure API keys in .env
       OPENAI_API_KEY=sk-...
       LANGSMITH_API_KEY=lsv2_pt_...

       # Ingest documents
       python scripts/ingest.py

       # Run application
       streamlit run app.py

       # Run evaluation
       python scripts/run_evaluation.py
       ```

       ## Evaluation Metrics

       The system tracks four key RAGAS metrics:

       | Metric | Target | Description |
       |--------|--------|-------------|
       | Faithfulness | ≥0.80 | Answer grounded in retrieved context |
       | Answer Relevancy | ≥0.80 | Answer addresses the question |
       | Context Recall | ≥0.75 | All necessary information retrieved |
       | Context Precision | ≥0.70 | Relevant contexts ranked highly |

       **Current Baseline**: 75% pass rate (3/4 metrics above threshold)

       ## CI/CD Quality Gates

       Automated workflows ensure quality:

       **PR Check** (`.github/workflows/pr_check.yml`):
       - Runs evaluation on golden dataset
       - Compares metrics to baseline
       - Blocks merge if metrics degrade >5%

       **Baseline Update** (`.github/workflows/update_baseline.yml`):
       - Updates baseline on main branch merges
       - Maintains quality standards over time

       ### GitHub Setup

       Add repository secret:
       ```
       OPENAI_API_KEY=sk-...
       ```

       ## Observability

       ### LangSmith Tracing

       View detailed traces at https://smith.langchain.com
       - Complete RAG chain execution
       - Individual component timings
       - Input/output logging

       ### Prometheus Metrics

       Available at `/metrics`:
       - `rag_questions_total`: Total questions asked
       - `rag_response_seconds`: Response time distribution
       - `rag_errors_total`: Error count

       ### Grafana Dashboards

       Access at http://localhost:3000 (via Docker Compose)

       ## Docker Deployment

       ```bash
       # Start full stack
       docker-compose up -d

       # Services:
       # - Streamlit: http://localhost:8501
       # - Prometheus: http://localhost:9090
       # - Grafana: http://localhost:3000
       ```

       ## Project Structure

       ```
       /rag-ops-framework
       ├── .github/workflows/      # CI/CD pipelines
       ├── data/                   # Documents & test datasets
       ├── scripts/
       │   ├── ingest.py          # Document ingestion
       │   ├── run_evaluation.py  # RAGAS evaluation
       │   └── compare_metrics.py # Baseline comparison
       ├── src/
       │   └── rag_chain.py       # Core RAG logic (LCEL)
       ├── app.py                 # Streamlit UI
       ├── baseline_report.json   # Quality baseline
       └── requirements.txt       # Dependencies
       ```

       ## Tech Stack

       **Core**: Python 3.12, LangChain, OpenAI, sentence-transformers, FAISS
       **Evaluation**: RAGAS
       **Observability**: LangSmith, Prometheus, Grafana
       **Infrastructure**: Docker, GitHub Actions, Streamlit


       ## License

       MIT
