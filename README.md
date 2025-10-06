# 🤖 Production-Grade RAG Ops Framework

[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com/yourusername/rag-ops-framework/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-green)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps framework for RAG (Retrieval-Augmented Generation) systems demonstrating industry best practices for AI engineering.

## ✨ Key Features

- **🎯 Hybrid Architecture**: Local embeddings + proprietary LLMs for cost-quality balance
- **📊 Automated Quality Assurance**: RAGAS-based evaluation with CI/CD quality gates
- **🔍 Production Observability**: LangSmith tracing, Prometheus metrics, Grafana dashboards
- **🔒 Security Best Practices**: Proper API key management with GitHub Secrets
- **🐳 Containerized Deployment**: Docker Compose for one-command startup
- **⚡ Professional Workflow**: Makefile for streamlined development

## 🏗️ Architecture

This project uses a **strategic hybrid approach**:

- **Embeddings**: Local `instructor-large` model (100% free, ~100ms latency)
- **Generation**: OpenAI GPT-3.5-turbo (~$0.002 per query)
- **Evaluation**: OpenAI GPT-4 for reliable quality assessment

**Cost Savings**: 90% reduction vs. full API approach (~$0.002 vs. $0.006 per query)

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   User      │─────>│  Streamlit   │─────>│ RAG Chain   │
│  Question   │<─────│     UI       │<─────│  (LCEL)     │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                       │
                            │                       ├─> Local Embeddings (FREE)
                            │                       ├─> FAISS Vector Store
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

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional, for full stack)
- OpenAI API key

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-ops-framework.git
cd rag-ops-framework

# Activate conda environment
conda activate llm

# Install dependencies
make install

# Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Prepare Your Document

```bash
# Place your PDF in data/source_document.pdf
# Then run ingestion
make ingest
```

This will:
- Load and chunk your PDF
- Create embeddings using the local `instructor-large` model
- Save FAISS index to `faiss_index/`

### 3. Run the Application

**Option A: Local Development**
```bash
make run
```
Visit http://localhost:8501

**Option B: Full Docker Stack**
```bash
make up
```
- Streamlit UI: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## 📊 Evaluation

### Create Golden Dataset

1. Copy the template:
```bash
cp data/golden_dataset.example.json data/golden_dataset.json
```

2. Manually create 15-20 high-quality question-answer pairs based on your document

### Run Evaluation

```bash
make eval
```

This generates `evaluation_report.json` with RAGAS metrics:
- **Faithfulness** (≥0.80): Is the answer grounded in retrieved context?
- **Answer Relevance** (≥0.80): Does the answer address the question?
- **Context Relevance** (≥0.75): Are retrieved documents relevant?
- **Context Precision**: Are relevant contexts ranked highly?
- **Context Recall**: Is all necessary information retrieved?

## 🔄 CI/CD Quality Gates

### Automated Quality Checks

Every pull request automatically:
1. Runs evaluation on golden dataset
2. Compares metrics to baseline
3. **Blocks merge** if metrics degrade >5%
4. Posts results as PR comment

### Setup GitHub Secrets

```bash
# In your GitHub repository:
# Settings → Secrets and variables → Actions → New repository secret

OPENAI_API_KEY=sk-...
```

### Workflows

- **PR Check** (`.github/workflows/pr_check.yml`): Validates quality on every PR
- **Update Baseline** (`.github/workflows/update_baseline.yml`): Updates baseline on main branch

## 📈 Observability

### LangSmith Tracing

Enable in `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_...
```

View traces at https://smith.langchain.com

### Prometheus Metrics

Available at http://localhost:8000/metrics:
- `rag_questions_total`: Total questions asked
- `rag_response_seconds`: Response time histogram
- `rag_errors_total`: Error count

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Add Prometheus datasource: http://prometheus:9090
3. Import dashboard or create custom panels

## 💰 Cost Management

### Cost Breakdown

| Component | Model | Cost |
|-----------|-------|------|
| Embeddings | instructor-large (local) | $0.00 |
| Generation | gpt-3.5-turbo | ~$0.002/query |
| Evaluation (20 questions) | gpt-4-turbo | ~$0.10-0.20 |

### Optimization Strategies

**For Production**:
- Use GPT-3.5 for PR check evaluations (~$0.05/run)
- Reserve GPT-4 for weekly baseline updates (~$0.20/run)
- Always use local embeddings (saves ~$0.0004/query)

## 🛠️ Development

### Makefile Commands

```bash
make install    # Install dependencies (uses conda env 'llm')
make ingest     # Run document ingestion
make run        # Start Streamlit app
make eval       # Run evaluation
make up         # Start Docker stack
make down       # Stop Docker containers
make clean      # Remove cache files
```

### Project Structure

```
/rag-ops-framework
├── .github/workflows/      # CI/CD pipelines
├── data/                   # Source docs & datasets
├── scripts/                # Helper scripts
│   ├── ingest.py          # Document ingestion
│   ├── run_evaluation.py  # RAGAS evaluation
│   └── compare_metrics.py # Baseline comparison
├── src/
│   └── rag_chain.py       # Core RAG logic (LCEL)
├── app.py                 # Streamlit UI
├── docker-compose.yml     # Multi-container orchestration
├── Dockerfile            # App containerization
├── Makefile              # Development commands
├── prometheus.yml        # Metrics configuration
└── requirements.txt      # Python dependencies
```

## 🔒 Security

**API Key Management**:
- ✅ Local: `.env` file (never commit!)
- ✅ CI/CD: GitHub Secrets
- ✅ Docker: Environment variables

**FAISS Deserialization**:
- Uses `allow_dangerous_deserialization=True` (safe because we control the source)
- Production systems with untrusted sources require additional validation

## 📚 Tech Stack

**Core**:
- Python 3.11
- LangChain (LCEL for chains)
- OpenAI (GPT-3.5/4)
- sentence-transformers (local embeddings)
- FAISS (vector store)

**Evaluation**:
- RAGAS
- datasets

**Observability**:
- LangSmith
- Prometheus
- Grafana

**Infrastructure**:
- Docker & Docker Compose
- GitHub Actions
- Streamlit

## 🎯 Success Metrics

**Quality Targets**:
- Faithfulness: ≥0.80
- Answer Relevance: ≥0.80
- Context Relevance: ≥0.75
- Context Precision: ≥0.70

**Operational Targets**:
- CI/CD pipeline: <10 minutes
- Response time (p95): <3 seconds
- Evaluation cost: <$0.20/run
- False positive rate: <5%

## 🚧 Troubleshooting

### "FAISS index not found"
Run `make ingest` to process your document first.

### "OpenAI API key not found"
Check `.env` file exists and contains valid API key.

### "GitHub Actions failing"
Verify GitHub Secrets are configured correctly.

### Docker can't access FAISS index
Check volume mount in `docker-compose.yml`.

## 📖 Learn More

- [CLAUDE.md](./CLAUDE.md) - Detailed implementation guide
- [final.md](./final.md) - Complete project blueprint
- [LangChain Docs](https://python.langchain.com/)
- [RAGAS Docs](https://docs.ragas.io/)
- [LangSmith Docs](https://docs.smith.langchain.com/)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run evaluation to ensure quality
5. Submit a pull request

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [RAGAS](https://github.com/explodinggradients/ragas)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

---

**Built with ❤️ as a demonstration of production-grade AI engineering practices**
