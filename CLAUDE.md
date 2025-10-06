# Production-Grade RAG Ops Framework - Claude Instructions

## Project Context

You are assisting with the development of a **Production-Grade RAG Ops Framework** - an MLOps system for a Retrieval-Augmented Generation (RAG) chatbot. This project demonstrates end-to-end AI engineering skills including application development, automated quality assurance, CI/CD integration, production observability, and strategic cost management.

## Core Architecture Principles

### Hybrid Architecture Strategy
This project uses a **strategic hybrid approach**:
- **Local Embeddings**: Use open-source models (e.g., `instructor-large`) for zero-cost vector embeddings
- **Proprietary LLMs**: Use best-in-class models (GPT-4, Claude) for generation and evaluation
- **Rationale**: Optimize costs where possible (embeddings) while maintaining quality where it matters (generation/evaluation)

This architecture mirrors real-world production systems at top tech companies.

## Project Structure

```
/rag-ops-framework
├── .github/workflows/         # CI/CD pipelines
│   ├── pr_check.yml          # Quality gates on PRs
│   └── update_baseline.yml   # Baseline updates on main
├── data/
│   ├── source_document.pdf   # Source document for RAG
│   └── golden_dataset.json   # Manual evaluation dataset
├── scripts/
│   ├── ingest.py             # Document ingestion with local embeddings
│   ├── run_evaluation.py     # RAGAS evaluation runner
│   └── compare_metrics.py    # Baseline comparison for CI/CD
├── src/
│   └── rag_chain.py          # Core RAG logic with LCEL
├── tests/                     # Unit tests
├── app.py                     # Streamlit UI
├── docker-compose.yml         # Full stack orchestration
├── Dockerfile                 # App containerization
├── prometheus.yml             # Metrics configuration
├── requirements.txt           # Python dependencies
├── Makefile                   # Professional command shortcuts
├── .env                       # API keys (NEVER commit!)
├── .gitignore                 # Must include .env
└── README.md                  # Project documentation
```

## Technology Stack

**Core Application:**
- Python 3.11+
- LangChain (LCEL for chain composition)
- Streamlit (web interface)
- FAISS (local vector store)

**LLM Providers:**
- Generation: OpenAI (`gpt-3.5-turbo`) or Anthropic (`claude-3-sonnet`)
- Evaluation: OpenAI (`gpt-4-turbo`) or Anthropic (`claude-3-opus`)
- Embeddings: `sentence-transformers` (local, free)

**MLOps Infrastructure:**
- Evaluation: RAGAS
- CI/CD: GitHub Actions
- Observability: LangSmith, Prometheus, Grafana
- Containerization: Docker, Docker Compose

## Key Implementation Patterns

### 1. LCEL Context Preservation Pattern

**Critical for evaluation**: Use `RunnableParallel` to maintain access to retrieved contexts after LLM generation.

```python
from langchain.schema.runnable import RunnableParallel

# Basic chain for UI
rag_chain = (
    setup_and_retrieval
    | prompt
    | model
    | StrOutputParser()
)

# Enhanced chain for evaluation - preserves contexts
full_chain = RunnableParallel(
    answer=rag_chain,
    contexts=lambda x: x["context"]
)
```

**Why this matters:**
- Enables comprehensive RAGAS evaluation (faithfulness, context relevance)
- Separates concerns: UI needs just answers, evaluation needs full context
- Demonstrates advanced LangChain expertise

### 2. Dual Interface Pattern

**Separation of Concerns**: Provide different interfaces for different use cases.

```python
def ask_question(question: str) -> str:
    """Simple interface for UI - returns just the answer"""
    return rag_chain.invoke(question)

def ask_question_with_context(question: str) -> dict:
    """For evaluation - returns both answer and contexts"""
    result = full_chain.invoke(question)
    return {
        "answer": result["answer"],
        "contexts": [doc.page_content for doc in result["contexts"]]
    }
```

### 3. Security-Aware Deserialization

**Production mindset**: Always document security trade-offs.

```python
# IMPORTANT: Document why this is safe in your context
vectorstore = FAISS.load_local(
    "faiss_index", 
    embedding_model, 
    allow_dangerous_deserialization=True  # Safe because we control the source
)
```

**Best practice**: Add a comment explaining:
- Why this is safe in the current context
- What would need to change in production
- Security considerations if the source is untrusted

### 4. Defensive Coding in CI/CD

**Robust metric comparison**: Handle missing metrics gracefully.

```python
def compare_metrics(baseline_path, current_path, threshold=0.05):
    # Use .get() with defaults to prevent pipeline crashes
    baseline_score = baseline.get(metric, 0)
    current_score = current.get(metric, 0)
    
    if current_score < baseline_score - threshold:
        print(f"❌ REGRESSION in {metric}")
        failed = True
```

**Why this matters:**
- Prevents CI/CD pipeline failures from missing keys
- Shows production-grade error handling
- Demonstrates anticipation of edge cases

## Security Best Practices (CRITICAL)

### API Key Management

**Local Development:**
```bash
# .env file (NEVER commit this!)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=lsv2_pt_...
```

**Application Code:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

**GitHub Actions:**
```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

**Critical Requirements:**
1. `.env` must be in `.gitignore`
2. Never hardcode API keys in code
3. Use GitHub Secrets for CI/CD
4. Document security practices in README

## Evaluation Strategy

### Golden Dataset Creation

**Manual, high-quality dataset** (15-20 examples):

```json
[
  {
    "question": "What is the main contribution of the Transformer?",
    "ground_truth_answer": "The main contribution is...",
    "ground_truth_context": "Direct quote from source document..."
  }
]
```

**Requirements:**
- Questions must be answerable from the source document
- Answers must be based only on document content
- Include exact context snippets that support the answer
- Cover diverse question types and difficulty levels

### RAGAS Metrics

**Key metrics to track:**

1. **Faithfulness** (0.0-1.0): Is the answer grounded in retrieved context?
   - Prevents hallucinations
   - Target: ≥ 0.80

2. **Context Relevance** (0.0-1.0): Are retrieved documents relevant to the query?
   - Measures retrieval quality
   - Target: ≥ 0.75

3. **Answer Relevance** (0.0-1.0): Does the answer address the question?
   - Measures generation quality
   - Target: ≥ 0.80

4. **Context Precision**: Are relevant contexts ranked highly?
   - Important for ranking quality

5. **Context Recall**: Is all necessary information retrieved?
   - Measures completeness

### Evaluation Implementation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevance, 
    context_recall, 
    context_precision
)
from langchain_openai import ChatOpenAI

# Use top-tier model for reliable evaluation judgments
judge_llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevance, context_recall, context_precision],
    llm=judge_llm
)
```

## CI/CD Quality Gates

### Regression Prevention

**Quality gate configuration:**
```yaml
gates:
  faithfulness:
    threshold: 0.80
    comparison: ">="
    blocking: true
    
  context_relevance:
    threshold: 0.75
    comparison: ">="
    blocking: true
    
  regression_threshold:
    max_degradation: 0.05  # Allow 5% degradation
    baseline: "main"
    blocking: true
```

**Workflow:**
1. PR opened → Run evaluation on golden dataset
2. Compare metrics to baseline
3. If any metric degrades > 5% → Block merge
4. On merge to main → Update baseline

## Cost Management Strategy

### Cost Breakdown

**Per-Component Costs:**
- Embeddings: **$0** (local `instructor-large`)
- Per user query: ~$0.001-0.002 (GPT-3.5-turbo)
- Evaluation run (20 questions): ~$0.10-0.20 (GPT-4 judge)

**Optimization Strategy:**
- Use local embeddings for all vector operations (biggest savings)
- Use mid-tier model for generation (GPT-3.5-turbo)
- Invest in top-tier model for evaluation (GPT-4)

**Tiered Approach for Production:**
- PR checks: Use cheaper model (GPT-3.5-turbo) for evaluation
- Weekly baselines: Use expensive model (GPT-4) for thorough assessment
- Always: Use local embeddings

### Why This Matters

**Interview talking point**: "I reduced operational costs by 90% by strategically using local embeddings while maintaining quality with proprietary models for generation. This shows I understand the economic constraints of AI systems."

## Observability Stack

### LangSmith Integration

**Tracing setup:**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "rag-ops-framework"
```

**What to trace:**
- Complete execution flow of RAG chains
- Individual component spans (retrieval, generation, post-processing)
- Input/output logging for debugging
- Latency and token usage

### Prometheus Metrics

**Key metrics to track:**
```python
from prometheus_client import Counter, Histogram

# Request metrics
questions_counter = Counter('rag_questions_total', 'Total questions asked')
response_time = Histogram('rag_response_seconds', 'Response time in seconds')

# Quality metrics (from evaluation)
faithfulness_gauge = Gauge('rag_faithfulness_score', 'Faithfulness score')
context_relevance_gauge = Gauge('rag_context_relevance_score', 'Context relevance')
```

### Grafana Dashboard Panels

**Recommended panels:**
1. **Overview**: Total requests, avg response time, error rate, total cost
2. **Quality**: Faithfulness/relevance over time, pass/fail ratio, regression alerts
3. **Performance**: Latency percentiles (p50, p95, p99), throughput
4. **Cost**: Cost per request trend, token usage, daily/monthly projections

## Development Workflow

### Makefile Commands

```makefile
make setup      # Create virtual environment
make install    # Install dependencies
make ingest     # Run document ingestion
make run        # Start Streamlit app
make eval       # Run evaluation
make up         # Start Docker stack
make down       # Stop Docker containers
make clean      # Remove cache files
```

**Why this matters**: Professional workflow automation signals senior-level engineering practices.

## Common Pitfalls & Solutions

### 1. API Key Leakage
**Problem**: Accidentally committing `.env` to Git
**Solution**: 
- Add `.env` to `.gitignore` immediately
- Use `git-secrets` or pre-commit hooks
- Document security practices clearly

### 2. Inconsistent Embedding Models
**Problem**: Using different embedding models for ingestion vs. retrieval
**Solution**: 
- Store model name in metadata
- Validate consistency on load
- Document which model was used

### 3. Evaluation Costs
**Problem**: Expensive evaluation runs on every commit
**Solution**:
- Use cheaper models for PR checks
- Reserve expensive models for baseline updates
- Cache evaluation results where possible

### 4. Docker Volumes
**Problem**: FAISS index not accessible in container
**Solution**:
```yaml
services:
  app:
    volumes:
      - ./faiss_index:/app/faiss_index
```

### 5. Context Loss in Chains
**Problem**: Can't evaluate because contexts aren't preserved
**Solution**: Use `RunnableParallel` pattern (see above)

## Code Review Guidelines

When reviewing code for this project, check for:

### ✅ Must Have
- [ ] API keys managed via environment variables
- [ ] `.env` in `.gitignore`
- [ ] LCEL used for chain composition
- [ ] Context preservation for evaluation
- [ ] Defensive coding (`.get()` with defaults)
- [ ] Security comments on dangerous operations
- [ ] Clear separation of concerns

### ✅ Nice to Have
- [ ] Type hints on function signatures
- [ ] Docstrings on public functions
- [ ] Error handling with informative messages
- [ ] Logging for debugging
- [ ] Configuration via YAML files

### ❌ Red Flags
- [ ] Hardcoded API keys
- [ ] Missing `.gitignore` for `.env`
- [ ] No baseline comparison in CI/CD
- [ ] Mixing UI and evaluation logic
- [ ] Using paid APIs for embeddings
- [ ] No security documentation

## Documentation Standards

### README.md Structure

**Required sections:**
1. **Title + Tagline**: Clear, compelling description
2. **Architecture Diagram**: Show hybrid architecture and data flow
3. **Key Features**: Bulleted list of accomplishments
4. **Cost Management**: Explain hybrid approach and savings
5. **Security**: Document API key practices
6. **Quick Start**: Simple `docker-compose up` instructions
7. **Observability**: Links to LangSmith traces, Grafana screenshots
8. **Tech Stack**: All dependencies and versions
9. **Evaluation**: Explain RAGAS metrics and quality gates

### Code Comments

**When to comment:**
- Security trade-offs (e.g., `allow_dangerous_deserialization`)
- Non-obvious architectural decisions
- Complex LCEL chains
- Performance optimizations
- Cost implications

**Comment style:**
```python
# COST OPTIMIZATION: Using local embeddings saves ~$0.0004 per query
# This adds ~100ms latency but eliminates API costs entirely
embedding_model = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)
```

## Testing Strategy

### Unit Tests (Optional but Recommended)

**Test coverage:**
- RAG chain initialization
- Document chunking logic
- Metric comparison logic
- API error handling

**Example:**
```python
def test_rag_chain_initialization():
    """Ensure RAG chain loads without errors"""
    from src.rag_chain import ask_question
    assert ask_question is not None

def test_compare_metrics_passes():
    """Verify comparison logic with passing metrics"""
    # Mock baseline and current with good scores
    assert compare_metrics(baseline, current) == 0
```

### Integration Tests

**Test scenarios:**
- End-to-end query through RAG chain
- Evaluation pipeline on small dataset
- Docker container startup
- Prometheus metrics export

## Presentation Tips

### For Interviews

**Technical depth questions:**
- "Walk me through your RAG architecture"
  - Lead with hybrid approach, explain cost/quality trade-offs
  
- "How do you ensure quality?"
  - Discuss RAGAS metrics, CI/CD quality gates, baseline comparisons

- "What would you do differently at scale?"
  - Mention caching, distributed vector stores, model distillation

**System design questions:**
- "How would you scale this to 10,000 users?"
  - Load balancing, Redis caching, distributed FAISS, rate limiting

- "How do you handle failures?"
  - Circuit breakers, fallback responses, retry logic, alerting

**Behavioral questions:**
- "Tell me about a time you balanced cost and quality"
  - Use hybrid architecture as the example

### Portfolio Presentation

**Visual assets needed:**
- Architecture diagram (hybrid approach highlighted)
- Grafana dashboard screenshot
- GitHub Actions passing/failing examples
- LangSmith trace screenshots
- Cost comparison chart

**Narrative structure:**
1. Problem: RAG systems are expensive and hard to maintain
2. Solution: Hybrid architecture + full MLOps pipeline
3. Impact: 90% cost reduction + automated quality assurance
4. Learning: Production-grade practices from day one

## Advanced Enhancements (Optional)

### Phase 2 Features

1. **Multi-document support**: Extend to multiple PDFs with metadata filtering
2. **Streaming responses**: Use LangChain streaming for better UX
3. **Query rewriting**: Add query expansion/decomposition
4. **Hybrid search**: Combine vector + keyword search
5. **Caching layer**: Redis for frequently asked questions
6. **A/B testing**: Compare different prompts/models
7. **Human feedback**: Add thumbs up/down with LangSmith
8. **Custom metrics**: Business-specific evaluation criteria

### Production Hardening

1. **Rate limiting**: Prevent API abuse
2. **Authentication**: Add user auth to Streamlit
3. **Error tracking**: Sentry integration
4. **Backup strategy**: Regular FAISS index backups
5. **Monitoring alerts**: PagerDuty/Slack integration
6. **Load testing**: Locust or k6 for stress testing
7. **Database**: Move from JSON to PostgreSQL for golden dataset
8. **Model versioning**: Track which model version generated each response

## Success Criteria

### Project Completion Checklist

**Phase 1: Core Application (Weeks 1-2)**
- [ ] Document ingestion with local embeddings
- [ ] FAISS vector store created and saved
- [ ] RAG chain implemented with LCEL
- [ ] Streamlit UI functional
- [ ] Can ask questions and get answers

**Phase 2: Evaluation (Weeks 3-4)**
- [ ] Golden dataset created (15-20 examples)
- [ ] RAGAS evaluation script working
- [ ] All four key metrics calculated
- [ ] Baseline report generated and committed

**Phase 3: MLOps (Weeks 5-6)**
- [ ] GitHub Actions workflows configured
- [ ] Quality gates blocking bad PRs
- [ ] Baseline auto-updates on main branch
- [ ] GitHub Secrets configured properly

**Phase 4: Observability (Weeks 7-8)**
- [ ] LangSmith tracing integrated
- [ ] Prometheus metrics exported
- [ ] Grafana dashboard created
- [ ] Docker Compose stack working

**Phase 5: Documentation (Week 9-10)**
- [ ] README with all required sections
- [ ] Architecture diagram created
- [ ] Security practices documented
- [ ] Screenshots and visuals added
- [ ] Cost analysis included

### Quality Metrics

**Target scores:**
- Faithfulness: ≥ 0.80
- Context Relevance: ≥ 0.75
- Answer Relevance: ≥ 0.80
- Context Precision: ≥ 0.70
- Context Recall: ≥ 0.75

**Operational metrics:**
- CI/CD pipeline: < 10 minutes
- Response time: < 3 seconds (p95)
- Evaluation cost: < $0.20 per run
- False positive rate: < 5%

## Troubleshooting Guide

### Common Issues

**Issue: "FAISS index not found"**
- Solution: Run `make ingest` or `python scripts/ingest.py` first
- Verify `faiss_index` directory exists

**Issue: "OpenAI API key not found"**
- Solution: Check `.env` file exists and has correct key
- Verify `python-dotenv` is installed
- Check environment variable loading in code

**Issue: "GitHub Actions failing on API calls"**
- Solution: Verify GitHub Secrets are configured
- Check secret name matches workflow file
- Review API key permissions and quotas

**Issue: "Evaluation taking too long"**
- Solution: Use smaller dataset for testing
- Switch to cheaper model for PR checks
- Cache evaluation results

**Issue: "Docker container can't access FAISS index"**
- Solution: Check volume mount in docker-compose.yml
- Verify file permissions
- Rebuild Docker image

**Issue: "Metrics not appearing in Prometheus"**
- Solution: Check Prometheus target configuration
- Verify port 8000 is exposed
- Review scrape interval settings

## Resources & References

### Official Documentation
- LangChain: https://python.langchain.com/docs/
- RAGAS: https://docs.ragas.io/
- LangSmith: https://docs.smith.langchain.com/
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/

### Key Papers & Articles
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)
- LangChain Blog: Best Practices for Production RAG

### Community Resources
- LangChain Discord
- RAGAS GitHub Issues
- r/LangChain subreddit
- MLOps Community Slack

---

## When to Ask for Help

**Always clarify:**
- Model selection trade-offs (cost vs. quality)
- Security implications of implementation choices
- Scaling considerations for production
- Best practices for specific integrations

**Request code review for:**
- LCEL chain implementations
- Security-sensitive code (API keys, deserialization)
- CI/CD pipeline configurations
- Docker multi-container setups

**Seek guidance on:**
- Architecture decisions
- Evaluation metric interpretations
- Cost optimization strategies
- Production deployment concerns

---

## Final Notes

This project is designed to demonstrate **production-grade AI engineering skills**. Every component—from the hybrid architecture to the CI/CD pipelines—mirrors real-world systems at top companies. 

**Key differentiators:**
1. **Strategic thinking**: Hybrid architecture shows cost-quality balance
2. **Production mindset**: Security, observability, automation from day one
3. **Quality focus**: Automated quality gates prevent regressions
4. **Professional practices**: Makefiles, Docker, proper documentation

Follow these guidelines, and you'll build a portfolio project that stands out in a sea of basic LLM demos.

---

**Document Version**: 2.0  
**Last Updated**: October 2025  
**Project Difficulty**: Intermediate to Advanced  
**Time Commitment**: 8-10 weeks part-time