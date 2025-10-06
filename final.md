# The Production-Grade RAG Ops Framework
### The Pragmatic, Industry-Standard Approach

**Project Goal:** To build, evaluate, and monitor a RAG (Retrieval-Augmented Generation) chatbot within a professional MLOps framework using a hybrid architecture that mirrors real-world production systems. This project demonstrates end-to-end skills: application development, automated quality assurance, CI/CD, production observability, and strategic cost management.

---

## The Elevator Pitch

"I built an MLOps framework for a RAG chatbot that mirrors how top tech companies operate their AI systems. It leverages a best-in-class proprietary model (like GPT-4 or Claude 3) for high-quality generation and evaluation, ensuring the best possible user experience. Crucially, it strategically uses a local, open-source model for the embedding process, drastically reducing API costs and latency. The entire system is automated with CI/CD quality gates and features a full observability stack, proving I can build, test, and operate AI solutions that balance performance, cost, and quality."

---

## Why This Hybrid Approach is Impressive

- **Reflects Reality:** This is how many real-world production systems are built. It shows you're not just a hobbyist; you're thinking like a professional engineer.
- **Demonstrates Cost Management:** You're not just blindly calling APIs. By using local embeddings, you show you can identify the most expensive part of a RAG pipeline and replace it with a free, efficient alternative. This is a huge talking point.
- **Focuses on Quality:** You are using the best possible "judge" model (e.g., GPT-4) for your evaluations. This makes your quality metrics more reliable and the entire premise of the project stronger.
- **Highlights Security Best Practices:** Using proprietary APIs forces you to handle API keys and secrets correctly, which is a critical skill.
- **Balances Performance and Cost:** Strategic architectural decisions that demonstrate senior-level engineering thinking.

---

## Part 0: Foundation & Secure Setup (The Blueprint)

Before writing a single line of application code, set up your project for success. A professional structure is a strong signal to recruiters.

### 1. Technology Stack

**Application:**
- Python, LangChain, Streamlit
- LLMs: `openai` or `anthropic` (proprietary, for generation and evaluation)
- Embeddings: `sentence-transformers` (Local & Free - this is your strategic cost-saving move)

**Infrastructure:**
- Vector Store: FAISS (for simplicity and local execution)
- Evaluation: RAGAS
- CI/CD: GitHub Actions
- Observability: LangSmith, Prometheus, Grafana
- Containerization: Docker, Docker Compose

### 2. Project Directory Structure

Create this structure at the beginning. It keeps your project clean and understandable.

```
/rag-ops-framework
├── .github/workflows/         # GitHub Actions workflows
│   ├── pr_check.yml
│   └── update_baseline.yml
├── data/                      # For your source doc and golden dataset
│   ├── source_document.pdf
│   └── golden_dataset.json
├── scripts/                   # Helper and evaluation scripts
│   ├── ingest.py
│   ├── run_evaluation.py
│   └── compare_metrics.py
├── src/                       # Your core application source code
│   └── rag_chain.py
├── tests/                     # (Optional but recommended) Unit tests
├── app.py                     # The Streamlit application entry point
├── docker-compose.yml         # For running the entire stack
├── Dockerfile                 # To containerize the Streamlit app
├── requirements.txt           # Python dependencies
├── prometheus.yml             # Prometheus configuration
├── .env                       # API keys (NEVER commit this!)
├── .gitignore                 # Must include .env
├── Makefile                   # Professional command shortcuts
└── README.md                  # Your project's most important file
```

### 3. API Key Management & Security Best Practices

**CRITICAL: This is a non-negotiable professional practice.**

**Local Development:**
- Create a `.env` file in your project root and add your API keys:
  ```
  OPENAI_API_KEY="sk-..."
  # or
  ANTHROPIC_API_KEY="sk-ant-..."
  ```
- **This file must be listed in your `.gitignore`**. Committing API keys is a cardinal sin.
- Use `python-dotenv` to load these keys in your application.

**GitHub Secrets for CI/CD:**
- In your GitHub repository settings, go to `Secrets and variables` -> `Actions`.
- Create a new repository secret named `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) and paste your key there.
- Your CI/CD pipeline will use this to run evaluations securely.

### 4. Initial Setup

- Create a virtual environment: `python -m venv venv` and activate it.
- Create `requirements.txt` and add initial libraries:
  ```
  langchain
  langchain-openai
  # or langchain-anthropic
  sentence-transformers
  streamlit
  faiss-cpu
  ragas
  python-dotenv
  pypdf
  prometheus-client
  ```
- Install them: `pip install -r requirements.txt`.
- Initialize a Git repository and create your first commit.

### 6. Create a Makefile for Professional Workflow

Add a `Makefile` to provide simple, repeatable commands. This signals professional-level workflow management.

```makefile
# Makefile

.PHONY: setup ingest run test eval up down

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

install:
	pip install -r requirements.txt

ingest:
	@echo "Running ingestion script..."
	python scripts/ingest.py
	@echo "Ingestion complete."

run:
	@echo "Starting Streamlit application..."
	streamlit run app.py

eval:
	@echo "Running evaluation script..."
	python scripts/run_evaluation.py

up:
	@echo "Starting Docker containers..."
	docker-compose up -d

down:
	@echo "Stopping Docker containers..."
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

**Usage Examples:**
- `make setup` - Create virtual environment
- `make install` - Install dependencies
- `make ingest` - Run ingestion pipeline
- `make run` - Start Streamlit app
- `make eval` - Run evaluation
- `make up` - Start full Docker stack
- `make down` - Stop Docker containers

### 7. Choose Your Source Document

- Select a single, non-trivial PDF document (~10-50 pages).
- **Good choices:** A famous research paper (like the original "Attention Is All You Need" or the Llama 2 paper), an open-source project's documentation, or a detailed product manual.
- Place it in `data/source_document.pdf`.

---

## Part 1: The Hybrid RAG Application (The Core Product)

Build a working, interactive RAG chatbot using the hybrid architecture.

### 1. The Ingestion Pipeline (`scripts/ingest.py`)

This script will be run once to prepare your vector database. **This is your strategic cost-saving move** - using local embeddings means this process is 100% free.

**Logic:**
1. Load the PDF from `data/source_document.pdf` using LangChain's `PyPDFLoader`.
2. Split the document into manageable chunks (e.g., 1000 characters with 200 overlap) using `RecursiveCharacterTextSplitter`.
3. **Use local embeddings** with `HuggingFaceInstructEmbeddings` or `sentence-transformers`.
4. Store the chunks and their embeddings in a FAISS vector store.
5. Save the FAISS index locally (e.g., to a file named `faiss_index`).

**Example Code:**
```python
# scripts/ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

# Load document
loader = PyPDFLoader("data/source_document.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create local embeddings (FREE!)
embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Create and save vector store
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("faiss_index")
```

### 2. The RAG Logic (`src/rag_chain.py`)

Create a function or class that encapsulates the RAG logic. This separates it from the UI. **Here we use a proprietary model for high-quality generation.**

**Logic:**
1. Load the pre-built FAISS index from disk using the same local embedding model.
2. Create a retriever from the vector store.
3. Define a prompt template that takes a `question` and `context` as input.
4. **Define your LLM using a proprietary model** (e.g., `ChatOpenAI` with GPT-3.5-turbo or GPT-4).
5. Combine these into a RAG chain (using LangChain's LCEL is modern and recommended). The chain should take a question, retrieve context, format the prompt, and return the answer.

**Example Code:**
```python
# src/rag_chain.py
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Load local embeddings (must match the model used in ingestion)
embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Load FAISS index with security note
# allow_dangerous_deserialization=True is used because we trust the source of the index file.
# In a production system where the index might come from an untrusted source,
# this would require careful security review.
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Use a powerful, fast proprietary model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Build RAG chain with context tracking for evaluation
# This uses RunnableParallel to preserve contexts alongside the answer
from langchain.schema.runnable import RunnableParallel

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# Basic chain for answer generation
rag_chain = (
    setup_and_retrieval
    | prompt
    | model
    | StrOutputParser()
)

# Full chain that returns both answer and contexts for evaluation
full_chain = RunnableParallel(
    answer=rag_chain,
    contexts=lambda x: x["context"]  # Pass through retrieved contexts
)

def ask_question(question: str) -> str:
    """Simple interface for the UI - returns just the answer"""
    return rag_chain.invoke(question)

def ask_question_with_context(question: str) -> dict:
    """For evaluation - returns both answer and retrieved contexts"""
    result = full_chain.invoke(question)
    return {
        "answer": result["answer"],
        "contexts": [doc.page_content for doc in result["contexts"]]
    }
```

### 3. The User Interface (`app.py`)

Use Streamlit to create a simple and clean UI.

**UI Elements:**
- A title: "Chat with [Your Document's Name]".
- A text input box for the user's question.
- A button to submit the question.
- A display area to show the LLM's answer.

**Logic:**
- On startup, import and initialize your RAG chain from `src/rag_chain.py`.
- When the user asks a question, call the RAG chain and display the generated answer.

**Example Code:**
```python
# app.py
import streamlit as st
from src.rag_chain import ask_question

st.title("Chat with [Your Document's Name]")

question = st.text_input("Ask a question about the document:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Thinking..."):
            answer = ask_question(question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
```

**Milestone:** You now have a working, interactive chatbot that you can run locally with `streamlit run app.py`.

---

## Part 2: The High-Fidelity Evaluation Layer (The Quality Check)

Quantitatively measure how good your chatbot is using the best possible evaluation models.

### 1. Create the Golden Dataset (`data/golden_dataset.json`)

- **Manually** read through your `source_document.pdf` and create 15-20 question-answer pairs.
- For each entry, include:
  - `question`: A clear, specific question.
  - `ground_truth_answer`: A perfect, human-written answer based *only* on the document.
  - `ground_truth_context`: The exact text snippet(s) from the document that justify the answer.
- This manual effort is a key part of the project.

**Example Format:**
```json
[
  {
    "question": "What is the main contribution of the Transformer architecture?",
    "ground_truth_answer": "The main contribution is...",
    "ground_truth_context": "Direct quote from the paper..."
  }
]
```

### 2. The Evaluation Script (`scripts/run_evaluation.py`)

**Use a top-tier model as the "judge" to get the most accurate and reliable quality scores.** This is where the hybrid approach shines - you invest in quality where it matters most.

**Logic:**
1. Load the `golden_dataset.json`.
2. Initialize your RAG chain from `src/rag_chain.py`.
3. Iterate through each item in the golden dataset. For each `question`, run it through your RAG chain to get the generated `answer` and the retrieved `contexts`.
4. Collect the results into a list suitable for RAGAS (a list of dictionaries containing `question`, `answer`, `contexts`, and `ground_truth_answer`).
5. **Use a top-tier model (like GPT-4) for reliable evaluation judgments.**
6. Use `ragas.evaluate()` on this dataset with the key metrics: `faithfulness`, `answer_relevance`, `context_relevance`, `context_recall`.
7. Print a clean, formatted report of the average scores to the console.
8. Save the detailed results to a file named `evaluation_report.json`.

**Example Code:**
```python
# scripts/run_evaluation.py
import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_recall, context_precision
from langchain_openai import ChatOpenAI
from src.rag_chain import ask_question_with_context  # Use the enhanced function
from datasets import Dataset

# Load golden dataset
with open("data/golden_dataset.json", "r") as f:
    golden_data = json.load(f)

# Generate answers using your RAG chain with context tracking
eval_data = []
for item in golden_data:
    question = item["question"]

    # Get both answer and contexts using the enhanced chain
    result = ask_question_with_context(question)

    eval_data.append({
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "ground_truth": item["ground_truth_answer"]
    })

# Convert to RAGAS dataset format
eval_dataset = Dataset.from_list(eval_data)

# Use a top-tier model for reliable evaluation judgments
judge_llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# Run the evaluation
result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevance, context_recall, context_precision],
    llm=judge_llm  # Use the best possible judge
)

# Print and save results
print("\n=== Evaluation Results ===")
print(result)

with open("evaluation_report.json", "w") as f:
    json.dump(result, f, indent=2)
```

**Milestone:** You can now run `python scripts/run_evaluation.py` to get a numeric score for your chatbot's quality.

---

### 3. Advanced Implementation Note: Context Tracking with LCEL

**Why This Matters:** The sophisticated use of `RunnableParallel` in the RAG chain demonstrates deep understanding of LangChain Expression Language (LCEL). This pattern:

1. **Preserves Data Flow:** Maintains access to retrieved contexts even after they're consumed by the LLM
2. **Enables Comprehensive Evaluation:** Provides everything RAGAS needs (question, answer, contexts, ground truth)
3. **Separation of Concerns:** Offers both a simple UI interface and a detailed evaluation interface
4. **Production Best Practice:** Shows how to build observable, debuggable LLM chains

This level of implementation detail signals senior-level engineering expertise and understanding of production LLM systems.

---

## Part 3: API-Aware Automation & MLOps (The Safety Net)

Integrate your evaluation into a CI/CD pipeline to automatically prevent regressions.

### 1. Establish a Baseline

- Run your evaluation script once and save the output as `baseline_report.json` in the root of your project. Check this into Git.

### 2. The Comparison Script (`scripts/compare_metrics.py`)

**Logic:**
1. Takes two file paths as command-line arguments: `baseline.json` and `current.json`.
2. Loads both JSON reports.
3. Compares the key metrics (e.g., `faithfulness`).
4. If a metric in the `current.json` is lower than the baseline by a certain threshold (e.g., 5%), print an error message and `exit(1)`.
5. Otherwise, print a success message and `exit(0)`.

**Example Code:**
```python
# scripts/compare_metrics.py
import json
import sys

def compare_metrics(baseline_path, current_path, threshold=0.05):
    with open(baseline_path, "r") as f:
        baseline = json.load(f)
    with open(current_path, "r") as f:
        current = json.load(f)

    failed = False
    for metric in ["faithfulness", "answer_relevance", "context_recall", "context_precision"]:
        # Use .get() with default value to handle missing metrics gracefully
        baseline_score = baseline.get(metric, 0)
        current_score = current.get(metric, 0)

        if current_score < baseline_score - threshold:
            print(f"❌ REGRESSION in {metric}: {current_score:.3f} < {baseline_score:.3f}")
            failed = True
        else:
            print(f"✅ {metric}: {current_score:.3f} (baseline: {baseline_score:.3f})")

    if failed:
        sys.exit(1)
    else:
        print("\n✅ All metrics passed!")
        sys.exit(0)

if __name__ == "__main__":
    compare_metrics(sys.argv[1], sys.argv[2])
```

**Why the `.get()` method matters:** This defensive coding prevents pipeline crashes if a metric is missing from either report, ensuring robust CI/CD operations.

### 3. GitHub Actions Workflows

**PR Check (`.github/workflows/pr_check.yml`):**

This workflow now needs secure access to your API key.

```yaml
# .github/workflows/pr_check.yml
name: PR Quality Check
on: pull_request

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run Evaluation
        run: python scripts/run_evaluation.py --output current_report.json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}  # Securely inject the key

      - name: Check for Regressions
        run: python scripts/compare_metrics.py baseline_report.json current_report.json
```

**Baseline Update (`.github/workflows/update_baseline.yml`):**

```yaml
# .github/workflows/update_baseline.yml
name: Update Baseline
on:
  push:
    branches: [ main ]

jobs:
  update-baseline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run Evaluation
        run: python scripts/run_evaluation.py --output baseline_report.json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Commit Updated Baseline
        uses: stefanzweifel/git-auto-commit-action@v4
        with:
          commit_message: "Update evaluation baseline [skip ci]"
          file_pattern: baseline_report.json
```

**Milestone:** You have a fully automated quality gate. No code that makes the chatbot worse can be merged.

---

## Part 4: Production Observability (The Mission Control)

Add tools to monitor and debug your application as if it were live.

### 1. Tracing with LangSmith

- Integrate LangSmith into your RAG chain. This is often as simple as adding a callback handler.
- In your README, link to one or two interesting public traces from your evaluation runs.

**Example:**
```python
# Add to your rag_chain.py
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "rag-ops-framework"
```

### 2. Metrics with Prometheus & Grafana

**Instrument your App:** In `app.py`, add the `prometheus-client` library. Create a few metrics:
- A `Counter` for the total number of questions asked.
- A `Histogram` to track the latency of the RAG chain's response time.

**Example:**
```python
# app.py - Add metrics
from prometheus_client import Counter, Histogram, start_http_server
import time

# Start Prometheus metrics server
start_http_server(8000)

# Define metrics
questions_counter = Counter('rag_questions_total', 'Total questions asked')
response_time = Histogram('rag_response_seconds', 'Response time in seconds')

# In your question handling logic:
if st.button("Get Answer"):
    if question:
        questions_counter.inc()
        start_time = time.time()

        with st.spinner("Thinking..."):
            answer = ask_question(question)

        response_time.observe(time.time() - start_time)
        st.success("Answer:")
        st.write(answer)
```

**Configure Prometheus:** Create `prometheus.yml`:
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-app'
    static_configs:
      - targets: ['app:8000']
```

**Containerize Everything (`Dockerfile` and `docker-compose.yml`):**

**Dockerfile:**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./faiss_index:/app/faiss_index

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Milestone:** You can run `docker-compose up` and get your entire environment—chatbot, metrics collector, and dashboarding tool—running with one command. Take a screenshot of a simple Grafana dashboard showing your custom metrics.

---

## Part 5: Cost Management & Strategic Planning

### Understanding the Costs

A single evaluation run of 20 questions against `gpt-4-turbo` will cost a few cents. This is a worthwhile investment in the quality of your project.

### Strategic Cost Optimization

**For a production system, consider this tiered approach:**
- **PR Checks:** Use a cheaper model (like `gpt-3.5-turbo`) for the automated PR check workflow.
- **Nightly/Weekly Baselines:** Use the more expensive model (`gpt-4-turbo`) for less frequent but more thorough baseline updates.
- **Embeddings:** Continue using local embeddings - this is where you save the most money.

**Example Cost Breakdown:**
- Embeddings: $0 (local model)
- Generation per query: ~$0.001-0.01 (depending on model)
- Evaluation per run: ~$0.05-0.20 (20 questions with GPT-4)

### Add This to Your README

Create a "Cost Management" section in your README explaining:
1. Your hybrid architecture choice and rationale
2. The strategic use of local embeddings vs. proprietary LLMs
3. Your tiered evaluation strategy (if implemented)
4. Estimated costs for different usage patterns

**Example:**
```markdown
## Cost Management Strategy

This project uses a **hybrid architecture** to balance quality and cost:

- **Embeddings:** Local `instructor-large` model (100% free)
- **Generation:** GPT-3.5-turbo for fast, cost-effective responses
- **Evaluation:** GPT-4-turbo for high-fidelity quality assessment

**Estimated Costs:**
- Embeddings: $0 (local)
- Per user query: ~$0.002
- Evaluation run (20 questions): ~$0.10

**Production Optimization:**
In a production environment, we could further optimize by using GPT-3.5-turbo for
PR checks and reserving GPT-4 for weekly baseline evaluations.
```

---

## Part 6: Presentation & Storytelling (The Portfolio Polish)

Your `README.md` is the most important part of the portfolio.

### Structure Your README

**Title and Badges:**
- Project title with a compelling tagline
- CI status badge from GitHub Actions
- License badge (if applicable)

**Architecture Diagram:**
- Create a clean diagram showing how all components connect
- Highlight the hybrid architecture (local embeddings + proprietary LLMs)
- Show the data flow from ingestion to response
- This is extremely impressive and demonstrates system-level thinking

**Live Demo/Visuals:**
- A GIF of your Streamlit app in action
- The screenshot of your Grafana dashboard showing metrics
- A screenshot of a passed/failed GitHub Actions run
- A link to a public LangSmith trace

**Key Features:**
A bulleted list of what you accomplished:
- "Hybrid RAG architecture balancing cost and quality"
- "Local embeddings for zero-cost vector storage"
- "Proprietary LLM for high-quality generation and evaluation"
- "Automated quality gates with CI/CD"
- "Production observability with LangSmith, Prometheus, and Grafana"
- "Secure API key management with GitHub Secrets"
- "Containerized deployment with Docker Compose"
- "RAGAS-based evaluation with GPT-4 judge model"

**Security Best Practices:**
- Explain your `.env` and `.gitignore` setup
- Document the GitHub Secrets configuration
- Highlight why this matters for production systems

**Observability Links:**
- Link to 1-2 interesting public LangSmith traces
- Include a screenshot of your Grafana dashboard
- Show example evaluation results

**How to Run:**
Simple, clear instructions, highlighting the `docker-compose up` command:

```markdown
## Quick Start

1. Clone the repository
2. Create a `.env` file with your API key:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. Run the ingestion script:
   ```
   python scripts/ingest.py
   ```
4. Start the entire stack:
   ```
   docker-compose up
   ```
5. Access the app at `http://localhost:8501`
6. View metrics in Grafana at `http://localhost:3000`
```

**The Final Narrative:**

Your project story is now about **pragmatism, intelligence, and professionalism.** You are not a purist; you are an engineer who chooses the right tool for the job to balance competing priorities:
- Quality (proprietary models where they matter)
- Cost (local models for expensive operations)
- Security (proper secrets management)
- Observability (production-grade monitoring)
- Automation (CI/CD quality gates)

This is what it means to be a senior-level contributor.

---

## Conclusion

By completing these six parts, you will have created a project that demonstrates a rare and highly valuable combination of skills:

1. **Technical Depth:** RAG implementation, LLM orchestration, vector databases
2. **Engineering Rigor:** CI/CD, automated testing, quality gates
3. **Production Readiness:** Observability, monitoring, containerization
4. **Strategic Thinking:** Cost optimization, security best practices
5. **Professional Communication:** Clear documentation, architecture diagrams

This hybrid approach perfectly mirrors how the best companies in the world are building with AI right now. It positions you as a top candidate for any AI engineering role.

---

## Appendix: Senior Engineer Refinements

### Key Implementation Details That Demonstrate Expertise

**1. LCEL Context Preservation**
Using `RunnableParallel` to maintain data flow through complex chains is a sophisticated pattern that:
- Enables comprehensive debugging and observability
- Supports evaluation without modifying core application logic
- Shows understanding of functional composition in LangChain

**2. Security Awareness**
The explicit documentation of `allow_dangerous_deserialization=True` demonstrates:
- Understanding of pickle-based serialization risks
- Security-conscious engineering practices
- Awareness of production vs. development trade-offs

**3. Defensive Coding**
Using `.get()` with defaults in metrics comparison shows:
- Anticipation of edge cases and failure modes
- Building robust CI/CD pipelines
- Production-grade error handling

**4. Professional Workflow**
The Makefile addition demonstrates:
- Command-line proficiency
- Build automation expertise
- Team collaboration awareness

**5. Separation of Concerns**
Offering both `ask_question()` and `ask_question_with_context()` shows:
- Interface design thinking
- Understanding of different use cases (UI vs. evaluation)
- Clean architecture principles

These refinements elevate the project from "good" to "exceptional" and signal to hiring managers that you think like a senior engineer.
