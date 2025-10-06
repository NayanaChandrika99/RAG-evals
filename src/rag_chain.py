"""
RAG Chain Implementation with LCEL and Context Tracking

This module implements the core RAG logic using LangChain Expression Language (LCEL).
It provides two interfaces:
1. ask_question() - Simple interface for UI (returns just the answer)
2. ask_question_with_context() - For evaluation (returns answer + retrieved contexts)

KEY ARCHITECTURAL DECISION: Hybrid Model Strategy
- Embeddings: Local model (all-mpnet-base-v2) - $0 cost
- Generation: Proprietary model (GPT-3.5/Claude) - ~$0.002 per query
- Total cost per query: ~$0.002 (vs. ~$0.006 with API embeddings)
"""

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME = "gpt-3.5-turbo"  # or use "gpt-4" for higher quality
FAISS_INDEX_PATH = "faiss_index"
TEMPERATURE = 0  # Deterministic responses for consistency


def initialize_rag_chain():
    """
    Initialize the RAG chain with local embeddings and proprietary LLM.

    Returns:
        tuple: (rag_chain, full_chain) - Simple chain and context-preserving chain
    """

    # Load local embeddings (must match the model used in ingestion)
    print("🔄 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

    # Load FAISS index with security note
    # SECURITY NOTE: allow_dangerous_deserialization=True is used because we control
    # the source of the index file (created by our ingestion script).
    # In a production system where the index might come from an untrusted source,
    # this would require careful security review and additional validation.
    print("🔄 Loading FAISS index...")
    index_path = Path(FAISS_INDEX_PATH)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Please run 'make ingest' or 'python scripts/ingest.py' first."
        )

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
    )

    # Use a powerful, fast proprietary model
    print("🔄 Initializing LLM...")
    model = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Define prompt template
    template = """Answer the question based only on the following context. If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # ADVANCED PATTERN: Context Preservation with RunnableParallel
    # This is critical for evaluation - we need to preserve retrieved contexts
    # even after they've been consumed by the LLM.

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    # Basic chain for answer generation (used by UI)
    rag_chain = (
        setup_and_retrieval
        | prompt
        | model
        | StrOutputParser()
    )

    # Full chain that returns both answer and contexts (used by evaluation)
    # This demonstrates sophisticated use of LCEL for production observability
    def create_full_response(x):
        # x is the output from setup_and_retrieval: {"context": docs, "question": question}
        # Generate the answer
        answer = (prompt | model | StrOutputParser()).invoke(x)
        # Return both answer and contexts
        return {
            "answer": answer,
            "contexts": x["context"]
        }

    full_chain = setup_and_retrieval | create_full_response

    print("✅ RAG chain initialized successfully!")
    return rag_chain, full_chain


# Initialize chains at module level
_rag_chain = None
_full_chain = None


def get_chains():
    """Lazy initialization of chains"""
    global _rag_chain, _full_chain
    if _rag_chain is None or _full_chain is None:
        _rag_chain, _full_chain = initialize_rag_chain()
    return _rag_chain, _full_chain


def ask_question(question: str) -> str:
    """
    Simple interface for the UI - returns just the answer.

    Args:
        question: User's question

    Returns:
        str: Generated answer
    """
    rag_chain, _ = get_chains()
    return rag_chain.invoke(question)


def ask_question_with_context(question: str) -> Dict[str, any]:
    """
    Enhanced interface for evaluation - returns both answer and retrieved contexts.

    This function is specifically designed for RAGAS evaluation, which needs
    access to both the generated answer and the retrieved context documents
    to compute metrics like faithfulness and context relevance.

    Args:
        question: User's question

    Returns:
        dict: {
            "answer": str,
            "contexts": List[str]  # List of retrieved document contents
        }
    """
    _, full_chain = get_chains()
    result = full_chain.invoke(question)

    return {
        "answer": result["answer"],
        "contexts": [doc.page_content for doc in result["contexts"]]
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing RAG Chain...")
    print("=" * 60)

    try:
        # Test simple question
        test_question = "What is the main topic of this document?"

        print(f"\n📝 Question: {test_question}")
        print("\n🤔 Generating answer...")

        # Test both interfaces
        simple_answer = ask_question(test_question)
        print(f"\n✅ Simple Answer:\n{simple_answer}")

        detailed_result = ask_question_with_context(test_question)
        print(f"\n✅ Detailed Result:")
        print(f"   Answer: {detailed_result['answer']}")
        print(f"   Number of contexts: {len(detailed_result['contexts'])}")
        print(f"   First context preview: {detailed_result['contexts'][0][:200]}...")

        print("\n" + "=" * 60)
        print("✨ RAG chain is working correctly!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
