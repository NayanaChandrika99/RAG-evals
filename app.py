"""
Streamlit UI for RAG Ops Framework

A clean, professional interface for chatting with your document using RAG.
Includes Prometheus metrics for production observability.
"""

import time
import streamlit as st
from prometheus_client import Counter, Histogram, start_http_server

# Import our RAG chain
from src.rag_chain import ask_question


# Initialize Prometheus metrics
# These metrics can be scraped by Prometheus for monitoring
# IMPORTANT: Wrap in try/except to handle Streamlit hot-reloads
try:
    # Start Prometheus metrics server on port 8000
    # Only start once (Streamlit may reload this multiple times)
    if 'metrics_started' not in st.session_state:
        start_http_server(8000)
        st.session_state.metrics_started = True
except Exception:
    # Port already in use - that's fine, metrics server is already running
    pass

# Define metrics - only create once to avoid duplicate registration
if 'prometheus_metrics' not in st.session_state:
    st.session_state.prometheus_metrics = {
        'questions_counter': Counter('rag_questions_total', 'Total questions asked'),
        'response_time_histogram': Histogram('rag_response_seconds', 'Response time in seconds'),
        'errors_counter': Counter('rag_errors_total', 'Total errors encountered')
    }

questions_counter = st.session_state.prometheus_metrics['questions_counter']
response_time_histogram = st.session_state.prometheus_metrics['response_time_histogram']
errors_counter = st.session_state.prometheus_metrics['errors_counter']


# Page configuration
st.set_page_config(
    page_title="RAG Ops Framework",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 RAG Ops Framework")
st.markdown("""
Chat with your document using a production-grade RAG system.

**Architecture**: Hybrid approach with local embeddings + GPT-3.5 generation
**Cost per query**: ~$0.002
**Metrics**: Prometheus endpoint on port 8000
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This is a production-grade RAG (Retrieval-Augmented Generation) system with:

    - ✅ Local embeddings (zero cost)
    - ✅ GPT-3.5 generation
    - ✅ LCEL-based chain composition
    - ✅ Prometheus metrics
    - ✅ Full observability

    ### System Status
    """)

    # Check if FAISS index exists (using pathlib for cross-platform compatibility)
    from pathlib import Path
    if Path("faiss_index").exists():
        st.success("✅ Vector store loaded")
    else:
        st.error("❌ Vector store not found")
        st.warning("Run `make ingest` first!")

    st.markdown("---")
    st.markdown("### Metrics")
    st.markdown(f"Prometheus: http://localhost:8000")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Track metrics
            questions_counter.inc()
            start_time = time.time()

            # Show loading state
            with st.spinner("Thinking..."):
                # Get answer from RAG chain
                answer = ask_question(prompt)

            # Record response time
            response_time = time.time() - start_time
            response_time_histogram.observe(response_time)

            # Display answer
            message_placeholder.markdown(answer)

            # Show response time (for demo purposes)
            st.caption(f"⏱️ Response time: {response_time:.2f}s")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except FileNotFoundError:
            error_msg = "❌ **Error**: FAISS index not found. Please run `make ingest` first to process your document."
            message_placeholder.error(error_msg)
            errors_counter.inc()

        except Exception as e:
            error_msg = f"❌ **Error**: {str(e)}"
            message_placeholder.error(error_msg)
            errors_counter.inc()

            # Show detailed error in expander for debugging
            with st.expander("See error details"):
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    Built with ❤️ using LangChain, FAISS, and Streamlit<br>
    Part of the Production-Grade RAG Ops Framework
</div>
""", unsafe_allow_html=True)

# Clear chat button in sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
