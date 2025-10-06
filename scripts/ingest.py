"""
Document Ingestion Script for RAG Ops Framework

This script loads a PDF document, splits it into chunks, creates embeddings using
a LOCAL model (cost: $0), and stores them in a FAISS vector store.

COST OPTIMIZATION: Using local embeddings (all-mpnet-base-v2) saves ~$0.0004 per chunk.
For a typical 50-page document with 100 chunks, this saves ~$0.04 per ingestion.
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def ingest_document(
    pdf_path: str = "data/source_document.pdf",
    index_path: str = "faiss_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"  # 768-dim, max ~384 tokens
):
    """
    Ingest a PDF document and create a FAISS vector store.

    Args:
        pdf_path: Path to the PDF document
        index_path: Path where FAISS index will be saved
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model_name: HuggingFace model for embeddings
    """

    # Check if document exists (using pathlib for modern, cross-platform path handling)
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: Document not found at {pdf_path}")
        print(f"Please place your PDF document at {pdf_path}")
        sys.exit(1)

    print(f"📄 Loading document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    # Split into chunks
    print(f"\n📝 Splitting document into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    # Create local embeddings (FREE!)
    print(f"\n🤖 Loading embedding model: {embedding_model_name}")
    print("   This may take a few minutes on first run (downloading model)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"}  # Use "cuda" if GPU available
    )
    print("✅ Embedding model loaded")

    # Create and save vector store
    print(f"\n💾 Creating FAISS index and computing embeddings...")
    print(f"   Processing {len(chunks)} chunks (this may take a few minutes)...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print("✅ FAISS index created")

    # Save the index
    print(f"\n💾 Saving index to {index_path}...")
    vectorstore.save_local(index_path)
    print(f"✅ Index saved successfully!")

    # Print summary
    print("\n" + "="*60)
    print("✨ Ingestion Complete!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   - Pages processed: {len(documents)}")
    print(f"   - Chunks created: {len(chunks)}")
    print(f"   - Embedding model: {embedding_model_name}")
    print(f"   - Index location: {index_path}")
    print(f"   - Cost: $0 (local embeddings)")
    print("="*60)
    print(f"\n✅ Ready to use! Run 'make run' or 'streamlit run app.py' to start chatting")


if __name__ == "__main__":
    # Allow optional command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDF document for RAG system")
    parser.add_argument(
        "--pdf",
        default="data/source_document.pdf",
        help="Path to PDF document"
    )
    parser.add_argument(
        "--output",
        default="faiss_index",
        help="Output path for FAISS index"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks"
    )

    args = parser.parse_args()

    try:
        ingest_document(
            pdf_path=args.pdf,
            index_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
