"""
ingest.py
---------
Step 1 of the pipeline: load a PDF -> split into chunks -> embed -> store in FAISS.

Run this directly first, with a test PDF, BEFORE touching the UI.
    python ingest.py path/to/your.pdf

This keeps ingestion and retrieval decoupled: you can re-ingest new documents
without needing the retrieval/generation code to work yet, and vice versa.
"""

import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

VECTORSTORE_DIR = "vectorstore"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def load_document(file_path: str):
    """Load a PDF or TXT into a list of LangChain Document objects."""
    if file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"[ingest] Loaded {len(documents)} item(s) from {file_path}")
    return documents


load_pdf = load_document


def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split documents into smaller overlapping chunks.

    Why chunk at all? Embedding models have a limited context window, and more
    importantly, smaller chunks give more precise retrieval -- you want to pull
    back the 2-3 sentences that answer the question, not an entire 10-page PDF.

    Why overlap? So a sentence that gets cut at a chunk boundary still appears
    fully in at least one chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunk(s)")
    return chunks


def build_vectorstore(chunks, persist_dir: str = VECTORSTORE_DIR):
    """
    Embed each chunk and store the vectors in a FAISS index, then persist it
    to disk so the app doesn't need to re-embed documents on every run.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_dir)
    print(f"[ingest] Vectorstore saved to '{persist_dir}/'")
    return vectorstore


def ingest_pdf(pdf_path: str, persist_dir: str = VECTORSTORE_DIR):
    """Full pipeline: single file path in, saved FAISS index out."""
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks, persist_dir)
    return vectorstore


def ingest_files(file_paths: list, persist_dir: str = VECTORSTORE_DIR):
    """
    Ingest MULTIPLE documents in one go. Every chunk keeps its own filename
    in metadata (set automatically by the loaders as 'source') -- that's what
    lets retrieval later pull the right chunk from the right document instead
    of mixing them up, since similarity search ranks across all chunks from
    all documents together and just returns whichever is closest to the query.
    """
    all_chunks = []
    for path in file_paths:
        documents = load_document(path)
        all_chunks.extend(chunk_documents(documents))

    print(f"[ingest] {len(all_chunks)} total chunk(s) across {len(file_paths)} file(s)")

    embeddings = get_embeddings()

    if os.path.exists(persist_dir):
        vectorstore = FAISS.load_local(
            persist_dir, embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(all_chunks)
    else:
        vectorstore = FAISS.from_documents(all_chunks, embeddings)

    vectorstore.save_local(persist_dir)
    print(f"[ingest] Vectorstore updated at '{persist_dir}/'")
    return vectorstore


def add_pdf_to_existing_vectorstore(pdf_path: str, persist_dir: str = VECTORSTORE_DIR):
    """Kept for backward compatibility -- single-file version of ingest_files."""
    return ingest_files([pdf_path], persist_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py path/to/your.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    ingest_pdf(pdf_path)
