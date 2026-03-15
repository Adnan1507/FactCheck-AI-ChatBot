"""
rag_agent.py
────────────
Agent 4 — RAG (Retrieval Augmented Generation)
Manages the FAISS knowledge base:
  • store_text() : Adds new text to the knowledge base.
  • retrieve()   : Finds the most relevant past knowledge for a query.

The chatbot "learns" from every query and document the user shares.
Embeddings are free via HuggingFace (no API key needed).
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.config import FAISS_PATH, EMBEDDING_MODEL


# ── Load the free HuggingFace embedding model ─────────────────────────────────
# This converts text into numbers (vectors) so FAISS can compare them.
# Downloads ~90MB on first run, then uses the cached version.
_embeddings = None

def _get_embeddings():
    """Returns a shared embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},   # works on any PC — no GPU needed
        )
    return _embeddings


# ── Text Splitter ─────────────────────────────────────────────────────────────
# Splits long text into smaller chunks so FAISS can search them better.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # each chunk = max 500 characters
    chunk_overlap=50,     # 50-char overlap keeps context between chunks
)


def store_text(text: str, source: str = "user_input") -> dict:
    """
    Adds a piece of text to the FAISS knowledge base.
    If the knowledge base doesn't exist yet, it creates one.

    Args:
        text   : The text to store (a claim, article, PDF content, etc.).
        source : Where the text came from (for reference labels).

    Returns:
        A dict with:
          - 'chunks_stored' : How many chunks were added.
          - 'error'         : Error message if something went wrong.
    """
    if not text or not text.strip():
        return {"chunks_stored": 0, "error": "⚠️ No text to store."}

    try:
        embeddings = _get_embeddings()

        # Split text into manageable chunks
        chunks = _splitter.split_text(text)
        docs   = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]

        # If knowledge base already exists → load and add to it
        if os.path.exists(FAISS_PATH):
            db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
        else:
            # First time — create a new knowledge base
            db = FAISS.from_documents(docs, embeddings)

        # Save to disk so it persists between sessions
        db.save_local(FAISS_PATH)

        return {"chunks_stored": len(chunks), "error": None}

    except Exception as e:
        return {"chunks_stored": 0, "error": f"❌ Knowledge base storage failed: {str(e)}"}


def retrieve(query: str, top_k: int = 4) -> dict:
    """
    Searches the knowledge base for content relevant to the query.

    Args:
        query : The claim or question to search for.
        top_k : How many matching chunks to return (default 4).

    Returns:
        A dict with:
          - 'results' : List of relevant text snippets from past knowledge.
          - 'found'   : True if anything was retrieved, False otherwise.
          - 'error'   : Error message if something went wrong.
    """
    # If no knowledge base exists yet, return gracefully
    if not os.path.exists(FAISS_PATH):
        return {
            "results": [],
            "found":   False,
            "error":   None,   # not an error — just empty db
        }

    try:
        embeddings = _get_embeddings()
        db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        # Similarity search — finds the most relevant stored chunks
        docs = db.similarity_search(query, k=top_k)

        results = [
            {
                "content": doc.page_content,
                "source":  doc.metadata.get("source", "unknown"),
            }
            for doc in docs
        ]

        return {
            "results": results,
            "found":   len(results) > 0,
            "error":   None,
        }

    except Exception as e:
        return {
            "results": [],
            "found":   False,
            "error":   f"❌ Knowledge base retrieval failed: {str(e)}",
        }
