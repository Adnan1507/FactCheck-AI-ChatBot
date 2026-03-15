"""
config.py
─────────
Central configuration for the FactCheck AI chatbot.
Loads API keys from .env and defines the 4 available models.
"""

import os
from dotenv import load_dotenv

# Load all keys from the .env file
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

# ── LangSmith Tracing ─────────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"]  = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]     = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGSMITH_PROJECT", "factcheck-ai")

# ── Available Models ──────────────────────────────────────────────────────────
# These 4 are the best free models available across Groq and Gemini
MODELS = {
    "⚡ Llama 3.3 70B (Groq)": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "description": "Most powerful free model. Best for deep fact-checking.",
    },
    "🚀 Llama 3.1 8B Instant (Groq)": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "description": "Fastest model. Great for quick checks.",
    },
    "🌟 Gemini 2.0 Flash (Google)": {
        "provider": "google",
        "model_id": "gemini-2.0-flash",
        "description": "Google's fast model. Great for images & grounding.",
    },
    "🧠 Gemini 1.5 Pro (Google)": {
        "provider": "google",
        "model_id": "gemini-1.5-pro",
        "description": "Google's smartest model. Best for complex analysis.",
    },
}

DEFAULT_MODEL = "⚡ Llama 3.3 70B (Groq)"

# ── HuggingFace Embeddings ─────────────────────────────────────────────────────
# Free embeddings — no API key needed!
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── FAISS Knowledge Base Path ─────────────────────────────────────────────────
import pathlib
BASE_DIR   = pathlib.Path(__file__).resolve().parents[2]   # backend/
FAISS_PATH = str(BASE_DIR / "knowledge_base")
