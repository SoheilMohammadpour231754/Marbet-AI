"""Configuration settings for the Marbet RAG system."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Paths ---
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "data/documents")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vector_store")

# --- Option 1: Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://194.171.191.226:3061")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:32b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest") # Used if LLM_SOURCE is ollama

# --- Option 2: Gemini ---
LLM_SOURCE = os.getenv("LLM_SOURCE", "gemini").lower()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001") # Used if LLM_SOURCE is gemini
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash-latest")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Loaded from .env
GEMINI_CONVERT_SYSTEM_MESSAGE = os.getenv("GEMINI_CONVERT_SYSTEM_MESSAGE", "True").lower() == "true"

# --- LLM Settings ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

# --- Text Splitting ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 128))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))

# --- Retriever ---
# Options: "similarity", "mmr"
RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "mmr")
# Number of source documents to retrieve
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 100))
# Only used for MMR: Number of documents to fetch before passing to MMR algorithm.
RETRIEVER_MMR_FETCH_K = int(os.getenv("RETRIEVER_MMR_FETCH_K", 100))

# --- Chat History ---
# Max number of messages (pairs) to keep in memory (user + assistant = 1 pair)
HISTORY_WINDOW_SIZE = int(os.getenv("HISTORY_WINDOW_SIZE", 6))

# --- Vector Store Rebuild ---
# Set to "True" or "true" to force rebuilding the vector store on startup
# MUST set back to False after next run if chunk/embedding settings don't change!
FORCE_REBUILD_VECTOR_STORE = os.getenv("FORCE_REBUILD_VECTOR_STORE", "False").lower() == "true"