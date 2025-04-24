# Marbet AI Event Assistant - RAG Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to assist attendees of Marbet incentive trips. The chatbot answers questions based *only* on provided event documents (PDFs), ensuring accurate and contextually relevant information delivery.

It leverages local Large Language Models (LLMs) and embedding models served via Ollama *or* cloud-based Google Gemini models, configured via environment variables. LangChain provides the orchestration framework, and ChromaDB handles local vector storage.

This system was developed as part of the Marbet challenge, aiming to enhance the guest experience by providing instant access to event schedules, packing lists, ship/venue information, and other relevant policies.

## Features

*   **Retrieval-Augmented Generation (RAG):** Answers questions by retrieving relevant information from provided PDF documents before generating a response.
*   **Flexible LLM/Embedding Source:** Supports local Ollama models (`config.LLM_SOURCE='ollama'`) or Google Gemini models (`config.LLM_SOURCE='gemini'`, default).
*   **Advanced PDF Document Processing:** Loads and processes text content from multiple PDF files using `UnstructuredPDFLoader` with strategies like `hi_res` for better layout and table extraction. Requires Tesseract OCR for image-based content.
*   **Configurable Text Chunking:** Splits documents into manageable chunks using `RecursiveCharacterTextSplitter` with configurable size (`config.CHUNK_SIZE`, default: 128) and overlap (`config.CHUNK_OVERLAP`, default: 20).
*   **Ollama/Gemini Integration:** Utilizes user-specified LLMs and embedding models hosted on an Ollama server or via the Gemini API.
*   **Vector Storage:** Employs ChromaDB to store and efficiently query document embeddings locally.
*   **Advanced Retrieval:** Uses Maximal Marginal Relevance (MMR) search by default (`config.RETRIEVER_SEARCH_TYPE='mmr'`) to retrieve diverse and relevant document chunks (`config.RETRIEVER_K`, default: 100). Similarity search is also available.
*   **Conversational Memory:** Maintains chat history (`config.HISTORY_WINDOW_SIZE`) to understand and respond to follow-up questions contextually.
*   **Source Attribution:** The RAG pipeline provides context labeled with source document/page to the LLM. The API extracts potential source filenames from the LLM response and returns relevant source metadata alongside the cleaned answer.
*   **Configurable:** Key parameters (models, paths, chunking, retrieval settings) are managed via `config.py` and environment variables (`.env` file).
*   **Interfaces:** Includes a Command-Line Interface (`main.py`) and a Flask API (`api.py`) to serve a basic React frontend.

## Technology Stack

*   **Python 3.9+**
*   **LangChain:** Framework for developing applications powered by language models.
*   **Ollama:** Service for running local LLMs and embedding models (optional, if `LLM_SOURCE='ollama'`).
*   **Google Generative AI SDK:** Used via `langchain-google-genai` for Gemini models (optional, if `LLM_SOURCE='gemini'`).
*   **LangChain Community Libraries:** For specific integrations (`ChatOllama`, `OllamaEmbeddings`, `Chroma`, `UnstructuredPDFLoader`).
*   **Unstructured:** Library for processing complex, unstructured documents like PDFs, leveraging models and rules. Requires `unstructured[pdf]`.
*   **Tesseract OCR Engine:** Required by `Unstructured` for extracting text from images within PDFs. Must be installed separately.
*   **ChromaDB:** Open-source embedding database.
*   **React:** JavaScript library for building user interfaces (for the example frontend).
*   **Vite:** Frontend build tool (for the example frontend).
*   **Axios:** Promise-based HTTP client (for the example frontend).
*   **python-dotenv:** For loading environment variables from `.env`.

## Project Structure

```
RAG-Marbet/ # Renamed parent folder likely
├── .env                 # For API keys and environment variables (GITIGNORED)
├── .gitignore
├── data/
│   ├── documents/         # Input PDF documents
│   └── vector_store/      # Persisted ChromaDB vector store (add to .gitignore)
├── frontend/              # React frontend application
│   ├── public/
│   ├── src/               # Frontend source code (components, App.jsx, etc.)
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── notebooks/
│   └── notebook.ipynb     # Experimentation notebook
├── src/
│   └── marbet_rag/        # Main application package
│       ├── __init__.py
│       ├── data_processing.py # Document loading/splitting
│       ├── prompts.py       # Prompt templates
│       ├── retrieval.py   # Vector store and RAG chain setup
│       └── utils.py       # Helper functions
├── tests/
│   └── __init__.py        # Test package
├── api.py                 # Flask API entry point
├── config.py              # Main configuration (reads from env vars)
├── main.py                # CLI application entry point
├── README.md              # This file
└── requirements.txt       # Dependencies
```
(Note: Structure slightly adjusted based on common practice and observed files)

## Setup Instructions

### 1. Prerequisites

*   Python 3.9 or higher installed.
*   `pip` (Python package installer).
*   Node.js (which includes `npm`) installed (LTS version recommended) - *Only if running the example frontend*.
*   **Tesseract OCR Engine:** **Required** for processing PDFs (especially scanned ones) by the `UnstructuredPDFLoader`.
    *   Install Tesseract separately and ensure it's available in your system's PATH.
    *   **Windows:** Download from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page. **Ensure you select the option to add Tesseract to the system PATH during installation.**
    *   **macOS:** Use Homebrew: `brew install tesseract`
    *   **Linux (Debian/Ubuntu):** `sudo apt-get update && sudo apt-get install tesseract-ocr`
    *   *Note:* You might also need to install language packs (e.g., `tesseract-ocr-eng` for English).
*   **Ollama Server Access (Optional):** If using `LLM_SOURCE='ollama'`, ensure you can reach the specified Ollama server (requires BUas network or VPN).
*   **Google Cloud / AI Studio Account (Optional):** If using `LLM_SOURCE='gemini'`, you need a Google account to generate an API key.
*   Git (optional, for cloning).

### 2. Clone the Repository (Optional)

```bash
# If you have the project files already, skip this step
# git clone <repository_url>
# cd <repository_directory>
```

### 3. Set Up Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Ensure you have the `requirements.txt` file in your project directory.

```bash
pip install -r requirements.txt
```
*Note:* This installs base dependencies including `langchain`, `chromadb`, `unstructured[pdf]`, `python-dotenv`. It also includes `langchain-google-genai` and potentially `ollama` depending on the finalized `requirements.txt`.

### 5. Prepare Event Documents

*   Place all relevant Marbet event PDF documents inside the `data/documents/` directory. The system will attempt to load all `.pdf` files found here.

### 6. Configure LLM Source and API Keys

*   **Select LLM Source:** The primary way to configure is via environment variables. Create a `.env` file in the project root directory.
    *   To use **Gemini (default)**:
        ```dotenv
        # .env
        LLM_SOURCE="gemini"
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        # Optional: Override default Gemini models
        # GEMINI_LLM_MODEL="gemini-1.5-pro-latest"
        # GEMINI_EMBEDDING_MODEL="models/text-embedding-004"
        ```
        Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   To use **Ollama**:
        ```dotenv
        # .env
        LLM_SOURCE="ollama"
        # Optional: Override default Ollama settings
        # OLLAMA_BASE_URL="http://localhost:11434"
        # OLLAMA_LLM_MODEL="llama3"
        # EMBEDDING_MODEL="nomic-embed-text"
        ```
        Ensure the specified `OLLAMA_BASE_URL` is reachable and the models are available on the server.
*   The `.env` file is automatically ignored by Git (via `.gitignore`).
*   You can also modify defaults directly in `config.py`, but using `.env` is recommended, especially for secrets.

## Configuration

Configuration is primarily managed via environment variables loaded from a `.env` file located in the project root, with fallbacks defined in `config.py`.

Key configurable parameters (set via `.env` or modify defaults in `config.py`):

*   `PDF_DIRECTORY`: Path to the input documents (Default: `data/documents`).
*   `VECTOR_DB_PATH`: Path for the persisted vector database (Default: `data/vector_store`).
*   `LLM_SOURCE`: Choose the LLM provider: `'gemini'` (default) or `'ollama'`. This also determines the embedding model source.
*   `GEMINI_API_KEY`: Your Google Gemini API key (Required if `LLM_SOURCE='gemini'`).
*   `GEMINI_LLM_MODEL`: Gemini model for generation (Default: `'gemini-1.5-flash-latest'`).
*   `GEMINI_EMBEDDING_MODEL`: Gemini model for embeddings (Default: `'models/embedding-001'`).
*   `GEMINI_CONVERT_SYSTEM_MESSAGE`: Convert system messages for Gemini (Default: `True`).
*   `OLLAMA_BASE_URL`: URL of the Ollama server (Default: `http://194.171.191.226:3061`, used if `LLM_SOURCE='ollama'`).
*   `OLLAMA_LLM_MODEL`: Name of the LLM on the Ollama server (Default: `deepseek-r1:32b`, used if `LLM_SOURCE='ollama'`).
*   `EMBEDDING_MODEL`: Name of the embedding model on the Ollama server (Default: `mxbai-embed-large:latest`, used if `LLM_SOURCE='ollama'`).
*   `LLM_TEMPERATURE`: LLM temperature setting (Default: `0.0`).
*   `CHUNK_SIZE`: Target size for document chunks (Default: `128`).
*   `CHUNK_OVERLAP`: Overlap between chunks (Default: `20`).
*   `RETRIEVER_SEARCH_TYPE`: Retrieval method (`'mmr'` (default) or `'similarity'`).
*   `RETRIEVER_K`: Number of documents to retrieve for context (Default: `100`).
*   `RETRIEVER_MMR_FETCH_K`: (Only for MMR) Initial number of documents to fetch (Default: `100`).
*   `HISTORY_WINDOW_SIZE`: Number of conversation turns (user + AI messages) to keep in memory (Default: `6`).
*   `FORCE_REBUILD_VECTOR_STORE`: Set to `True` (e.g., `FORCE_REBUILD_VECTOR_STORE="True"` in `.env`) to force re-indexing of documents on startup. **Important:** Required if `PDF_DIRECTORY`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, embedding model, or document loading strategy changes. Remember to set it back to `False` after the rebuild.

## Running the Application

### Running the Backend (Python CLI)

1.  Ensure all backend setup steps (Python environment, dependencies, documents, `.env` file configured) are complete.
2.  Navigate to the project's root directory in your terminal.
3.  Activate your virtual environment (`source venv/bin/activate` or `.\venv\Scripts\activate`).
4.  Run the main script:

    ```bash
    python main.py
    ```

4.  **Initial Run / Rebuild:** If `FORCE_REBUILD_VECTOR_STORE` is `True` in `config.py`, the vector store will be rebuilt from the documents in `PDF_DIRECTORY`. This may take some time depending on the number and size of documents. Set it back to `False` for subsequent runs.
5.  Interact with the chatbot via the command line prompts.

### Running the Backend (Flask API Server)

**Note:** This server provides the API endpoint used by the React frontend.

1.  Ensure all backend setup steps (Python environment, dependencies, documents, `.env` configured) are complete.
2.  Navigate to the project's root directory in your terminal.
3.  Activate your virtual environment.
4.  Run the API server script:
    ```bash
    python api.py
    ```
4.  The server will start, initialize the chatbot (this might take a moment), and listen for requests (usually on `http://localhost:5000` or `http://0.0.0.0:5000`).
5.  Keep this server running while using the frontend UI.

### Running the Frontend (React UI)

**Note:** The frontend requires the Flask API server (see above) to be running.

1.  Ensure you have Node.js and `npm` installed.
2.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
3.  Install frontend dependencies:
    ```bash
    npm install
    ```
    *   *(Troubleshooting)* If you encounter issues with `npm install` or `npm run dev` on Windows PowerShell regarding execution policies, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in your terminal or use a different terminal like Command Prompt (cmd) or Git Bash.
4.  Start the frontend development server:
    ```bash
    npm run dev
    ```
5.  Open your web browser and navigate to the local URL provided by Vite (usually `http://localhost:5173/`).
6.  Interact with the chatbot UI.