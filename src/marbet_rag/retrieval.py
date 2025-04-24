import os
import shutil
import config
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
# Conditionally import Google GenAI classes
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    # Specify the expected type for clarity
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings as GoogleEmbeddingsType
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI as GoogleChatType
    GOOGLE_GENAI_INSTALLED = True
except ImportError:
    GoogleEmbeddingsType = None # Define as None if not installed
    GoogleChatType = None # Define as None if not installed
    GOOGLE_GENAI_INSTALLED = False
    
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores.utils import filter_complex_metadata

# Import prompts from the dedicated prompts file
from .prompts import contextualize_q_system_prompt, qa_system_prompt
from .data_processing import load_documents, split_documents

# --- Helper Function for Google GenAI Initialization ---

def _initialize_google_genai_component(component_type: str, model_name: str, api_key: str):
    """Helper to initialize Google Generative AI components (LLM or Embeddings)."""
    if not GOOGLE_GENAI_INSTALLED:
        print(f"FATAL: Gemini source selected for {component_type}, but 'langchain-google-genai' is not installed.")
        print("Please install it: pip install langchain-google-genai")
        raise ImportError(f"langchain-google-genai package not found for {component_type}.")
    
    if not api_key:
        print(f"FATAL: Gemini source selected for {component_type}, but GEMINI_API_KEY is not set.")
        raise ValueError(f"GEMINI_API_KEY is required for Gemini {component_type}.")

    component_instance = None
    try:
        if component_type == "embeddings" and GoogleEmbeddingsType:
            print(f"Initializing Gemini embeddings using {model_name}...")
            component_instance = GoogleEmbeddingsType(model=model_name, google_api_key=api_key)
            _ = component_instance.embed_query("test connection") # Test connection
            print(f"Successfully initialized Gemini embeddings.")
        elif component_type == "llm" and GoogleChatType:
            print(f"Initializing Gemini LLM {model_name}...")
            component_instance = GoogleChatType(
                model=model_name, 
                google_api_key=api_key, 
                temperature=config.LLM_TEMPERATURE, # Use configured temperature
                convert_system_message_to_human=config.GEMINI_CONVERT_SYSTEM_MESSAGE # Use configured conversion
            )
            _ = component_instance.invoke("Respond with OK if you are ready.") # Test connection
            print(f"Successfully initialized Gemini LLM.")
        else:
             # Should not happen if GOOGLE_GENAI_INSTALLED is True, but safety check
             raise TypeError(f"Unsupported Google GenAI component type or component class not loaded: {component_type}")
             
    except Exception as e:
        print(f"FATAL: Error initializing Gemini {component_type} (model: {model_name}).")
        print(f"Error details: {e}")
        print(f"Please ensure your GEMINI_API_KEY is correct and the model '{model_name}' is valid.")
        raise # Re-raise the exception after logging
        
    if component_instance is None:
        # This case might occur if the try block somehow completes without assignment or error
        print(f"FATAL: Google GenAI {component_type} object could not be initialized.")
        raise RuntimeError(f"Google GenAI {component_type} initialization failed unexpectedly.")
        
    return component_instance

# --- Core Setup Functions ---

def get_vector_store(chunks: list, force_recreate: bool = False) -> Chroma:
    """Creates or loads a Chroma vector store using the configured embedding model."""
    persist_directory = config.VECTOR_DB_PATH
    embeddings: Embeddings | None = None # Type hint for clarity
    embedding_model_name = ""

    if force_recreate and os.path.exists(persist_directory):
        print(f"Removing existing vector store at: {persist_directory}")
        shutil.rmtree(persist_directory)

    # --- Initialize Embeddings based on LLM_SOURCE ---
    if config.LLM_SOURCE == "ollama":
        embedding_model_name = config.EMBEDDING_MODEL
        base_url = config.OLLAMA_BASE_URL
        try:
            print(f"Initializing Ollama embeddings using {embedding_model_name} from {base_url}...")
            embeddings = OllamaEmbeddings(model=embedding_model_name, base_url=base_url)
            _ = embeddings.embed_query("test connection") # Test connection
            print(f"Successfully initialized Ollama embeddings.")
        except Exception as e:
            print(f"FATAL: Error initializing Ollama embeddings (model: {embedding_model_name}, url: {base_url}).")
            print(f"Error details: {e}")
            print(f"Please ensure Ollama server is running, accessible, and the model '{embedding_model_name}' is available.")
            raise
            
    elif config.LLM_SOURCE == "gemini":
        embeddings = _initialize_google_genai_component(
            component_type="embeddings",
            model_name=config.GEMINI_EMBEDDING_MODEL,
            api_key=config.GEMINI_API_KEY
        )
    else:
        print(f"FATAL: Invalid LLM_SOURCE configured for embeddings: '{config.LLM_SOURCE}'. Must be 'ollama' or 'gemini'.")
        raise ValueError(f"Invalid LLM_SOURCE for embeddings: {config.LLM_SOURCE}")
        
    if embeddings is None:
        # Should be caught by earlier checks, but defensive programming
        print("FATAL: Embeddings object could not be initialized.")
        raise RuntimeError("Embeddings initialization failed.")

    # --- Load or Create Vector Store ---
    if os.path.exists(persist_directory):
        print(f"Loading existing vector store from: {persist_directory}")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Existing vector store loaded.")
    else:
        if not chunks:
            print("Error: No document chunks provided to create a new vector store.")
            raise ValueError("Cannot create vector store with empty chunks list.")
        print(f"Creating new vector store at: {persist_directory} with {len(chunks)} chunks...")
        filtered_chunks = filter_complex_metadata(chunks)
        print(f"Filtered complex metadata. Remaining chunks: {len(filtered_chunks)}")
        if not filtered_chunks:
            print("Warning: All chunks were filtered out after removing complex metadata. Vector store will be empty.")

        vector_store = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print("New vector store created and persisted.")
    return vector_store

def setup_rag_chain(vector_store: Chroma):
    """Sets up the RAG retrieval chain with history awareness using LCEL."""
    print("Setting up RAG chain with history using LCEL...")

    llm: BaseChatModel | None = None # Type hint for clarity

    if config.LLM_SOURCE == "ollama":
        try:
            print(f"Initializing Ollama LLM {config.OLLAMA_LLM_MODEL} from {config.OLLAMA_BASE_URL}...")
            llm = ChatOllama(
                model=config.OLLAMA_LLM_MODEL, 
                base_url=config.OLLAMA_BASE_URL, 
                temperature=config.LLM_TEMPERATURE # Use configured temperature
            )
            _ = llm.invoke("Respond with OK if you are ready.") # Test connection
            print(f"Successfully initialized Ollama LLM.")
        except Exception as e:
            print(f"FATAL: Error initializing Ollama LLM (model: {config.OLLAMA_LLM_MODEL}, url: {config.OLLAMA_BASE_URL}).")
            print(f"Error details: {e}")
            print(f"Please ensure Ollama server is running, accessible, and the model '{config.OLLAMA_LLM_MODEL}' is downloaded/available.")
            raise
    elif config.LLM_SOURCE == "gemini":
         llm = _initialize_google_genai_component(
            component_type="llm",
            model_name=config.GEMINI_LLM_MODEL,
            api_key=config.GEMINI_API_KEY
         )
    else:
        print(f"FATAL: Invalid LLM_SOURCE configured: '{config.LLM_SOURCE}'. Must be 'ollama' or 'gemini'.")
        raise ValueError(f"Invalid LLM_SOURCE: {config.LLM_SOURCE}")

    if llm is None:
         # Should be caught by earlier checks, but defensive programming
         print("FATAL: LLM object could not be initialized.")
         raise RuntimeError("LLM initialization failed.")

    retriever_search_kwargs = {"k": config.RETRIEVER_K}
    if config.RETRIEVER_SEARCH_TYPE.lower() == "mmr":
        retriever_search_kwargs["fetch_k"] = config.RETRIEVER_MMR_FETCH_K

    retriever = vector_store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs=retriever_search_kwargs
    )
    print(f"Retriever setup: type={config.RETRIEVER_SEARCH_TYPE}, k={config.RETRIEVER_K}", end="")
    if config.RETRIEVER_SEARCH_TYPE.lower() == "mmr":
        print(f", fetch_k={config.RETRIEVER_MMR_FETCH_K}")
    else:
        print()

    # 1. Contextualize question prompt and retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt), # Use imported constant
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Define the prompt for the final answer generation
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt), # Use imported constant
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 3. Define the RAG chain using LCEL
    def format_docs_for_context(docs):
        """Formats retrieved documents for insertion into the prompt context."""
        if not docs:
            return "No context provided."
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            formatted_docs.append(f"Source {i+1} [{source}, Page {page}]:\n{doc.page_content}")
        return "\n\n".join(formatted_docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_for_context(x["context"]))) 
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain = (
        RunnablePassthrough.assign(
            context=history_aware_retriever
        )
        .assign(answer=rag_chain_from_docs)
    )

    print("History-aware RAG chain setup complete using LCEL.")
    return final_rag_chain 

def setup_chatbot():
    """Initializes the vector store and RAG chain based on config.

    Handles loading documents, splitting, vector store creation/loading,
    and RAG chain setup. Exits if critical errors occur.

    Returns:
        The initialized LangChain RAG runnable.
    """
    print("--- Initializing Marbet RAG Chatbot Dependencies ---")
    # Determine which LLM and Embedding model names to display based on source
    llm_model_display = config.OLLAMA_LLM_MODEL if config.LLM_SOURCE == "ollama" else config.GEMINI_LLM_MODEL
    embedding_model_display = config.EMBEDDING_MODEL if config.LLM_SOURCE == "ollama" else config.GEMINI_EMBEDDING_MODEL
    print(f"Configuration: PDF_DIR='{config.PDF_DIRECTORY}', DB_PATH='{config.VECTOR_DB_PATH}', EMBED_SRC='{config.LLM_SOURCE}', EMBED_MDL='{embedding_model_display}', LLM_SRC='{config.LLM_SOURCE}', LLM_MDL='{llm_model_display}', FORCE_REBUILD='{config.FORCE_REBUILD_VECTOR_STORE}'")
    vector_store = None
    try:
        # --- Initialization Phase (Load/Index Documents) ---
        if config.FORCE_REBUILD_VECTOR_STORE or not os.path.exists(config.VECTOR_DB_PATH):
            print("Attempting to build/rebuild vector store...")
            docs = load_documents(config.PDF_DIRECTORY)
            if docs:
                chunks = split_documents(docs)
                vector_store = get_vector_store(chunks, force_recreate=config.FORCE_REBUILD_VECTOR_STORE)
            else:
                print("Error: No documents loaded. Cannot initialize vector store.")
                raise RuntimeError("Failed to load documents for vector store initialization.")
        else:
            # Just load the existing store
            print(f"Found existing vector store at {config.VECTOR_DB_PATH}. Loading...")
            # Ensure we load it correctly even if chunks list is empty
            vector_store = get_vector_store([], force_recreate=False)

        if not vector_store:
            print("Error: Failed to initialize vector store.")
            raise RuntimeError("Vector store initialization failed.")
        print("Vector store is ready.")

        # --- Setup RAG Chain ---
        rag_chain = setup_rag_chain(vector_store)
        print("--- Chatbot Setup Complete ---")
        return rag_chain

    except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
        print(f"\nFATAL ERROR DURING CHATBOT SETUP: {e}")
        import traceback
        traceback.print_exc()
        # In a real application, you might want more robust error handling
        # or a way to signal the failure without exiting, but for now, exiting
        # prevents the API/CLI from starting in a broken state.
        exit(1) # Exit if setup fails 