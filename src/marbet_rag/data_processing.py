import os
import config
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory_path: str) -> list:
    """Loads PDF documents from the specified directory using PyMuPDFLoader."""
    print(f"Loading documents from: {directory_path}")
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    all_docs = []
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Warning: No PDF files found in {directory_path}")
        return []

    print(f"Found {len(pdf_files)} PDF files to load.")
    loaded_pages_count = 0
    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        try:
            print(f"  Loading: {pdf_file}...", end="")
            loader = UnstructuredPDFLoader(
                file_path,
                mode="elements",          # Keep: Partitions into meaningful elements
                strategy="hi_res",        # Keep: Best quality via visual layout detection
                languages=["eng"],        # Adjust if other languages are present
                strategy_kwargs={
                    "pdf_infer_table_structure": True, # Improves table extraction accuracy
                    "extract_images_in_pdf": True,     # Uncomment if you need images
                }
            )
            docs = loader.load()
            processed_docs = []
            for i, doc in enumerate(docs):
                doc.metadata['page'] = doc.metadata.get('page_number', i + 1)
                doc.metadata['source'] = doc.metadata.get('filename', pdf_file)
                processed_docs.append(doc)

            all_docs.extend(processed_docs)
            loaded_pages_count += len(processed_docs)
            print(f" OK ({len(processed_docs)} elements/chunks loaded)")
        except Exception as e:
            print(f" FAILED - Error loading {pdf_file}: {e}")

    print(f"Finished loading. Total pages loaded successfully: {loaded_pages_count}")
    return all_docs

def split_documents(documents: list) -> list:
    """Splits documents into smaller chunks based on config."""
    if not documents:
        print("No documents to split.")
        return []
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks 