# rag_processor.py
# ollama pull bge-m3
# pip install chromadb
import ollama
import chromadb
from chromadb.utils import embedding_functions
import traceback

OLLAMA_EMBED_MODEL = "bge-m3"

# --- ChromaDB Client Initialization ---
try:
    client = chromadb.Client()
except Exception as e:
    print(f"[DEBUG RAG CLIENT] CRITICAL ERROR initializing ChromaDB client: {e}")
    traceback.print_exc()
    raise RuntimeError("Failed to initialize ChromaDB") from e


# --- Auxiliar functions ---
def get_ollama_embedding_function():
    """
    Creates and returns an OllamaEmbeddingFunction instance.
    This function is used to generate embeddings for text data.
    """
    try:
        ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=OLLAMA_EMBED_MODEL
        )
        return ef
    except Exception as e_ef:
        print(f"[DEBUG RAG EF] ERROR creating OllamaEmbeddingFunction: {e_ef}")
        traceback.print_exc()
        raise

def create_or_get_collection_for_patient(collection_name: str, ef_to_use):
    """
    Creates or retrieves a collection for a specific patient in ChromaDB.
    """
    try:
        client.delete_collection(name=collection_name)
    except chromadb.errors.NotFoundError: 
        print(f"The collection '{collection_name}' did not exist and will be created.")
    except Exception as e_del:
        print(f"Failed to delete '{collection_name}' (error: {e_del}). Will attempt to create it.")

    try:
        collection = client.create_collection(name=collection_name, embedding_function=ef_to_use)
        return collection
    except chromadb.errors.DuplicateCollectionError: 
        collection = client.get_collection(name=collection_name, embedding_function=ef_to_use)
        return collection
    except Exception as e_create:
        print(f"Critical error creating/getting collection '{collection_name}': {e_create}")
        traceback.print_exc()
        raise

# --- Main Indexing Function ---
def index_patient_texts(id_paciente: str, clinical_record: dict):
    """
    Indexes the clinical text entries for a given patient in ChromaDB using Ollama embeddings.
    Raises an error if no text entries are found.
    """
    collection_name = f"pacient_{id_paciente.replace('-', '_')}_ollama_rag_data"

    # Ensure clinical_record contains text entries
    text_entries = clinical_record.get('text_entries')
    if not text_entries:
        raise ValueError(f"No clinical text entries found for patient {id_paciente}. Cannot proceed with indexing.")

    # Prepare embedding function
    try:
        ollama_ef = get_ollama_embedding_function()
    except Exception:
        print("Failed to initialize OllamaEmbeddingFunction. Aborting indexing.")
        return

    # Create or retrieve collection
    try:
        patient_collection = create_or_get_collection_for_patient(collection_name, ollama_ef)
    except Exception:
        print("Failed to create or retrieve patient collection. Aborting indexing.")
        return

    # Create metadata and unique IDs for each document
    metadatas = [{"source": f"clinical_record_{id_paciente}", "doc_idx": i} for i in range(len(text_entries))]
    ids = [f"doc_{id_paciente}_{i}" for i in range(len(text_entries))]

    try:
        patient_collection.add(
            documents=text_entries,
            metadatas=metadatas,
            ids=ids
        )
        current_count = patient_collection.count()
        if current_count <= 0:
            print(f"[WARNING] .add() completed but collection is still empty for '{collection_name}'.")
    except Exception as e_add:
        print(f"[ERROR] Failed during patient_collection.add(): {e_add}")
        traceback.print_exc()


# --- Función de Recuperación (ajustada para errores comunes de ChromaDB 0.4.x) ---
def retrieve_relevant_chunks(patient_id: str, query_text: str, n_results: int = 5):
    """
    Retrieves the most relevant text chunks for a given patient using a vector search in ChromaDB.
    
    Args:
        patient_id (str): The patient's unique identifier.
        query_text (str): The query used to retrieve relevant documents.
        n_results (int): Maximum number of documents to retrieve.

    Returns:
        list: A list of retrieved document strings.
    """
    collection_name = f"pacient_{patient_id.replace('-', '_')}_ollama_rag_data"

    try:
        embedding_function = get_ollama_embedding_function()
        collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

        doc_count = collection.count()
        if doc_count == 0:
            print(f"[RAG RETRIEVE] Collection '{collection_name}' is empty.")
            return []

        num_to_retrieve = min(n_results, doc_count)
        if num_to_retrieve <= 0:
            print(f"[RAG RETRIEVE] No documents available to retrieve.")
            return []

        results = collection.query(query_texts=[query_text], n_results=num_to_retrieve)
        retrieved_docs = results.get("documents", [[]])[0] if results else []
        print(f"[RAG RETRIEVE] Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs

    except chromadb.errors.NotFoundError:
        print(f"[RAG RETRIEVE] ERROR: Collection '{collection_name}' not found.")
        print(f"[RAG RETRIEVE] Ensure the patient '{patient_id}' was properly indexed first.")
        return []

    except Exception as e:
        print(f"[RAG RETRIEVE] Unexpected error during retrieval: {e}")
        traceback.print_exc()
        return []
