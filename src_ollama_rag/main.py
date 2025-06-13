# main.py
import time
import traceback
from src_ollama_rag.build_structured_report import build_structured_info
from src_ollama_rag.generate_narrative import generate_summary_with_rag
from src_ollama_rag.utils import load_datasets, build_clinical_record, extract_free_texts
from src_ollama_rag.rag_processor import index_patient_texts, retrieve_relevant_chunks, OLLAMA_EMBED_MODEL
from src_ollama_rag.ollama_runner import is_ollama_running, start_ollama_server

def split_into_chunks(text: str, max_length: int = 512) -> list:
    """
    Splits the input text into chunks of a specified maximum length.
    Args:
        text (str): The input text to be split.
        max_length (int): The maximum length of each chunk.
    Returns:
        list: A list of text chunks.
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def main():
    if not is_ollama_running():
        print("The Ollama server is not running.")
        try:
            start_ollama_server()
            time.sleep(10)
            if not is_ollama_running():
                print("The Ollama server could not be started. Please start it manually and run the script again.")
                return
        except Exception as e:
            print(f"Error starting Ollama server: {e}")
            return
    else:
        print("Ollama server is already running.")

    patient_id = input("Enter the patient ID: ").strip()
    if not patient_id:
        print("Invalid patient ID.")
        return

    # Load datasets
    try:
        patients, episodes, movements, diagnoses, texts_df = load_datasets()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        traceback.print_exc()
        return

    # Validate patient existence
    if patient_id not in patients['id_paciente'].values:
        print(f"Patient ID '{patient_id}' not found in the dataset.")
        return

    # Build structured data
    structured_data, episode_timeline = build_structured_info(patient_id, patients, episodes)

    # Build complete clinical record (for indexing)
    clinical_record_dict = build_clinical_record(patient_id, patients, episodes, movements, diagnoses, texts_df)
    full_clinical_text = extract_free_texts(clinical_record_dict)

    # --- RAG STEP: Prepare and index patient texts ---
    record_for_indexing = {'text_entries': []}

    if texts_df is not None and not texts_df.empty:
        patient_texts_df = texts_df[texts_df['id_paciente'] == patient_id]
        if not patient_texts_df.empty and 'texto_nota' in patient_texts_df.columns:
            note_texts = patient_texts_df['texto_nota'].dropna().astype(str).tolist()
            if note_texts:
                record_for_indexing['text_entries'] = note_texts
    
    if not record_for_indexing['text_entries'] and full_clinical_text:
        chunks = split_into_chunks(full_clinical_text)
        if chunks:
            record_for_indexing['text_entries'] = chunks
        else:
            print("Could not generate chunks from the clinical text.")
            return

    elif not record_for_indexing['text_entries']:
        print(f"No texts found to index for patient {patient_id}.")
        return

    try:
        index_patient_texts(patient_id, record_for_indexing)
    except Exception as e:
        print(f"Error during text indexing: {e}")
        traceback.print_exc()
        return

    # --- RAG STEP: Query for retrieval ---
    retrieval_query = f"General clinical summary of patient {patient_id}, including clinical course, relevant history, and active problems."
    retrieved_chunks = retrieve_relevant_chunks(patient_id, retrieval_query, n_results=7)

    #print("\n--- RETRIEVED CHUNKS  ---")
    for i, ch in enumerate(retrieved_chunks):
        print(f"[{i+1}] {ch[:200]}...\n")

    summary = ""
    if not retrieved_chunks:
        summary = "Summary not available (no relevant texts retrieved)."
    else:
        try:
            summary = generate_summary_with_rag(retrieved_chunks)
        except Exception as e:
            print(f"Error generating summary with LLM: {e}")
            traceback.print_exc()
            summary = "Error during summary generation."

    # Mostrar resultats a la consola
    print("\n================================")
    print("=== INFORME CLÍNIC GENERAT ===")
    print("================================")
    print("\n=== DADES IDENTIFICATIVES ===")
    print(structured_data)
    print("\n=== LÍNIA TEMPORAL D'EPISODIS ===")
    print(episode_timeline)
    print("\n=== RESUM CLÍNIC ESTRUCTURAT ===")
    print(summary)

     # Save output to file
    output_filename = f"clinical_report_{patient_id}.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("DADES IDENTIFICATIVES\n" + structured_data + "\n\n")
            f.write("LÍNIA TEMPORAL D'EPISODIS\n" + episode_timeline + "\n\n")
            f.write( summary + "\n\n")
            if retrieved_chunks:
                f.write("\n\n--- CHUNKS UTILITZATS PER GENERAR EL RESUM ---\n")
                for i, chunk in enumerate(retrieved_chunks):
                    f.write(f"\nCHUNK {i+1}:\n{chunk}\n")
    except Exception as e:
        print(f"Error saving the report: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()