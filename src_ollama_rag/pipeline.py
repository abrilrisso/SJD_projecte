# pipeline.py
from src_ollama_rag.build_structured_report import build_structured_info
from src_ollama_rag.generate_narrative import generate_summary_with_rag
from src_ollama_rag.utils import load_datasets, build_clinical_record, extract_free_texts
from src_ollama_rag.ollama_runner import is_ollama_running, start_ollama_server
from src_ollama_rag.rag_processor import index_patient_texts, retrieve_relevant_chunks
from src_ollama_rag.main import split_into_chunks
import time
import traceback


def run_pipeline(patient_id: str):
    """
    Executes the full clinical summary pipeline for a given patient.

    This function performs the following steps:
    1. Starts the Ollama server if not already running.
    2. Loads all necessary datasets (patients, episodes, movements, diagnoses, clinical texts).
    3. Builds a structured clinical summary (basic info and episode timeline).
    4. Extracts and prepares free-text notes for semantic search indexing.
    5. Indexes the patient's clinical notes into the vector store.
    6. Performs retrieval of the most relevant text chunks using a RAG strategy.
    7. Generates a structured clinical narrative based on retrieved information.
    8. Saves the final report to a `.txt` file, including the chunks used.

    Args:
        patient_id (str): The unique identifier of the patient.

    Returns:
        bool | None: Returns True if the report was successfully created,
                     False if the patient ID was not found,
                     or None if another error occurred during the pipeline.
    """
    # --- Check if Ollama server is running ---
    if not is_ollama_running():
        print("Ollama server no està actiu. Intentant engegar-lo...")
        try:
            start_ollama_server()
            time.sleep(10)
            if not is_ollama_running():
                print("No s'ha pogut engegar Ollama. Cal iniciar-lo manualment.")
                return
        except Exception as e:
            print(f"Error en engegar Ollama: {e}")
            return

    # --- Load clinical data ---
    try:
        patients, episodes, movements, diagnoses, texts_df = load_datasets()
    except Exception as e:
        print(f"Error carregant datasets: {e}")
        traceback.print_exc()
        return

    if patient_id not in patients['id_paciente'].values:
        print(f"ID de pacient '{patient_id}' no trobat.")
        return False

    # --- Build structured summary and extract clinical text ---
    structured_data, episode_timeline = build_structured_info(patient_id, patients, episodes)
    clinical_record = build_clinical_record(patient_id, patients, episodes, movements, diagnoses, texts_df)
    full_text = extract_free_texts(clinical_record)

    # --- Prepare record for indexing ---
    record = {'text_entries': []}
    if texts_df is not None and not texts_df.empty:
        patient_df = texts_df[texts_df['id_paciente'] == patient_id]
        if not patient_df.empty and 'texto_nota' in patient_df.columns:
            notes = patient_df['texto_nota'].dropna().astype(str).tolist()
            if notes:
                record['text_entries'] = notes

    if not record['text_entries'] and full_text:
        chunks = split_into_chunks(full_text)
        if chunks:
            record['text_entries'] = chunks
        else:
            print("No s'han pogut generar chunks.")
            return
    elif not record['text_entries']:
        print("No hi ha textos disponibles per aquest pacient.")
        return

    # --- Index patient texts into vector database ---
    try:
        index_patient_texts(patient_id, record)
    except Exception as e:
        print(f"Error en indexació: {e}")
        traceback.print_exc()
        return

    # --- Retrieve relevant chunks and generate summary ---
    query = f"General clinical summary of patient {patient_id}, including clinical course, relevant history, and active problems."
    retrieved_chunks = retrieve_relevant_chunks(patient_id, query, n_results=7)

    if not retrieved_chunks:
        summary = "Summary not available (no relevant texts retrieved)."
    else:
        try:
            summary = generate_summary_with_rag(retrieved_chunks)
        except Exception as e:
            print(f"Error generant el resum: {e}")
            traceback.print_exc()
            summary = "Error durant la generació del resum."

    # --- Save report to file ---
    output_filename = f"output_informe_{patient_id}.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n\n\n\n" + "DADES IDENTIFICATIVES\n" + structured_data + "\n\n\n\n")
            f.write("\n\n\n\n" + "LÍNIA TEMPORAL D'EPISODIS\n" + episode_timeline + "\n\n\n\n")
            f.write("RESUM CLÍNIC ESTRUCTURAT" + summary + "\n")
            if retrieved_chunks:
                f.write("\n\n--- CHUNKS UTILITZATS PER GENERAR EL RESUM ---\n")
                for i, chunk in enumerate(retrieved_chunks):
                    f.write(f"\nCHUNK {i+1}:\n{chunk}\n")
    except Exception as e:
        print(f"Error guardant el fitxer: {e}")
        traceback.print_exc()

    print("Informe guardat amb èxit.")
    return True