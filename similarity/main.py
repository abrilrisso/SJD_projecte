import pandas as pd
from patient_text_builder import build_patient_texts, build_patient_texts
from embedding_indexer import EmbeddingIndexer
from patient_search import find_most_similar_patient

def main():

    # Load data
    pacientes_df = pd.read_csv("dades/dades_preprocessades/Pacientes.csv", dtype={'id_paciente': str})
    episodios_df = pd.read_csv("dades/dades_preprocessades/Episodios.csv", dtype={'id_paciente': str})
    movimientos_df = pd.read_csv("dades/dades_preprocessades/Movimientos.csv", dtype={'id_paciente': str})
    diagnosticos_df = pd.read_csv("dades/dades_preprocessades/Diagnosticos.csv", dtype={'id_paciente': str})
    textos_df = pd.read_csv("dades/dades_preprocessades/Textos.csv", dtype={'id_paciente': str})

    # Create patient texts
    patient_texts = build_patient_texts(pacientes_df, episodios_df, movimientos_df, diagnosticos_df, textos_df)

    # Build embeddings
    indexer = EmbeddingIndexer()
    patient_embeddings = indexer.build_embeddings(patient_texts)
    indexer.save_embeddings(patient_embeddings)
    
    # Buscar el pacient més similar
    query_id = input("Introduceix l'id del pacient a buscar: ").strip()
    query_id = str(query_id)
    if query_id not in patient_embeddings:
        print("El pacient no té textos clínics.")
        return

    best_id, best_score = find_most_similar_patient(query_id, patient_embeddings)
    print(f"\nEl paciente més similar a {query_id} es {best_id} (similitud: {best_score:.2%})")
    print("\n--- Text del pacient consultat ---")
    print(patient_texts[query_id][:2500])
    print("\n--- Text del pacient més similar ---")
    print(patient_texts[best_id][:2500])

if __name__ == "__main__":
    main()