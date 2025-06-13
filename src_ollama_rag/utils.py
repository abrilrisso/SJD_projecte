# utils.py
import pandas as pd
import os

DATA_FOLDER = "dades/dades_preprocessades"

def load_datasets():
    """
    Load datasets from CSV files and return them as pandas DataFrames.
    """
    pacientes = pd.read_csv(os.path.join(DATA_FOLDER, "Pacientes.csv"), dtype={"id_paciente": str})
    episodios = pd.read_csv(os.path.join(DATA_FOLDER, "Episodios.csv"), dtype={"id_paciente": str, "id_episodio": str})
    movimientos = pd.read_csv(os.path.join(DATA_FOLDER, "Movimientos.csv"), dtype={"id_episodio": str, "id_movimiento": str})
    diagnosticos = pd.read_csv(os.path.join(DATA_FOLDER, "Diagnosticos.csv"), dtype={"id_episodio": str, "movimiento_asociado": str})
    textos = pd.read_csv(os.path.join(DATA_FOLDER, "Textos.csv"), dtype={"id_paciente": str, "id_episodio": str})
    return pacientes, episodios, movimientos, diagnosticos, textos

def build_clinical_record(id_paciente, pacientes, episodios, movimientos, diagnosticos, textos):
    """
    Build a clinical record for a given patient ID by aggregating information from various datasets.
    """
    record = {}

    patient_info = pacientes[pacientes['id_paciente'] == id_paciente]
    record['patient_info'] = patient_info.iloc[0].fillna("").to_dict() if not patient_info.empty else {}

    patient_episodes = episodios[episodios['id_paciente'] == id_paciente]
    episodes_list = []
    for _, episode in patient_episodes.iterrows():
        id_episodio = episode['id_episodio']
        episode_info = episode.fillna("").to_dict()

        episode_movements = movimientos[movimientos['id_episodio'] == id_episodio]
        episode_info['movements'] = episode_movements.fillna("").to_dict(orient='records')

        episode_diagnosticos = diagnosticos[diagnosticos['id_episodio'] == id_episodio]
        episode_info['diagnostics'] = episode_diagnosticos.fillna("").to_dict(orient='records')

        episode_texts = textos[textos['id_episodio'] == id_episodio]
        episode_info['texts'] = episode_texts.fillna("").to_dict(orient='records')

        episodes_list.append(episode_info)

    record['clinical_episodes'] = episodes_list
    return record

def extract_free_texts(clinical_record):
    """
    Extract free text notes from the clinical record.
    """
    texts = []
    for episode in clinical_record['clinical_episodes']:
        for note in episode.get('texts', []):
            if note.get('texto_clinico'):
                texts.append(note['texto_clinico'])
    return "\n".join(texts)
