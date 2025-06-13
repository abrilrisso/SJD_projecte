# build_structured_report.py
import pandas as pd
from datetime import datetime

def calculate_age(birth_date_str):
    """
    Calculate the age of a patient based on their birth date.
    """
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except:
        return "Desconeguda"


def build_structured_info(id_paciente, pacientes, episodios):
    """
    Build structured information about a patient and their episodes.
    """
    patient_info = pacientes[pacientes['id_paciente'] == id_paciente].iloc[0]

    sexe = "Home" if str(patient_info['sexo']) == "1" else "Dona"
    edat = calculate_age(patient_info['fecha_nacimiento'])

    dades_identificatives = []
    dades_identificatives.append(f"ID pacient: {patient_info.get('id_paciente', 'No disponible')}")
    dades_identificatives.append(f"Edat: {edat}")
    dades_identificatives.append(f"Sexe: {sexe}")
    dades_identificatives.append(f"Data de naixement: {patient_info.get('fecha_nacimiento', 'No disponible')}")
    if pd.notna(patient_info.get('fecha_fallecimiento')) and patient_info.get('fecha_fallecimiento') != "":
        dades_identificatives.append(f"Data de defunció: {patient_info['fecha_fallecimiento']}")

    linia_temporal = []
    patient_episodes = episodios[episodios['id_paciente'] == id_paciente]
    patient_episodes = patient_episodes.sort_values(by='fecha_inicio_episodio')
    for _, ep in patient_episodes.iterrows():
        fecha_fin = ep['fecha_fin_episodio'] if pd.notna(ep['fecha_fin_episodio']) and ep['fecha_fin_episodio'] != "" else "en curs"
        linia_temporal.append(f"- {ep['fecha_inicio_episodio']} -> {fecha_fin} | Tipus: {ep.get('tipo_episodio', 'Desconegut')} | ID Episodi: {ep['id_episodio']}")

    return "\n".join(dades_identificatives), "\n".join(linia_temporal)
