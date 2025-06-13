import pandas as pd

def build_patient_texts(pacientes_df, episodios_df, movimientos_df, diagnosticos_df, textos_df):
    """
    Returns a dictionary with patient IDs as keys and concatenated clinical texts as values.
    """
    from datetime import datetime

    # Auxiliar function
    def safe_str(val):
        if pd.isna(val):
            return ""
        return str(val)

    patient_texts = {}

    # Get unique patient IDs from the clinical texts
    unique_patients = textos_df['id_paciente'].unique()

    for id_paciente in unique_patients:
        # Demographic information
        paciente_info = pacientes_df[pacientes_df['id_paciente'] == id_paciente]
        if not paciente_info.empty:
            paciente = paciente_info.iloc[0]
            # Calculate age
            try:
                fecha_nac = pd.to_datetime(paciente['fecha_nacimiento'])
                edad = (datetime.now() - fecha_nac).days // 365
            except:
                edad = "No disponible"

            info_basica = (
                f"Informació demogràfica: "
                f"Edat {edad} años, "
                f"Sexe {safe_str(paciente['sexo'])}, "
                f"Nacionalitat {safe_str(paciente['nacionalidad'])}. "
            )
        else:
            info_basica = ""

        # Diagnostics
        diags_df = diagnosticos_df[diagnosticos_df['id_episodio'].isin(
            episodios_df[episodios_df['id_paciente'] == id_paciente]['id_episodio']
        )]
        diags_principales = [safe_str(d) for d in diags_df[diags_df['indica_diag_principal'] == "Si"]['diagnostico'].unique()]
        diags_principales_str = "Diagnostics principals: " + ", ".join(diags_principales) + ". " if diags_principales else ""
        motivos = [safe_str(d) for d in diags_df[diags_df['indica_motivo_consulta'] == "Si"]['diagnostico'].unique()]
        motivos_str = "Motius de consulta: " + ", ".join(motivos) + ". " if motivos else ""

        # Other diagnostics
        otros_diags = [safe_str(d) for d in diags_df[
            (diags_df['indica_diag_principal'] != "Si") &
            (diags_df['indica_motivo_consulta'] != "Si")
        ]['diagnostico'].unique()]
        otros_diags_str = "Altres diagnostics: " + ", ".join(otros_diags) + ". " if otros_diags else ""

        # Episodes
        episodios_paciente = episodios_df[episodios_df['id_paciente'] == id_paciente]

        # Episode types
        tipos_episodio = [safe_str(t) for t in episodios_paciente['tipo_episodio'].unique()]
        tipos_str = "Tipos de episodio: " + ", ".join(tipos_episodio) + ". " if tipos_episodio else ""

        # Movements and services
        movs_paciente = movimientos_df[
            movimientos_df['id_episodio'].isin(episodios_paciente['id_episodio'])
        ]
        servicios = [safe_str(s) for s in movs_paciente['servicio_medico'].unique()]
        unidades = [safe_str(u) for u in movs_paciente['unidad_tratamiento'].unique()]

        servicios_str = "Serveis mèdics: " + ", ".join(servicios) + ". " if servicios else ""
        unidades_str = "Unitats de tractament: " + ", ".join(unidades) + ". " if unidades else ""

        # Clinical texts
        textos_paciente = textos_df[textos_df['id_paciente'] == id_paciente]
        textos_str = " ".join(safe_str(t) for t in textos_paciente['texto_clinico'] if pd.notnull(t))
        textos_str = "Textos clínics: " + textos_str if textos_str.strip() else ""

        # Concatenate all information
        texto_completo = (
            f"{info_basica} "
            f"{diags_principales_str} "
            f"{motivos_str} "
            f"{otros_diags_str} "
            f"{tipos_str} "
            f"{servicios_str} "
            f"{unidades_str} "
            f"{textos_str}"
        ).strip()

        # Save the text
        if texto_completo:
            patient_texts[str(id_paciente)] = texto_completo

    return patient_texts