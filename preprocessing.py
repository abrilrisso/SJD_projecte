'''
UNIFICACIÓ I PREPROCESSAMENT DE TAULES

'''

import pandas as pd
from datetime import datetime
import os
import unicodedata

# AUXILIAR FUNCTIONS

# Function to normalize column names
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    def clean_name(name):
        # Remove accents
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode()
        # Convert to lowercase
        name = name.lower()
        # Change spaces to underscores
        name = name.replace(' ', '_')
        return name
    df.columns = [clean_name(col) for col in df.columns]
    return df


# Function to preprocess the Pacientes table
def preprocess_pacientes(df):
    
    # Rename columns manually
    df = df.rename(columns={
            'paciente_id': 'id_paciente',
            'pais_nac': 'pais_nacimiento'
        })

    # Convert id_paciente to string
    df['id_paciente'] = df['id_paciente'].astype('Int64').astype(str)
    
    # Checl if dates are not future dates
    today = pd.Timestamp(datetime.today().date())
    df.loc[df['fecha_nacimiento'] > today, 'fecha_nacimiento'] = pd.NaT
    df.loc[df['fecha_fallecimiento'] > today, 'fecha_fallecimiento'] = pd.NaT

    # Check if fecha_nacimiento is before fecha_fallecimiento
    mask = (df['fecha_fallecimiento'].notna()) & (df['fecha_nacimiento'].notna())
    df.loc[mask & (df['fecha_fallecimiento'] < df['fecha_nacimiento']), 'fecha_fallecimiento'] = pd.NaT

    # Change 1/2 to Home/Dona
    df['sexo'] = df['sexo'].replace({1: 'Home', 2: 'Dona'})     

    # Remove column pac_fallecido
    if 'pac_fallecido' in df.columns:
        df = df.drop(columns=['pac_fallecido'])    

    return df


# Function to preprocess the Episodios table
def preprocess_episodios(df):

    # Rename columns manually
    df = df.rename(columns={
            'episodio': 'id_episodio',
            'paciente_id': 'id_paciente',
            'inicio_episodio': 'fecha_inicio_episodio',
            'fin_episodio': 'fecha_fin_episodio'
        })
    
    # Convert ids to string
    df['id_episodio'] = df['id_episodio'].astype('Int64').astype(str)
    df['id_paciente'] = df['id_paciente'].astype('Int64').astype(str)

    # Convert dates to datetime
    df['fecha_inicio_episodio'] = pd.to_datetime(df['fecha_inicio_episodio'], errors='coerce')
    df['fecha_fin_episodio'] = pd.to_datetime(df['fecha_fin_episodio'], errors='coerce')

    # Check if dates are not future dates
    today = pd.Timestamp(datetime.today().date())
    df.loc[df['fecha_inicio_episodio'] > today, 'fecha_inicio_episodio'] = pd.NaT
    df.loc[df['fecha_fin_episodio'] > today, 'fecha_fin_episodio'] = pd.NaT

    # Check if fecha_inicio_episodio is before fecha_fin_episodio
    mask = (df['fecha_inicio_episodio'].notna()) & (df['fecha_fin_episodio'].notna())
    df.loc[mask & (df['fecha_inicio_episodio'] > df['fecha_fin_episodio']), 'fecha_fin_episodio'] = pd.NaT

    # Convert tipo_episodio code to description
    df_tipos = pd.read_excel('dades/diccionaris/Tipos Episodio.xlsx')
    tipo_episodio_dict = dict(zip(df_tipos['Tipo_Episodio'], df_tipos['Tipo_Episodio_Desc']))
    df['tipo_episodio'] = df['tipo_episodio'].str.upper().str.strip().replace(tipo_episodio_dict)

    return df


# Function to preprocess the Movimientos table
def preprocess_movimientos(df):

    # Rename columns manually
    df = df.rename(columns={
            'episodio': 'id_episodio',
            'fecha_mov': 'fecha_movimiento',
            'hora_mov': 'hora_movimiento',
            'tipo_mov_clase_mov': 'clase_tipo_movimiento'
        })
    
    # Convert id_episodio to string
    df['id_episodio'] = df['id_episodio'].astype('Int64').astype(str)
    
    # Unify date ans time in one variable
    df['fecha_movimiento'] = pd.to_datetime(df['fecha_movimiento'].dropna().astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
    df['hora_movimiento'] = pd.to_datetime(df['hora_movimiento'].dropna().astype('Int64').astype(str), format='%H%M%S', errors='coerce').dt.time
    df['fecha_hora_movimiento'] = pd.to_datetime(
        df['fecha_movimiento'].astype(str) + ' ' +
        df['hora_movimiento'].astype(str),
        errors='coerce'
    )
    df = df.drop(['fecha_movimiento', 'hora_movimiento'], axis=1)

    # Cehck if dates are not future dates
    today = pd.Timestamp(datetime.now())
    df.loc[df['fecha_hora_movimiento'] > today, 'fecha_hora_movimiento'] = pd.NaT

    # Convert variable codes to description
    df_unidades = pd.read_excel('dades/diccionaris/Unidad Tratamiento.xlsx')
    df_servicios = pd.read_excel('dades/diccionaris/Servicios Médicos.xlsx')
    df_movimientos = pd.read_excel('dades/diccionaris/Clases Movimiento.xlsm')
    unidades_dict = dict(zip(df_unidades['Unidad_Tratamiento'], df_unidades['Unidad_Tratamiento_Desc']))
    servicios_dict = dict(zip(df_servicios['Servicio_Medico'], df_servicios['Servicio_Medico_Desc']))
    movimientos_dict = dict(zip(df_movimientos['Tipo_Mov_Clase_Mov'], df_movimientos['Clase_Movimiento_desc']))
    df['unidad_tratamiento'] = df['unidad_tratamiento'].str.upper().str.strip().replace(unidades_dict)
    df['servicio_medico'] = df['servicio_medico'].str.upper().str.strip().replace(servicios_dict)
    df['clase_tipo_movimiento'] = df['clase_tipo_movimiento'].str.upper().str.strip().replace(movimientos_dict)

    # Remove columns that are not needed
    df = df.drop(['clase_mov', 'tipo_movimiento'], axis=1)

    return df


# Function to preprocess the Diagnosticos table
def preprocess_diagnosticos(df):

    # Rename columns manually
    df = df.rename(columns={
            'episodio': 'id_episodio',
            'catalogo_diag_codi': 'diagnostico'
        })
    
    df['id_episodio'] = df['id_episodio'].astype('Int64').astype(str)
    df['movimiento_asociado'] = df['movimiento_asociado'].astype('Int64').astype(str)

    # Check if dates are not future dates
    df['fecha_diagnostico'] = pd.to_datetime(df['fecha_diagnostico'], errors='coerce')
    today = pd.Timestamp(datetime.today().date())
    df.loc[df['fecha_diagnostico'] > today, 'fecha_diagnostico'] = pd.NaT

    # Convert diagnostic codes to descriptions
    df_maestro1 = pd.read_excel('dades/diccionaris/Maestro de Diagnosticos 1.xlsx')
    df_maestro2 = pd.read_excel('dades/diccionaris/Maestro de Diagnosticos 2.xlsx')

    diagnosticos_dict1 = dict(zip(df_maestro1['Catalogo_Diag_Codi'], df_maestro1['Diagnostico_Descripcion']))
    diagnosticos_dict2 = dict(zip(df_maestro2['Catalogo_Diag_Codi'], df_maestro2['Diagnostico_Descripcion']))
    df['diagnostico'] = df['diagnostico'].str.upper().str.strip()
    df['diagnostico'] = df['diagnostico'].replace(diagnosticos_dict1)
    df['diagnostico'] = df['diagnostico'].replace(diagnosticos_dict2)

    # Remove columns that are not needed
    df = df.drop(['catalogo', 'diagnostico_codigo'], axis=1)

    # Convert variables Si/No (X -> Si, NaN -> No)
    variables_a_tractar = ['indica_diag_iq', 'indica_diag_principal', 'indica_diag_tratamiento', 'indica_motivo_consulta']

    for var in variables_a_tractar:
        df[var] = df[var].fillna('No')
        df[var] = df[var].replace({'X': 'Si'})

    return df

# Function to preprocess the Textos table
def preprocess_textos(df):

    # Rename columns manually
    df = df.rename(columns={
            'texto': 'texto_clinico',
            'episodio': 'id_episodio',
            'paciente_id': 'id_paciente'
        })
    
    # Convert ids to string
    df['id_episodio'] = df['id_episodio'].astype('Int64').astype(str)
    df['id_paciente'] = df['id_paciente'].astype('Int64').astype(str)

    return df




# PROCESSING THE DATA 

data_folder = "dades/dades_originals"

file_paths = {
    "Pacientes": os.path.join(data_folder, "Pacientes.xlsx"),
    "Episodios": os.path.join(data_folder, "Episodios.xlsx"),
    "Movimientos": os.path.join(data_folder, "Movimientos.xlsx"),
    "Diagnosticos": os.path.join(data_folder, "Diagnosticos.xlsx"),
    "Textos": os.path.join(data_folder, "Textos.xlsx"),
}

datasets = {}

for name, path in file_paths.items():
    df = pd.read_excel(path)

    # Normalize column names
    df = normalize_column_names(df)

    # Preprocess the data based on the table
    if name == "Pacientes":
        df = preprocess_pacientes(df)
    elif name == "Episodios":
        df = preprocess_episodios(df)
    elif name == "Movimientos":
        df = preprocess_movimientos(df)
    elif name == "Diagnosticos":
        df = preprocess_diagnosticos(df)
    elif name == "Textos":
        df = preprocess_textos(df)

    datasets[name] = df

# Save the processed datasets to CSV files
output_folder = "dades/dades_preprocessades"
os.makedirs(output_folder, exist_ok=True)

for name, df in datasets.items():
    output_path = os.path.join(output_folder, f"{name}.csv")
    df.to_csv(output_path, index=False)