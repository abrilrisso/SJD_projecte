# app.py
import pandas as pd
import re
import streamlit as st

from src_ollama_rag.pipeline import run_pipeline as executa_pipeline
from reportlab.lib.utils import simpleSplit
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from similarity.embedding_indexer import EmbeddingIndexer
from similarity.patient_text_builder import build_patient_texts
from similarity.patient_search import find_most_similar_patient

# --- Constants for PDF layout ---
LEFT_MARGIN = 40
TOP_START_Y = 750
BOTTOM_MARGIN = 100
LINE_HEIGHT = 20
MAX_LINE_WIDTH = 500

# Regular expressions to identify report sections
_HEADINGS = [
    ("dades_identificatives", r"DADES IDENTIFICATIVES"),
    ("linia_temporal",        r"LÍNIA TEMPORAL D['’]EPISODIS"),
    ("resum_general",         r"RESUM CL[ÍI]NIC(?: ESTRUCTURAT)?"),
]

_HEADINGS_RGX = [(name, re.compile(rf"^{pat}", re.MULTILINE)) for name, pat in _HEADINGS]
_bullet_fix = re.compile(r"\*\s*\n")


def extract_sections(text: str) -> dict[str, str]:
    """Extracts structured sections from report text."""
    sections: dict[str, str] = {}
    n = len(_HEADINGS_RGX)

    for i, (name, start_regex) in enumerate(_HEADINGS_RGX):
        match_start = start_regex.search(text)
        if not match_start:
            sections[name] = ""
            continue

        start = match_start.end()
        if i + 1 < n:
            remaining_patterns = "|".join(r.pattern[1:] for _, r in _HEADINGS_RGX[i+1:])
            match_end = re.search(rf"^({remaining_patterns})", text[start:], re.MULTILINE)
            end = start + match_end.start() if match_end else len(text)
        else:
            end = len(text)

        block = text[start:end].strip()
        block = _bullet_fix.sub("* ", block)
        sections[name] = block

    return sections


def read_report(patient_id: str, base_path="output_informe_"):
    """Reads a saved text report and extracts its sections."""
    path = f"{base_path}{patient_id}.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return extract_sections(f.read())
    except FileNotFoundError:
        return None

def create_text_object(c: canvas.Canvas, font_name="Helvetica"):
    """Creates a new canvas text object."""
    t = c.beginText(LEFT_MARGIN, TOP_START_Y)
    t.setFont(font_name, 11, leading=LINE_HEIGHT)
    t.setLeading(LINE_HEIGHT)
    return t

def similar_patient(patient_id: str):
    """Finds the most similar patient using embedding similarity."""
    try:
        path = "dades/dades_preprocessades"
        pac = pd.read_csv(f"{path}/Pacientes.csv", dtype={'id_paciente': str})
        epi = pd.read_csv(f"{path}/Episodios.csv", dtype={'id_paciente': str})
        mov = pd.read_csv(f"{path}/Movimientos.csv", dtype={'id_paciente': str})
        dia = pd.read_csv(f"{path}/Diagnosticos.csv", dtype={'id_paciente': str})
        tex = pd.read_csv(f"{path}/Textos.csv", dtype={'id_paciente': str})

        texts = build_patient_texts(pac, epi, mov, dia, tex)
        if patient_id not in texts:
            return None

        indexer = EmbeddingIndexer()
        embeddings = indexer.build_embeddings(texts)
        return find_most_similar_patient(patient_id, embeddings)
    except Exception as e:
        print("Error en similaritat:", e)
        return None
    
def add_bold_text(text_obj, line: str, c: canvas.Canvas):
    """Adds a line of text with optional bold sections to the canvas."""
    parts = re.split(r"(\*\*.*?\*\*)", line)
    for part in parts:
        is_bold = part.startswith("**") and part.endswith("**")
        content = part[2:-2] if is_bold else part
        font = "Helvetica-Bold" if is_bold else "Helvetica"

        text_obj.setFont(font, 11, leading=LINE_HEIGHT)
        for segment in simpleSplit(content, font, 11, MAX_LINE_WIDTH):
            if text_obj.getY() - LINE_HEIGHT < BOTTOM_MARGIN:
                c.drawText(text_obj)
                c.showPage()
                text_obj = create_text_object(c, font)
            text_obj.textLine(segment)

    return text_obj


def generate_pdf(name, age, gender, birth_date, death_date, timeline_text, clinical_summary, patient_id, chunks: list[str] = None):
    """Generate a structured clinical report in PDF format."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawImage("logo.jpg", x=450, y=770, width=110, height=55, preserveAspectRatio=True)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(300, 740, f"HISTORIAL CLÍNIC DEL PACIENT {patient_id}")

    text = create_text_object(c)
    text.moveCursor(0, 30)

    content = [
        "**DADES IDENTIFICATIVES:**",
        f"Nom: {name}",
        f"Edat: {age}",
        f"Sexe: {gender}",
        f"Data de naixement: {birth_date}",
        f"Data de defunció (en cas de mort): {death_date}",
        "",
        "",
        "",
        "**LÍNIA TEMPORAL D’EPISODIS:**",
        timeline_text,
        "",
        "",
        "",
        "**RESUM CLÍNIC ESTRUCTURAT:**",
        clinical_summary,
    ]

    for line in content:
        for subline in line.split("\n"):
            text = add_bold_text(text, subline, c)

    c.drawText(text)

    if chunks:
        c.showPage()
        text = create_text_object(c)
        text = add_bold_text(text, "**CHUNKS UTILITZATS PER GENERAR EL RESUM:**", c)
        for chunk in chunks:
            text = add_bold_text(text, chunk, c)
        c.drawText(text)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# --- Streamlit Interface ---
st.set_page_config(page_title="Descarregar PDF historial clínic")

col1, col2, col3 = st.columns([0.5, 6, 0.5])
with col2:
    logo_col1, logo_col2 = st.columns([5, 2])
    with logo_col2:
        st.image("logo.jpg", width=340)

    st.markdown("<h2 style='text-align: left;'>Generador d'historials clínics resumits</h2>", unsafe_allow_html=True)
    st.markdown("Per generar el document, ompliu el següent:")

    with st.form("formulari_pacient", clear_on_submit=False):
        patient_id = st.text_input("Identificador del pacient", placeholder="Ex: 123456")
        submitted = st.form_submit_button("Generar informe")

    if submitted and patient_id:
        st.markdown("---")
        with st.spinner("Generant l’informe clínic amb el model..."):
            resultat_ok = executa_pipeline(patient_id)

        if not resultat_ok:
            st.error("⚠️ No s'ha trobat cap pacient amb aquest ID. Torna a indicar-ne un altre.")
            st.stop()

        similar_result = similar_patient(patient_id)
        st.success(f"Document generat per al pacient amb ID: **{patient_id}**")

        # Read sections from .txt file
        try:
            with open(f"output_informe_{patient_id}.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            st.error("No s'ha pogut trobar el fitxer d'informe generat.")
            st.stop()

        # Separate sections
        blocs = {"dades": "", "episodis": "", "resum": "", "chunks": []}
        current = None
        chunk = ""
        for line in lines:
            if line.startswith("DADES IDENTIFICATIVES"):
                current = "dades"
            elif line.startswith("LÍNIA TEMPORAL"):
                current = "episodis"
            elif line.startswith("RESUM CLÍNIC"):
                current = "resum"
            elif "--- CHUNKS UTILITZATS PER GENERAR EL RESUM ---" in line:
                current = "chunks"
            elif line.startswith("CHUNK") and current == "chunks":
                if chunk:
                    blocs["chunks"].append(chunk.strip())
                chunk = line
            elif current == "chunks":
                chunk += line
            elif current in blocs:
                blocs[current] += line

        if chunk:
            blocs["chunks"].append(chunk.strip())

        # Extract basic patient data
        nom = "No disponible"
        edat = sexe = "?"
        data_defuncio = "-"
        for linia in blocs["dades"].splitlines():
            if "Edat:" in linia:
                edat = linia.split(":", 1)[-1].strip()
            elif "Sexe:" in linia:
                sexe = linia.split(":", 1)[-1].strip()
            elif "Data de defunció:" in linia:
                data_defuncio = linia.split(":", 1)[-1].strip()
            elif "Data de naixement:" in linia:
                data_naixement = linia.split(":", 1)[-1].strip()
                if data_naixement != "No disponible":
                    try:
                        edat = str(2023 - int(data_naixement.split("-")[0]))  # Assumint any actual
                    except ValueError:
                        edat = "Desconeguda"

        # Generate and download PDF
        pdf = generate_pdf(
            nom, edat, sexe, data_naixement, data_defuncio,
            blocs["episodis"].strip(), blocs["resum"].strip(),
            patient_id, blocs["chunks"]
        )

        st.download_button(
            label="Descarregar PDF",
            data=pdf,
            file_name=f"resum_historial_{patient_id}.pdf",
            mime="application/pdf",
        )

        if similar_result is None:
            st.warning("No s'ha pogut calcular el pacient més similar.")
        else:
            best_id, best_score = similar_result
            st.success(f"Pacient més similar: **{best_id}**")