# generate_narrative.py
from src_ollama_rag.ollama_runner import run_ollama #src_ollama_rag

def clean_ollama_output(summary: str, section_header: str) -> str:
    """
    Cleans the output from the Ollama model by extracting the relevant section.
    Args:
        summary (str): The output summary from the model.
        section_header (str): The header of the section to extract.
    Returns:
        str: The cleaned summary.
    """
    if section_header in summary:
        return summary.split(section_header, 1)[-1].strip()
    elif ":" in summary:
        return summary.split(":", 1)[-1].strip()
    else:
        return summary.strip()

def generate_summary_with_rag(retrieved_chunks: list) -> str:
    """
    Generates a structured clinical summary based on the retrieved chunks of clinical notes.
    Args:
        retrieved_chunks (list): A list of retrieved clinical notes.
    Returns:
        str: The generated clinical summary.
    """
    if not retrieved_chunks:
        return "No relevant information was found in the clinical notes to generate a summary."

    context_text = "\n\n---\n\n".join(retrieved_chunks)
    #print(f"Context text: {context_text}") 
    prompt = f"""
    Ets un expert en medicina clínica. Basant-te EXCLUSIVAMENT en les següents notes clíniques recuperades, genera un resum estructurat i clar en català per incloure en un historial clínic.

    ### Instruccions generals:
    - No inventis informació ni afegeixis cap dada que no es trobi explícitament a les notes.
    - Escriu amb un estil mèdic professional però entenedor.
    - Utilitza frases concises i informatives. Evita repeticions.
    - El resum complet ha de tenir un màxim de 350 paraules.
    - Si alguna secció no es pot completar amb la informació disponible, indica-ho explícitament amb una frase clara.

    ### Seccions obligatòries del resum:

    1. **RESUM GENERAL DEL CURS CLÍNIC:**  
    Breu explicació del motiu de consulta, les actuacions clíniques més destacades i la situació actual o final del pacient. Destaca la línia temporal si és rellevant.

    2. **ANTECEDENTS MÈDICS RELLEVANTS:**  
    Resumeix els diagnòstics o condicions prèvies (amb data de finalització si està disponible) extrets dels episodis tancats. No especulis.

    3. **ANTECEDENTS FAMILIARS:**  
    Informa sobre malalties familiars mencionades (al·lèrgies, malalties hereditàries, etc.). Si no n’hi ha, escriu: "No hi ha informació sobre antecedents familiars a les notes proporcionades."

    4. **PROBLEMES MÈDICS ACTIUS:**  
    Enumera i **explica breument** cadascun dels problemes mèdics actuals del pacient. Per a cada problema, descriu-ne els símptomes, l’evolució recent o el tractament actiu, segons aparegui a les notes. Ex.: "El pacient presenta al·lèrgia als àcars, amb simptomatologia persistent que requereix antihistamínics diaris i immunoteràpia específica."

    5. **RECOMANACIONS:**  
    Proposa un pla clínic basat en la informació disponible. Utilitza el teu coneixement mèdic per suggerir els següents passos més raonables, encara que no estiguin explicitats a les notes."
    ---

    Notes clíniques recuperades:
    ---
    {context_text}
    ---

    Resum clínic estructurat:
    """
    output = run_ollama(prompt)
    return clean_ollama_output(output, "Resum clínic estructurat:")


