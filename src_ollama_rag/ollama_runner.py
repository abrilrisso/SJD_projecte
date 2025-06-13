# ollama_runner.py
import ollama
import requests
import subprocess
import time
import psutil  # pip install psutil

OLLAMA_PORT = 11434

def is_ollama_running():
    """
    Checks if there is an 'ollama serve' process running or if port 11434 is open.
    Returns:
        bool: True if 'ollama serve' is running or port 11434 is open, False otherwise.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'ollama' in proc.info['name'].lower() and 'serve' in ' '.join(proc.info['cmdline']):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Also check if the port is open
    try:
        requests.get(f"http://localhost:{OLLAMA_PORT}", timeout=1)
        return True
    except:
        return False

def start_ollama_server():
    """
    Starts 'ollama serve' as a background process (Windows only).
    """
    subprocess.Popen(
        ["ollama", "serve"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(3)  

# gemma3:4b
# alibayram/medgemma:latest
def run_ollama(prompt, model="gemma3:4b", temperature=0.1):
    if not is_ollama_running():
        start_ollama_server()

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature}
    )
    return response['response'].strip()

