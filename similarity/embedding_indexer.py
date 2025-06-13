import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle

class EmbeddingIndexer:
    def __init__(self, model_name = "xlm-roberta-base", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return emb

    def build_embeddings(self, patient_texts):
        """
        patient_texts: dict {id_paciente: text}
        Returns: dict {id_paciente: embedding (np.array)}
        """
        embeddings = {}
        for pid, text in patient_texts.items():
            if text.strip():
                embeddings[pid] = self.get_embedding(text)
        return embeddings

    def save_embeddings(self, embeddings, filename="patient_embeddings.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, filename="patient_embeddings.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)