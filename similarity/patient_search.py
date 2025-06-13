from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar_patient(query_id, patient_embeddings):
    """
    Find the most similar patient to the given query_id based on cosine similarity of embeddings.
    """
    query_emb = patient_embeddings[query_id].reshape(1, -1)
    best_id = None
    best_score = -1

    for pid, emb in patient_embeddings.items():
        if pid == query_id:
            continue
        score = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]
        if score > best_score:
            best_score = score
            best_id = pid

    return best_id, best_score