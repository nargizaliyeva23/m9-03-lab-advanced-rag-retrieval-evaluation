import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# -----------------------
# LOAD DATA
# -----------------------
with open("knowledge_base.json", "r") as f:
    kb = json.load(f)

texts = [x["text"] for x in kb]

# -----------------------
# EMBEDDINGS
# -----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)

# -----------------------
# BM25 SETUP
# -----------------------
tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

# -----------------------
# COSINE SIMILARITY
# -----------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# -----------------------
# BASELINE RETRIEVAL
# -----------------------
def retrieve_dense(query):
    q_vec = model.encode([query])[0]

    scores = []
    for v in vectors:
        scores.append(cosine(q_vec, v))

    top_idx = np.argsort(scores)[::-1][:3]
    return [kb[i]["id"] for i in top_idx]

# -----------------------
# HYBRID RETRIEVAL
# -----------------------
def retrieve_hybrid(query):
    q_vec = model.encode([query])[0]

    dense_scores = [cosine(q_vec, v) for v in vectors]
    bm25_scores = bm25.get_scores(query.lower().split())

    final = []
    for i in range(len(kb)):
        score = 0.6 * dense_scores[i] + 0.4 * bm25_scores[i]
        final.append(score)

    top_idx = np.argsort(final)[::-1][:3]
    return [kb[i]["id"] for i in top_idx]

# -----------------------
# EVALUATION SET
# -----------------------
eval_set = [
    {"q": "Where can I park after 6pm?", "expected": "kb-01"},
    {"q": "error 0x80070005 login", "expected": "kb-08"},
    {"q": "how do I reset my password?", "expected": "kb-07"},
    {"q": "how to cancel subscription?", "expected": "kb-05"},
    {"q": "refund policy", "expected": "kb-04"}
]

# -----------------------
# HIT RATE FUNCTION
# -----------------------
def hit_rate(func):
    correct = 0

    for item in eval_set:
        retrieved = func(item["q"])

        if item["expected"] in retrieved:
            correct += 1

    return correct / len(eval_set)

# -----------------------
# RUN COMPARISON
# -----------------------
baseline = hit_rate(retrieve_dense)
hybrid = hit_rate(retrieve_hybrid)

print("\n=== RAG EVALUATION RESULTS ===")
print("Baseline Hit Rate:", baseline)
print("Hybrid Hit Rate:", hybrid)

# -----------------------
# SIMPLE CONCLUSION
# -----------------------
if hybrid > baseline:
    print("\nHybrid search improved retrieval performance.")
else:
    print("\nNo improvement or degraded performance.")