
from rank_bm25 import BM25Okapi
from src.ingest import load_documents


def build_bm25_index():
    documents = load_documents()

    tokenized_docs = [
        doc["text"].lower().split()
        for doc in documents
    ]

    index = BM25Okapi(tokenized_docs)
    return documents, index


def bm25_search(query, documents, index, top_k=5):
    tokens = query.lower().split()
    scores = index.get_scores(tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for i in ranked_indices[:top_k]:
        if scores[i] > 0:
            results.append({
                **documents[i],
                "score": float(scores[i])
            })

    return results
