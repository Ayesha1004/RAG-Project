
from src.dense_retrieval import dense_search
from src.bm25_retrieval import build_bm25_index, bm25_search

documents, bm25_index = build_bm25_index()


def hybrid_search(query, top_k=5, candidate_k=10):
    dense_results = dense_search(query, top_k=candidate_k)

    bm25_results = bm25_search(
        query, documents, bm25_index, top_k=candidate_k
    )

    # Combine rankings using Reciprocal Rank Fusion (RRF)
    scores = {}
    passages = {}

    for results in [dense_results, bm25_results]:
        for rank, result in enumerate(results, start=1):
            key = (result["source"], result["page"], result["text"])
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            passages[key] = result

    ranked = sorted(scores, key=scores.get, reverse=True)

    return [
        {**passages[key], "hybrid_score": scores[key]}
        for key in ranked[:top_k]
    ]
