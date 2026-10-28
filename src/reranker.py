
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_passages(query, passages, top_k=3):
    if not passages:
        return []

    pairs = [
        (query, passage["text"])
        for passage in passages
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(passages, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        {**passage, "rerank_score": float(score)}
        for passage, score in ranked[:top_k]
    ]
