
from src.hybrid_retrieval import hybrid_search
from src.reranker import rerank_passages

query = input("Enter a question about your lectures: ")

candidates = hybrid_search(query, top_k=10, candidate_k=20)
results = rerank_passages(query, candidates, top_k=5)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Rerank score:", result["rerank_score"])
    print("Text:", result["text"][:500])
