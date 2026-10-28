
from src.bm25_retrieval import build_bm25_index, bm25_search

documents, index = build_bm25_index()

query = input("Enter a question about your lectures: ")
results = bm25_search(query, documents, index)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Score:", result["score"])
    print("Text:", result["text"][:500])
