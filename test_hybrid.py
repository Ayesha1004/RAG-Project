
from src.hybrid_retrieval import hybrid_search

query = input("Enter a question about your lectures: ")
results = hybrid_search(query)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Hybrid score:", result["hybrid_score"])
    print("Text:", result["text"][:500])
