
from src.dense_retrieval import dense_search

query = input("Enter a question about your lectures: ")

results = dense_search(query)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Text:", result["text"][:500])
