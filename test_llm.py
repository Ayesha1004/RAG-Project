from src.hybrid_retrieval import hybrid_search
from src.reranker import rerank_passages
from src.llm_generation import generate_answer

query = input("Enter your lecture question: ")

passages = hybrid_search(query, top_k=10, candidate_k=20)
passages = rerank_passages(query, passages, top_k=3)

answer = generate_answer(query, passages)

print("\nAnswer:\n")
print(answer)