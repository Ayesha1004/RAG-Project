
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="lecture_chunks")


def dense_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    passages = []

    for i, text in enumerate(results["documents"][0]):
        passages.append({
            "text": text,
            "source": results["metadatas"][0][i].get("source", "Unknown"),
            "page": results["metadatas"][0][i].get("page", "Unknown"),
            "distance": results["distances"][0][i]
        })

    return passages
