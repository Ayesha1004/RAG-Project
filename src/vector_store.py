
import chromadb


def store_embeddings(chunks, embeddings):
    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name="lecture_chunks"
    )

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": chunk["source"], "page": chunk["page"]}
            for chunk in chunks
        ]
    )

    return collection
