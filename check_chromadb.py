import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection("lecture_chunks")

print("Total indexed chunks:", collection.count())

sample = collection.peek(limit=5)

print("Sample metadata:", sample.get("metadatas"))