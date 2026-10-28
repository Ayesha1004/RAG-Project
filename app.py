
import streamlit as st
from sentence_transformers import SentenceTransformer
from src.vector_store import store_embeddings
import chromadb

from src.ingest import load_documents
from src.chunk import chunk_documents
from src.hybrid_retrieval import hybrid_search
from src.reranker import rerank_passages
from src.llm_generation import generate_answer

st.set_page_config(
    page_title="Lecture RAG System",
    layout="wide"
)

st.title("NLP Lecture RAG Question Answering System")
st.write("Ask questions based on your university lecture PDFs.")

with st.expander("View RAG Pipeline"):
    st.markdown("""
    1. **Document extraction:** Extract text from lecture PDFs.
    2. **Chunking:** Split extracted text into overlapping chunks.
    3. **Embedding generation:** Convert chunks into vector representations.
    4. **Dense retrieval:** Find semantically similar chunks.
    5. **BM25 retrieval:** Find chunks using keyword relevance.
    6. **Hybrid retrieval:** Combine dense and BM25 rankings.
    7. **Reranking:** Reorder passages by query relevance.
    8. **LLM generation:** Generate an answer grounded in retrieved passages.
    """)


if st.button("Build / Refresh Vector Index"):
    with st.spinner("Extracting, chunking, and generating embeddings..."):
        documents = load_documents()
        chunks = chunk_documents(documents)

        if not chunks:
            st.warning("No readable text was found in the PDF documents.")
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(
                [chunk["text"] for chunk in chunks],
                show_progress_bar=False
            )

            client = chromadb.PersistentClient(path="chroma_db")

            try:
                client.delete_collection(name="lecture_chunks")
            except Exception:
                pass

            store_embeddings(chunks, embeddings)

            st.session_state["documents"] = documents
            st.session_state["chunks"] = chunks

            st.success("Vector index created successfully!")

            st.subheader("Document Extraction")
            st.write(f"Pages extracted: {len(documents)}")

            st.subheader("Chunking")
            st.write(f"Chunks created: {len(chunks)}")

            st.subheader("Embedding Generation")
            st.write(f"Embedding model: all-MiniLM-L6-v2")
            st.write(f"Embedding dimensions: {embeddings.shape[1]}")
            st.write(f"Embeddings generated: {len(embeddings)}")

            with st.expander("View extracted text sample"):
                st.write(documents[0]["text"][:1000])

            with st.expander("View chunk sample"):
                st.write(chunks[0]["text"])


query = st.text_input(
    "Ask a question",
    placeholder="What is softmax?"
)

if st.button("Get Answer", key="get_answer") and query.strip():
    with st.spinner("Retrieving passages and generating answer..."):
        try:
            passages = hybrid_search(
                query,
                top_k=10,
                candidate_k=30
            )

            ranked_passages = rerank_passages(
                query,
                passages,
                top_k=5
            )

            if not ranked_passages:
                st.warning("No relevant passages were found.")
            else:
                answer = generate_answer(query, ranked_passages)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Retrieved Sources")
                for i, passage in enumerate(ranked_passages, start=1):
                    with st.expander(
                        f"{i}. {passage['source']} — Page {passage['page']}"
                    ):
                        st.write(passage["text"])
                        if "rerank_score" in passage:
                            st.caption(
                                f"Reranking score: "
                                f"{passage['rerank_score']:.4f}"
                            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
