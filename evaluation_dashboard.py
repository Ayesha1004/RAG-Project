"""Streamlit RAGAS dashboard (RAGAS collections API, 0.4.x).

Run from project root: python -m streamlit run evaluation_dashboard.py --server.port 8502
"""
import json
import os
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

from src.hybrid_retrieval import hybrid_search
from src.llm_generation import generate_answer
from src.reranker import rerank_passages

load_dotenv()
st.set_page_config(page_title="RAGAS Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("RAGAS Evaluation Dashboard")
st.caption("Groq RAG generation + Groq RAGAS evaluation | Lecture and research paper contexts")

DATASET_PATH = Path("Evaluation/test_questions.csv")
ANSWERS_PATH = Path("Evaluation/generated_answers.json")
RESULTS_PATH = Path("Evaluation/ragas_results.csv")
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


@st.cache_resource
def initialize_evaluators():
    api_key = os.getenv("GROQ_EVAL_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing from .env")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=120.0,
        max_retries=2,
    )
    llm = llm_factory(
        "openai/gpt-oss-20b",
        provider="openai",
        client=client,
        temperature=0,
        max_tokens=4096,
    )
    # Local embedding model: no extra embedding API costs.
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
    }


def run_rag(question):
    candidates = hybrid_search(question, top_k=15, candidate_k=30)
    passages = rerank_passages(question, candidates, top_k=5)
    if not passages:
        return "No relevant passages found.", []
    answer = generate_answer(question, passages)
    return answer, [p["text"] for p in passages]


def generate_records(questions):
    records = []
    progress = st.progress(0)
    for i, question in enumerate(questions, 1):
        try:
            answer, contexts = run_rag(question)
            record = {"question": question, "generated_answer": answer,
                      "retrieved_contexts": contexts, "generation_error": ""}
        except Exception as exc:
            record = {"question": question, "generated_answer": "",
                      "retrieved_contexts": [], "generation_error": repr(exc)}
            st.error(f"Generation failed for {question}: {exc!r}")
            traceback.print_exc()
        records.append(record)
        # Persist partial progress in case an API request fails midway.
        ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANSWERS_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        progress.progress(i / len(questions))
    return records


def evaluate_saved(records, evaluators):
    results = []
    progress = st.progress(0)
    for i, record in enumerate(records, 1):
        question = record["question"]
        answer = record.get("generated_answer", "")
        contexts = record.get("retrieved_contexts", [])
        # Older saved data might contain joined text rather than a list.
        if isinstance(contexts, str):
            contexts = [part for part in contexts.split("\n\n---\n\n") if part]
        result = {"question": question, "generated_answer": answer,
                  "retrieved_contexts": "\n\n---\n\n".join(contexts),
                  **{metric: None for metric in METRICS}, "error": ""}
        errors = []
        if record.get("generation_error"):
            errors.append("generation: " + record["generation_error"])
        elif not contexts or not answer.strip():
            errors.append("Missing answer or retrieved contexts")
        else:
            metric_inputs = {
                "faithfulness": {"user_input": question, "response": answer,
                                 "retrieved_contexts": contexts},
                "answer_relevancy": {"user_input": question, "response": answer},
                "context_precision": {"user_input": question, "response": answer,
                                      "retrieved_contexts": contexts},
            }
            for metric, scorer in evaluators.items():
                try:
                    result[metric] = scorer.score(**metric_inputs[metric]).value
                except Exception as exc:
                    errors.append(f"{metric}: {exc!r}")
                    traceback.print_exc()
        result["error"] = " | ".join(errors)
        results.append(result)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
        progress.progress(i / len(records))
    return pd.DataFrame(results)


st.subheader("1. Evaluation Dataset")
if not DATASET_PATH.exists():
    st.error(f"Create {DATASET_PATH} with a 'question' column first.")
    st.stop()
try:
    dataset = pd.read_csv(DATASET_PATH, dtype=str).fillna("")
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()
if "question" not in dataset:
    st.error("The CSV needs a 'question' column. Reference answers are optional.")
    st.stop()
questions = dataset.loc[dataset["question"].str.strip() != "", "question"].tolist()
st.write(f"Questions: {len(questions)}")
st.dataframe(dataset, use_container_width=True)

st.subheader("2. Generate / Evaluate")
st.caption("Generate Answers saves full answers and individual contexts. Evaluate Saved Answers reuses those records without another Groq answer-generation call, but the Groq judge still makes API calls.")
col_generate, col_score = st.columns(2)
with col_generate:
    generate_clicked = st.button("Generate & Save Answers", disabled=not questions, type="primary")
with col_score:
    evaluate_clicked = st.button("Evaluate Saved Answers", disabled=not ANSWERS_PATH.exists())

if generate_clicked:
    generated = generate_records(questions)
    st.success(f"Saved {len(generated)} records to {ANSWERS_PATH}.")
    st.info("Click 'Evaluate Saved Answers' to score them.")
if evaluate_clicked:
    try:
        records = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
        if not records:
            st.warning("No saved answers found.")
        else:
            evaluators = initialize_evaluators()
            results_df = evaluate_saved(records, evaluators)
            st.session_state["evaluation_results"] = results_df
            st.success(f"Finished scoring {len(records)} saved answers. Results saved to {RESULTS_PATH}.")
    except Exception as exc:
        st.error(f"Could not initialize or run evaluation: {exc!r}")
        traceback.print_exc()

st.subheader("3. Evaluation Results")
if "evaluation_results" in st.session_state:
    results_df = st.session_state["evaluation_results"].copy()
elif RESULTS_PATH.exists():
    results_df = pd.read_csv(RESULTS_PATH).fillna("")
else:
    results_df = None

if results_df is not None and not results_df.empty:
    for metric in METRICS:
        if metric not in results_df.columns:
            results_df[metric] = None
        results_df[metric] = pd.to_numeric(results_df[metric], errors="coerce")
    if "error" not in results_df:
        results_df["error"] = ""

    columns = st.columns(3)
    for col, metric in zip(columns, METRICS):
        scored = int(results_df[metric].notna().sum())
        average = results_df[metric].mean()
        col.metric(metric.replace("_", " ").title(),
                   f"{average:.3f}" if pd.notna(average) else "N/A")
        col.caption(f"Scored {scored}/{len(results_df)}")
    st.caption("N/A means no valid score. Failed scores are not counted as zero.")
    st.subheader("Metric Averages")
    st.bar_chart(results_df[METRICS].mean())
    st.subheader("Question-level Scores")
    st.dataframe(results_df[["question", *METRICS, "error"]], use_container_width=True)

    st.subheader("Full Answers and Retrieved Sources")
    for _, row in results_df.iterrows():
        with st.expander(str(row["question"])):
            st.markdown("**Generated Answer**")
            st.write(row.get("generated_answer", ""))
            st.markdown("**Retrieved Contexts**")
            st.text(str(row.get("retrieved_contexts", "")))
            for metric in METRICS:
                if pd.notna(row[metric]):
                    st.write(f"{metric.replace('_', ' ').title()}: {row[metric]:.3f}")
            if pd.notna(row.get("error")) and str(row.get("error", "")).strip():
                st.error(str(row["error"]))
    st.download_button("Download Evaluation Results", results_df.to_csv(index=False),
                       file_name="ragas_results.csv", mime="text/csv")

