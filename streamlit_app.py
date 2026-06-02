from __future__ import annotations

import textwrap

import streamlit as st

from src.researchmate.config import settings
from src.researchmate.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="ResearchMate-RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipeline = get_pipeline()

st.title("📚 ResearchMate-RAG")
st.caption("A local Retrieval-Augmented Generation system for papers, CVs, project reports, and job matching.")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Embedding model:** `{settings.embedding_model}`")
    st.write(f"**Local LLM:** `{settings.ollama_model}`")
    st.write(f"**Top-k retrieval:** `{settings.top_k}`")
    st.write(f"**Vector chunks:** `{pipeline.store.count()}`")

    if st.button("Reset vector database", type="secondary"):
        pipeline.reset()
        st.success("Vector database reset. Re-ingest your files.")
        st.rerun()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No retrieved sources.")
        return
    for i, src in enumerate(sources, start=1):
        page = f", page {src['page']}" if src.get("page") else ""
        score = src.get("similarity")
        score_text = f" | similarity: {score:.3f}" if isinstance(score, float) else ""
        with st.expander(f"Source {i}: {src.get('source')}{page}{score_text}"):
            st.write(src.get("text", ""))


tab_upload, tab_chat, tab_match, tab_examples, tab_about = st.tabs(
    ["1 Upload & Ingest", "2 Ask Documents", "3 CV ↔ Job Match", "Examples", "About"]
)

with tab_upload:
    st.subheader("Upload documents")
    st.write("Supported formats: PDF, DOCX, TXT, MD, CSV")
    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        ingest_clicked = st.button("Ingest files", type="primary", disabled=not uploaded_files)

    if ingest_clicked and uploaded_files:
        with st.spinner("Parsing, chunking, embedding, and storing documents..."):
            try:
                result = pipeline.ingest_uploaded_files(uploaded_files)
                st.success("Ingestion complete.")
                st.json(result)
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

with tab_chat:
    st.subheader("Ask questions grounded in your documents")
    question = st.text_area(
        "Question",
        placeholder="Example: What are the main contributions of this paper?",
        height=90,
    )
    top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=10, value=settings.top_k)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving sources and generating answer..."):
            result = pipeline.ask(question.strip(), top_k=top_k)
        if result["ok"]:
            st.markdown("### Answer")
            st.write(result["answer"])
        else:
            st.warning(result["answer"])
        st.markdown("### Retrieved sources")
        render_sources(result["sources"])

with tab_match:
    st.subheader("Match your CV/research profile to a job description")
    st.write("First upload your CV, research statement, or project documents in the Upload tab.")
    jd = st.text_area(
        "Paste job/PhD description",
        placeholder="Paste the full job description here...",
        height=220,
    )
    if st.button("Analyze fit", type="primary", disabled=not jd.strip()):
        with st.spinner("Comparing documents with the job description..."):
            result = pipeline.match_cv_to_job(jd.strip())
        if result["ok"]:
            st.markdown("### Fit Analysis")
            st.write(result["answer"])
        else:
            st.warning(result["answer"])
        st.markdown("### Retrieved candidate evidence")
        render_sources(result["sources"])

with tab_examples:
    st.subheader("Good demo questions for your portfolio")
    examples = [
        "Summarize the main contribution of the uploaded paper in 5 bullet points.",
        "Extract the dataset, models, metrics, and limitations from this project report.",
        "What evidence supports the methodology section? Cite sources.",
        "Create 10 viva/interview questions from this project report.",
        "Compare the uploaded CV with the job description and suggest improvements.",
        "Find risks, missing details, or weak claims in this report.",
    ]
    for q in examples:
        st.code(q)

with tab_about:
    st.subheader("Project purpose")
    st.write(
        textwrap.dedent(
            """
            ResearchMate-RAG is designed as a portfolio-ready local RAG system. It demonstrates document parsing,
            chunking, embedding generation, vector search, source-grounded prompting, local LLM usage, and a clean
            user interface. It is especially suitable for academic papers, CVs, project reports, PhD applications,
            and research document analysis.
            """
        )
    )
    st.markdown(
        """
        **Recommended local model on Mac Mini M4:**
        ```bash
        ollama pull llama3.2:3b
        ```
        For stronger but slower answers, try:
        ```bash
        ollama pull llama3.1:8b
        ```
        """
    )
