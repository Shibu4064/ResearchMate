# ResearchMate-RAG

**ResearchMate-RAG** is a local, portfolio-ready Retrieval-Augmented Generation system for academic papers, CVs, project reports, thesis chapters, and job descriptions. It is designed to run on a **Mac Mini M4** using local embeddings, ChromaDB, Streamlit, FastAPI, and an Ollama local LLM.

## Why this project is strong for your portfolio

This project demonstrates practical LLM engineering beyond a basic chatbot:

- Document parsing for PDF, DOCX, TXT, MD, and CSV files
- Text chunking with overlap
- SentenceTransformer embeddings
- Persistent ChromaDB vector storage
- Source-grounded retrieval
- Local LLM generation with Ollama
- Streamlit interface
- FastAPI backend
- CV-to-job matching workflow
- Evaluation script
- Docker support

## System architecture

```text
User uploads PDF / DOCX / TXT / CSV
        ↓
Document parser extracts clean text
        ↓
Text is split into overlapping chunks
        ↓
Chunks are embedded with SentenceTransformers
        ↓
Embeddings are stored in ChromaDB
        ↓
User asks a question
        ↓
Retriever finds top-k relevant chunks
        ↓
Ollama LLM generates a grounded answer
        ↓
Answer is returned with source references
```

## Project structure

```text
researchmate-rag-mac-m4/
│
├── src/researchmate/
│   ├── config.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── llm.py
│   └── rag_pipeline.py
│
├── api/
│   └── main.py
│
├── evaluation/
│   ├── test_questions.json
│   └── evaluate_rag.py
│
├── data/
│   ├── sample_docs/
│   └── uploads/
│
├── chroma_store/
├── scripts/
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Mac Mini M4 setup

### 1. Install Ollama

Install Ollama from the official website, then pull a lightweight model:

```bash
ollama pull llama3.2:3b
```

For stronger answers, you can also try:

```bash
ollama pull llama3.1:8b
```

Keep Ollama running in the background.

### 2. Create environment and install dependencies

From the project folder:

```bash
bash scripts/setup_mac.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.example .env
```

### 3. Run the Streamlit app

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 4. Optional: run the FastAPI backend

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs:

```text
http://localhost:8000/docs
```

## How to use the app

1. Open the Streamlit app.
2. Go to **Upload & Ingest**.
3. Upload PDFs, CVs, project reports, or job descriptions.
4. Click **Ingest files**.
5. Go to **Ask Documents** and ask source-grounded questions.
6. Use **CV ↔ Job Match** to compare your CV or research profile against a PhD/job description.

## Example questions

```text
What are the main contributions of this paper?
Summarize the methodology section.
Which dataset, models, and metrics are used?
What are the limitations of this project report?
Generate 10 viva/interview questions from this document.
Compare my CV with this PhD position and suggest improvements.
Extract all research skills and publications from the uploaded CV.
```

## API examples

### Health check

```bash
curl http://localhost:8000/health
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main contribution of the uploaded document?", "top_k":5}'
```

## Evaluation

After ingesting documents, run:

```bash
python evaluation/evaluate_rag.py
```

The starter evaluation script calculates a simple keyword hit rate for test questions. For a stronger research-oriented GitHub version, you can extend it with:

- Retrieval Precision@k
- Answer groundedness
- Citation accuracy
- Hallucination rate
- Response time

## Portfolio README pitch

You can use this short description on GitHub:

> ResearchMate-RAG is a local Retrieval-Augmented Generation system for academic papers, CVs, project reports, and job descriptions. It combines document parsing, SentenceTransformer embeddings, ChromaDB vector search, Ollama-based local LLM generation, source-cited answering, and CV-to-job matching in a Streamlit and FastAPI application.

## Suggested GitHub topics

```text
rag
llm
langchain-alternative
chromadb
sentence-transformers
ollama
streamlit
fastapi
document-ai
research-assistant
cv-screening
portfolio-project
```

## Troubleshooting

### Ollama is not reachable

Run:

```bash
ollama serve
```

Then in another terminal:

```bash
ollama pull llama3.2:3b
```

### Embedding model download is slow

The first run downloads the SentenceTransformer model. Later runs will use the cached model.

### ChromaDB database needs reset

Use the **Reset vector database** button in the Streamlit sidebar, or delete the contents of:

```text
chroma_store/
```

### Large PDFs are slow

Use fewer documents first, or reduce chunk size in `.env`:

```text
CHUNK_SIZE=700
CHUNK_OVERLAP=100
```

## Future improvements

- Hybrid dense + keyword retrieval
- Reranking with cross-encoders
- Better RAG evaluation dashboard
- Multi-language/Bangla document support
- Citation accuracy scoring
- Authentication and user-specific document stores
- Deployment on Hugging Face Spaces or Render

