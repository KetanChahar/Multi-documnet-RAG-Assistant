# 📄 Multi-Document RAG Assistant

A Retrieval-Augmented Generation (RAG) app that lets you upload multiple
documents (PDF/TXT) and chat with them — answers are grounded strictly in
your documents, with source citations and a built-in retrieval evaluation
suite to measure how well the system actually retrieves relevant content.

Built with **LangChain**, **LangGraph**, **FAISS**, **Streamlit**, and **Groq**.

---

## ✨ Features

- **Multi-document ingestion** — upload several PDFs/TXT files at once; each
  chunk is tagged with its source file, so retrieval surfaces the right
  document per question instead of mixing sources together.
- **Grounded answers with citations** — the LLM is prompted to answer only
  from retrieved context and cite its sources; if the context doesn't
  contain the answer, it says so instead of hallucinating.
- **Graph-based pipeline (LangGraph)** — retrieval and generation are
  explicit nodes in a state graph, making the pipeline easy to extend
  (e.g. adding a "no relevant docs" branch or a web-search fallback).
- **Chat interface** — Streamlit chat UI with conversation history and an
  expandable view of exactly which chunks were retrieved for each answer.
- **Retrieval evaluation suite** — a standalone eval script scoring the
  retriever on Hit Rate@k, MRR, and keyword coverage against a hand-built
  question set, decoupled from generation quality.

## 🏗️ Architecture

```
                 ┌──────────────┐
   PDF / TXT  →  │   ingest.py  │  → chunk → embed → FAISS index
                 └──────────────┘

                 ┌──────────────────────────────┐
   Question   →  │      rag_chain.py (LangGraph) │
                 │                               │
                 │   START → retrieve → generate → END
                 └──────────────────────────────┘
                          │
                          ▼
                 ┌──────────────┐
                 │    app.py    │  Streamlit chat UI
                 └──────────────┘

                 ┌──────────────┐
                 │   eval.py    │  Hit Rate@k / MRR / keyword coverage
                 └──────────────┘  (tests retrieval in isolation)
```

## 🛠️ Tech Stack

| Layer            | Tool                                   |
|-------------------|-----------------------------------------|
| Orchestration     | LangChain, LangGraph                   |
| Embeddings        | Google Generative AI Embeddings         |
| LLM (generation)  | Groq (`openai/gpt-oss-120b`)            |
| Vector store       | FAISS                                   |
| UI                | Streamlit                               |
| Doc loading       | PyPDFLoader, TextLoader                 |

## 🚀 Getting Started

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/multi-doc-rag-assistant.git
cd multi-doc-rag-assistant
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
```
Add your `GOOGLE_API_KEY` (embeddings) and `GROQ_API_KEY` (generation) to `.env`.

### 3. Ingest a document
```bash
python ingest.py path/to/your.pdf
```

### 4. Test retrieval + generation
```bash
python rag_chain.py
```

### 5. Evaluate retrieval quality
```bash
python eval.py
```
Edit `EVAL_SET` inside `eval.py` with question/expected-source pairs specific
to your documents.

### 6. Run the app
```bash
streamlit run app.py
```

## 📊 Evaluating Retrieval Quality

Generation quality is downstream of retrieval — if the answer is wrong, the
first thing to check is whether the right chunks were retrieved at all.
`eval.py` isolates that step and reports:

- **Hit Rate@k** — % of test questions where the correct source document was
  retrieved
- **MRR (Mean Reciprocal Rank)** — rewards finding the right chunk *earlier*
  in the ranked results
- **Keyword coverage** — how much expected content actually appears in the
  retrieved text

## 📁 Project Structure

```
multi-doc-rag-assistant/
├── app.py              # Streamlit chat UI
├── ingest.py            # Document loading, chunking, embedding, FAISS storage
├── rag_chain.py          # LangGraph retrieval + generation pipeline
├── eval.py               # Retrieval quality evaluation suite
├── requirements.txt
├── .env.example
└── README.md
```

## 🔭 Possible Extensions

- Confidence-based routing: fall back to "no relevant document found" (or a
  web search node) when retrieval similarity is below a threshold
- Swap FAISS for Chroma or a managed vector DB for production scale
- Add RAGAS for automated, LLM-judged evaluation (context precision/recall,
  faithfulness) as a step up from the manual eval set

## 📄 License

MIT — feel free to use this as a reference or starting point for your own
RAG projects.
