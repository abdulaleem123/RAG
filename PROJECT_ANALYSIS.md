# Project Overview

- **Project name:** RAG (Retrieval-Augmented Generation)
- **Domain / problem space:** Document Q&A over custom knowledge bases; RAG pipelines for PDF and web content.
- **Primary purpose:** Index documents (PDF or web), embed and store them in vector stores (Pinecone or FAISS), and answer user questions by retrieving relevant chunks and generating answers via an LLM.

---

# Tech Stack

- **Frontend technologies:** None. No web UI or SPA. User interaction is via:
  - Terminal stdin/stdout in `rag_doc.py` (input/print loop).
  - Jupyter notebook cells (interactive execution).
- **Backend technologies:** Python 3. Scripts and notebooks only; no web framework (Flask/FastAPI), no REST API.
- **Databases:** Vector stores only:
  - **Pinecone** (cloud, serverless AWS us-east-1): used in `rag_doc.py` and `rag.ipynb`.
  - **FAISS** (local, in-memory/disk): used in `007_rag.ipynb`.
- **External services / APIs:**
  - **Pinecone** (index creation, vector upsert, similarity search).
  - **Google Gemini** (embeddings: `models/text-embedding-004`, 768 dimensions).
  - **OpenAI** (LLM: `gpt-4.1-nano` via `init_chat_model` in `007_rag.ipynb`).
- **Libraries & frameworks actually used:**
  - `python-dotenv` (load env; not listed in requirements.txt).
  - `langchain_community` (document loaders: PyPDFLoader, WebBaseLoader).
  - `langchain_text_splitters` (RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter).
  - `langchain_google_genai` (GoogleGenerativeAIEmbeddings).
  - `langchain_openai` (OpenAIEmbeddings in rag.ipynb).
  - `langchain_pinecone` (PineconeVectorStore; not in requirements.txt).
  - `langchain_core` (prompts: ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate; Document).
  - `langchain.chat_models` (init_chat_model).
  - `langgraph` (StateGraph, START, END, TypedDict State).
  - `pinecone` (Pinecone client, ServerlessSpec; not in requirements.txt).
  - `bs4` (BeautifulSoup for WebBaseLoader).
  - `pypdf` (via PyPDFLoader).
  - `faiss-cpu`, `nltk`, `langchain_experimental` (in requirements; nltk/experimental usage not observed in analyzed code).
  - `ipython`, `ipykernel` (notebooks).

---

# High-Level System Behavior

- **rag_doc.py:** On run: load env → load PDF with PyPDFLoader → split with RecursiveCharacterTextSplitter (1000/150) → create or reuse Pinecone index (768 dim, cosine) → embed with Google Gemini and upsert to Pinecone → enter loop: read user query from stdin → similarity_search_with_score(query, k=3) → print top snippets with score and page → repeat until user types exit/quit/q.
- **rag.ipynb:** In-memory markdown → MarkdownHeaderTextSplitter → Google Gemini embeddings → Pinecone index (testing-pinecone-gemini) → add_documents with UUIDs → demo similarity_search and similarity_search_with_score (no LLM answer generation).
- **007_rag.ipynb:** Load from web (WebBaseLoader) or PDF (PyPDFLoader) → RecursiveCharacterTextSplitter (1000/200, add_start_index=True) → Google Gemini embeddings → FAISS vector store (in-memory or load_local) → RAG agent: StateGraph with nodes "retrieve" and "generator". Flow: user question → retrieve (similarity_search k=4) → generator (prompt with context + question, LLM gpt-4.1-nano) → return answer. Invoked via graph.invoke({'question': '...'}).

Overall: **User question** → **vector similarity search** over indexed chunks → **top-k chunks as context** → **LLM prompt (context + question)** → **model answer**. No HTTP layer; CLI or notebook only.

---

# Implemented Features

## 1. PDF document loading and chunking (rag_doc.py, 007_rag.ipynb)

- **What it does:** Loads a single PDF from path, splits into text chunks.
- **Where:** `rag_doc.py` (PyPDFLoader, RecursiveCharacterTextSplitter chunk_size=1000, overlap=150, separators); `007_rag.ipynb` (PyPDFLoader, RecursiveCharacterTextSplitter 1000/200, add_start_index=True).
- **Inputs:** PDF file path (e.g. `2509.03680v1.pdf`).
- **Outputs:** List of LangChain Document objects with metadata (e.g. page).

## 2. Web page loading (007_rag.ipynb)

- **What it does:** Fetches and parses a URL with WebBaseLoader and optional BeautifulSoup strainer.
- **Where:** `007_rag.ipynb` (WebBaseLoader, bs4 SoupStrainer for post-title, post-header, post-content).
- **Inputs:** URL string.
- **Outputs:** List of Document(s) with page_content and source metadata.

## 3. Markdown splitting (rag.ipynb)

- **What it does:** Splits markdown text by headers (#, ##, ###) into chunks with header metadata.
- **Where:** `rag.ipynb` (MarkdownHeaderTextSplitter).
- **Inputs:** Markdown string (hardcoded “Healthy Living” / “Travel Tips” content).
- **Outputs:** List of Document-like chunks with Header 1/2/3 metadata.

## 4. Embedding with Google Gemini (all three assets)

- **What it does:** Produces 768-dimensional embeddings for text.
- **Where:** `rag_doc.py`, `rag.ipynb`, `007_rag.ipynb` (GoogleGenerativeAIEmbeddings model=`models/text-embedding-004`).
- **Inputs:** Text or documents.
- **Outputs:** Embedding vectors (used internally by vector stores).

## 5. Pinecone index lifecycle (rag_doc.py, rag.ipynb)

- **What it does:** Creates index if missing (dimension=768, metric=cosine, ServerlessSpec AWS us-east-1), then uses it for vector storage and search.
- **Where:** `rag_doc.py` (INDEX_NAME=luxdit-paper-index), `rag.ipynb` (index_name=testing-pinecone-gemini).
- **Inputs:** API key from env (PINECONE_API_KEY).
- **Outputs:** Index ready for from_documents / add_documents and similarity_search.

## 6. Vector storage to Pinecone (rag_doc.py, rag.ipynb)

- **What it does:** Embeds chunks and upserts to Pinecone.
- **Where:** `rag_doc.py` (PineconeVectorStore.from_documents); `rag.ipynb` (PineconeVectorStore, add_documents with UUIDs).
- **Inputs:** Chunks (documents), embedding model, index name/index.
- **Outputs:** Vectors stored in Pinecone.

## 7. FAISS vector store (007_rag.ipynb)

- **What it does:** Builds or loads a local FAISS index for similarity search.
- **Where:** `007_rag.ipynb` (FAISS from langchain_community.vectorstores, save/load_local with allow_dangerous_deserialization).
- **Inputs:** Documents + embeddings, or path to saved index.
- **Outputs:** FAISS object used for similarity_search.

## 8. Similarity search (all three)

- **What it does:** Returns top-k document chunks by similarity to a query.
- **Where:** `rag_doc.py` (similarity_search_with_score, k=3); `rag.ipynb` (similarity_search, similarity_search_with_score, k=2); `007_rag.ipynb` (similarity_search, k=4 inside retrieve node).
- **Inputs:** Query string, k.
- **Outputs:** List of (Document, score) or list of Document.

## 9. RAG Q&A with LLM (007_rag.ipynb only)

- **What it does:** Runs a two-step RAG pipeline: retrieve context from vector store, then generate answer with an LLM using a fixed prompt template.
- **Where:** `007_rag.ipynb` (StateGraph: retrieve → generator; ChatPromptTemplate with context + question; init_chat_model('gpt-4.1-nano', 'openai')).
- **Inputs:** state with 'question'; optional 'context' from retrieve.
- **Outputs:** state with 'answer' (LLM response content).

## 10. Interactive CLI Q&A (rag_doc.py only)

- **What it does:** After indexing, loops on stdin: user types a question, system prints top-3 snippets (score + page + content preview); no LLM-generated answer.
- **Where:** `rag_doc.py` (while True input loop, similarity_search_with_score, print).
- **Inputs:** User string (exit/quit/q to stop).
- **Outputs:** Printed snippets; no structured API response.

---

# Data Handling

- **What enters:** (1) PDF files or URLs / markdown text. (2) User questions (free text). (3) Env: PINECONE_API_KEY; Google/OpenAI API keys implied for embeddings/LLM (not explicitly shown in code).
- **Processing:** Load → split → embed (Google) → store in Pinecone or FAISS. At query time: embed query → similarity search → optional prompt+LLM (007_rag.ipynb).
- **Where stored:** Pinecone (named indexes) or local FAISS directory. No relational DB; no user/session persistence.
- **Returned to user:** In rag_doc.py: printed snippets (score, page, content prefix). In rag.ipynb: notebook outputs (chunks, search results). In 007_rag.ipynb: graph output dict with 'answer' and retrieved docs.

---

# Architecture Observations

- **Monolith / modular:** Single Python process; notebooks and one script. No services or microservices. Logic is modular within notebooks (load → split → embed → store → retrieve → generate).
- **Separation of concerns:** Document loading, splitting, embedding, and vector storage are separate steps; 007_rag.ipynb separates retrieval and generation into distinct graph nodes.
- **Patterns:** RAG (retrieve then generate); pipeline/chain (load → split → embed → store); in 007_rag.ipynb a simple two-node LangGraph (retrieve → generator) with TypedDict State. No checkpointing or threading observed in the compiled graph.

---

# Performance Considerations

- **Optimizations:** Chunk overlap and size chosen to keep context (e.g. 150–200 overlap). Pinecone serverless for managed scale. FAISS for local fast ANN. Top-k limited (e.g. k=3 or 4) to cap context size.
- **Bottlenecks/limits:** No caching of embeddings or responses. Single-threaded; no async. rag_doc.py uses time.sleep(2) after index creation. Large PDFs or many chunks will increase embed and upsert time. No batching logic shown for upsert.

---

# Security Considerations

- **Auth/validation/sanitization:** API keys via environment (load_dotenv). No auth for the CLI or notebook (local use). User input is passed directly to embedding and LLM; no input validation or sanitization. No rate limiting or abuse controls.
- **Gaps:** No .env or secrets in repo (good). requirements.txt omits pinecone and python-dotenv despite use. FAISS load_local uses allow_dangerous_deserialization (documented risk). No encryption at rest/transit beyond provider defaults.

---

# Error Handling & Edge Cases

- **Explicit handling:** rag_doc.py checks if Pinecone index exists before create; no try/except. No explicit error handling in the analyzed code.
- **Not handled:** Missing or invalid PDF path, network failures, API errors (Pinecone, Gemini, OpenAI), empty chunk list, missing env vars, invalid graph state. Failures will surface as Python exceptions.

---

# Deployment & Environment

- **Hosting:** Not implemented. No Dockerfile, docker-compose, or deployment config. Assumed local or ad-hoc run (python rag_doc.py or Jupyter).
- **Environment variables:** PINECONE_API_KEY (rag_doc.py, rag.ipynb, 007_rag.ipynb). Google and OpenAI API keys required for embeddings and LLM but not referenced in the analyzed snippets.
- **Build/runtime:** Python with pip install from requirements.txt. No version pinning of Pinecone or python-dotenv in requirements. Jupyter for notebooks.

---

# Known Limitations

- **Partially or narrowly implemented:** RAG with LLM only in 007_rag.ipynb; rag_doc.py and rag.ipynb do not generate answers, only retrieval. No Gradio or web UI despite gradio in requirements. Many requirements (langsmith, langgraph-checkpoint-sqlite, langchain-tavily, etc.) have no visible use in the analyzed files.
- **Missing but implied:** Re-running rag_doc.py with the same index will re-upsert (no “index only if empty” logic). No multi-document or multi-source routing. No citations or source links in the generator output in the notebook. No conversation history or multi-turn state beyond a single invoke.
