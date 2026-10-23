# NEXUS Intelligence — AI News RAG & Claim Checker

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FC521F?style=flat-square)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-F55036?style=flat-square)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

> **A high-precision, citation-grounded Retrieval-Augmented Generation (RAG) platform and automated Claim Verification system for breaking news, regulatory circulars, and viral information.**

---

## 📌 Overview

**NEXUS Intelligence** is an end-to-end news synthesis and verification platform designed to address the key challenges of Large Language Models in current affairs: **knowledge cutoffs** and **hallucination**. 

Rather than relying on ungrounded generative responses or unpredictable live web scraping, NEXUS maintains an indexed repository of primary government gazettes, regulatory announcements, certified fact-checker debunks, and credible news wires. It pairs a **two-stage hybrid retrieval engine** with strict **citation-enforced generation** to produce objective, fact-based answers and forensic audits of viral rumors.

---

## ✨ Key Features

* **Multi-Tier News Ingestion:**
  * Collects and processes articles across structured credibility tiers:
    * **Tier 1:** Official statutory gazettes and regulators (RBI, SEBI, PIB, WHO, NASA).
    * **Tier 2:** Certified fact-checking organizations (BOOM Live, Factly).
    * **Tier 3:** Verified financial and general news wires (The Hindu, Livemint, Reuters).
  * Automatically extracts clean full-text articles via statistical web parsing and eliminates syndicated wire duplicates using SimHash fingerprinting.

* **Two-Stage Hybrid Retrieval & Reranking:**
  * Combines dense semantic vector search (`sentence-transformers/all-MiniLM-L6-v2`) with sparse keyword matching (BM25) using **Reciprocal Rank Fusion (RRF)**.
  * Re-ranks candidates using a dedicated **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`) for deep token-level query-document attention.

* **100% Grounded News Intelligence:**
  * Powered by Groq-accelerated Llama 3.3 70B with LangChain Expression Language (LCEL).
  * Enforces line-level bracketed citations `[1]`, `[2]` linking every fact directly to its source URL, publisher, and publication timestamp.
  * Explicit abstention guardrails prevent speculative or out-of-corpus hallucinations.

* **Forensic News Claim Checker:**
  * Dedicated verification pipeline for viral claims, headlines, and social media forwards.
  * Audits claims into structured verdicts (`TRUE`, `FALSE`, `MISLEADING`, `UNVERIFIED`) backed by confidence scores and supporting evidence snippets.

* **Minimalist Aerospace Web Console:**
  * Dark, high-contrast user interface inspired by industrial aerospace design for exploring intelligence queries and verifying claims in real time.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language & Frameworks** | Python 3.10+, FastAPI, Uvicorn, Pydantic v2 |
| **LLM & Orchestration** | LangChain (LCEL), Groq Cloud API (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) |
| **Vector Database & Embeddings** | ChromaDB (`PersistentClient`), `sentence-transformers/all-MiniLM-L6-v2` |
| **Retrieval & Reranking** | Rank-BM25, Reciprocal Rank Fusion (RRF), Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **Ingestion & Text Processing** | `feedparser`, `trafilatura`, `BeautifulSoup4`, `simhash`, `rapidfuzz` |
| **User Interface** | Streamlit |
| **Packaging & Deployment** | Docker, Docker Compose, Uvicorn ASGI |

---

## 🏗️ System Architecture & Methodology

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         1. HIERARCHICAL INGESTION                           │
 │                                                                             │
 │  Tier 1: Official Gazettes     Tier 2: Fact-Checkers     Tier 3: News Media │
 │  (RBI, SEBI, PIB, WHO)         (BOOM Live, Factly)       (The Hindu, Mint)  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ Full-Text Extraction (trafilatura)
                                        │ Near-Duplicate Deduplication (SimHash)
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         2. CHUNKING & DUAL INDEXING                         │
 │                                                                             │
 │       ChromaDB Vector Store                     In-Memory BM25 Index        │
 │     (384-dim Dense Embeddings)                (Exact Token Inverted Index)  │
 └──────────────────┬──────────────────────────────────────┬───────────────────┘
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     3. TWO-STAGE RETRIEVE & RERANK                          │
 │                                                                             │
 │   1. Dense Search (Top 20)           2. BM25 Keyword Search (Top 20)        │
 │   3. Reciprocal Rank Fusion (RRF)    ──► Merged Candidate Set               │
 │   4. Cross-Encoder Reranker          ──► Filtered Top Context Passages      │
 └─────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                       4. GENERATION & VERIFICATION                          │
 │                                                                             │
 │   • Grounded Q&A Chain (LCEL): Synthesizes facts with line citations [1], [2]│
 │   • Claim Auditor: Classifies claims into TRUE / FALSE / MISLEADING / UNVER.│
 └─────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
                     FastAPI REST Service & Streamlit Console
```

---

## 📂 Project Structure

```text
nexus-news-rag/
├── src/nexus/
│   ├── api/                 # FastAPI REST API routes, models, and schemas
│   │   ├── main.py          # REST endpoints (/v1/ask, /v1/check-claim, /health)
│   │   └── models.py        # Pydantic request and response contracts
│   ├── chunking/            # Recursive character and semantic chunking engines
│   ├── claims/              # Forensic claim verification pipeline and models
│   ├── config/              # Central configuration and environment settings
│   ├── embeddings/          # Embedding model wrappers (all-MiniLM-L6-v2)
│   ├── evaluation/          # Evaluation harness, metrics, and LLM judge
│   ├── generation/          # Grounded LCEL RAG chains and prompt templates
│   ├── indexing/            # ChromaDB vector store and BM25 index managers
│   ├── ingestion/           # RSS fetching, full-text extraction, and dedup
│   ├── rerank/              # Cross-Encoder joint scoring pipeline
│   ├── retrieval/           # Hybrid RRF search and conversational query rewriter
│   └── ui/                  # Streamlit web application
├── data/                    # Snapshot datasets and local storage
├── news_sources.json        # Configured registry of news feeds and credibility tiers
├── app.py                   # Streamlit application entry point
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service deployment compose specification
├── pyproject.toml           # Project dependencies and build configuration
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10 or 3.11
* A [Groq API Key](https://console.groq.com/) for LLM inference

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SatyamRanjan-404/nexus-news-rag.git
cd nexus-news-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Configure your environment variables:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
PORT=8000
CHROMA_PERSIST_DIR=./data/chroma_db
```

---

## 🖥️ Running the Application

### Option A: Running with Streamlit Web Console
To launch the interactive intelligence and claim checking portal:

```bash
streamlit run app.py
```
Access the web portal at `http://localhost:8501`.

### Option B: Running the FastAPI REST Backend
To run the high-performance API server:

```bash
uvicorn nexus.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.

### Option C: Running with Docker Compose

```bash
docker-compose up --build
```

---

## 📡 REST API Reference

### 1. Intelligence Query (`POST /v1/ask`)
Generates a grounded, cited answer for a news or policy inquiry.

**Request:**
```json
POST /v1/ask
Content-Type: application/json

{
  "query": "What repo rate policy decision was announced by the RBI Monetary Policy Committee?",
  "top_k": 4
}
```

**Response:**
```json
{
  "query": "What repo rate policy decision was announced by the RBI Monetary Policy Committee?",
  "answer": "The Reserve Bank of India Monetary Policy Committee kept the policy repo rate unchanged at 6.50% [1], maintaining focus on withdrawal of accommodation to ensure inflation progressively aligns to the 4.0% target [2].",
  "citations": [
    {
      "title": "Monetary Policy Committee (MPC) Statement on Policy Repo Rate",
      "url": "https://rbi.org.in/pressreleases/mpc_resolution.html",
      "source_name": "RBI Press Releases",
      "tier": 1
    }
  ],
  "latency_ms": 320.5
}
```

### 2. Claim Verification (`POST /v1/check-claim`)
Audits a headline, rumor, or forward against indexed primary gazettes and fact-checkers.

**Request:**
```json
POST /v1/check-claim
Content-Type: application/json

{
  "claim": "RBI has completely cancelled the legal tender status of all 500 rupee notes starting next month."
}
```

**Response:**
```json
{
  "claim": "RBI has completely cancelled the legal tender status of all 500 rupee notes starting next month.",
  "verdict": "FALSE",
  "confidence": 0.98,
  "explanation": "The Reserve Bank of India has issued no directive or circular invalidating 500-rupee notes. All legal tender currency remains valid in circulation.",
  "evidence_snippets": [
    {
      "title": "Fact Check: Viral Claim Alleging RBI Demonetisation of 500 Rupee Notes is Completely False",
      "source": "BOOM Live Fact Check",
      "tier": 2,
      "url": "https://boomlive.in/fact-check/rbi-has-not-demonetised-500-rupee-currency-notes.html"
    }
  ]
}
```

### 3. System Health (`GET /health`)
Returns service and database health status.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
