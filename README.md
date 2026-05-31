# NEXUS Intelligence

An automated, multi-threaded Retrieval-Augmented Generation (RAG) system for real-time global news synthesis.

NEXUS continuously ingests, processes, and vectorizes articles from approximately 90 global RSS feeds, enabling natural language queries against live news content with grounded, cited responses.

## Live Deployment

**Hosted Application:**  
https://huggingface.co/spaces/Maytas-Labs/News_rag

---

# Table of Contents

- Architecture Overview
- Data Pipeline
- User Interface
- Technology Stack
- Local Setup
- Environment Variables
- Docker Deployment
- Project Structure

---

# Architecture Overview

NEXUS is structured as a five-stage pipeline:

## 1. Configuration

A centralized `news_sources.json` file defines all active feeds, their display names, categories, and enabled status.

This file is the single source of truth for feed management across both the ingestion layer and the UI layer.

No code changes are required to add, remove, or disable a source.

---

## 2. Ingestion

A Python script reads `news_sources.json`, fetches live RSS XML data using `feedparser`, and normalizes each article into a structured dictionary containing:

- title
- summary
- link
- published date
- source name

HTML markup is stripped from summaries using regular expressions before any further processing.

---

## 3. Chunking and Embedding

Article text is split using LangChain's `RecursiveCharacterTextSplitter`.

**Configuration**

- Chunk Size: 500 characters
- Chunk Overlap: 50 characters

Each chunk is vectorized using:

```text
all-MiniLM-L6-v2
```

from Hugging Face Sentence Transformers.

ChromaDB telemetry is explicitly disabled in the client configuration to suppress instrumentation errors.

---

## 4. Storage and Metadata

Vectors are persisted to a local ChromaDB collection.

Each stored chunk carries the following metadata fields:

| Field | Description |
|---------|-------------|
| source_title | Article headline |
| source_link | Canonical article URL |
| source | Feed name defined in news_sources.json |
| published | Human-readable ISO date string |
| published_ts | Unix timestamp used for date filtering |

`published_ts` enables high-speed chronological range filtering using ChromaDB's native operators:

```python
$lt
$gte
```

---

## 5. Retrieval and Generation

User queries are embedded using the same embedding model:

```text
all-MiniLM-L6-v2
```

ChromaDB performs cosine similarity search with:

```python
top_k = 4
```

Optional metadata filtering allows retrieval from a specific source selected by the user.

Retrieved chunks are passed into a LangChain LCEL chain, which calls the Groq API:

```text
llama-3.3-70b-versatile
```

with:

```python
temperature = 0.2
```

Responses are streamed back to the frontend.

---

# Data Pipeline

## Feed Configuration

All feeds are defined in `news_sources.json`.

Example:

```json
{
  "name": "Reuters Business",
  "url": "https://feeds.reuters.com/reuters/businessNews",
  "category": "Business",
  "icon": "📈",
  "enabled": true
}
```

To add a new feed, append an entry to this file.

The application reloads sources automatically every 30 seconds using:

```python
@st.cache_data(ttl=30)
```

No application restart is required.

---

## Automated Maintenance

### Scheduled Ingestion

APScheduler runs a background task every:

```text
15 minutes
```

to fetch and embed new articles.

### Retention Policy

Any vector chunk older than:

```text
14 days
```

is identified using:

```python
$lt
```

filters and deleted automatically.

This maintains a rolling 14-day searchable news archive while preventing unbounded storage growth.

---

## Startup Architecture

On startup, the initial database build executes in a background daemon thread:

```python
threading.Thread(daemon=True)
```

This allows:

- Immediate UI rendering
- Non-blocking startup
- Background database population

A loading indicator is shown while the vector database initializes.

---

# User Interface

The frontend is built using Streamlit with custom CSS.

## Theme

### Colors

| Element | Color |
|----------|---------|
| Background | #0A1128 |
| Accent | #e8ff47 |

### Typography

- Syne (headings)
- DM Mono (labels and metadata)

Loaded through Google Fonts.

---

## Sidebar Filters

The sidebar implements a hierarchical filtering system.

### Step 1

Category dropdown populated from distinct category values in:

```text
news_sources.json
```

### Step 2

Source dropdown dynamically updates based on the selected category.

This prevents source lists from becoming unwieldy as additional feeds are added.

---

## Additional Features

### Suggested Query Chips

One-click query submission from the empty state.

### Chat History

Maintained using:

```python
st.session_state
```

Persists across Streamlit reruns.

### Session Reset

A clear-session button resets chat history.

### Source Citations

Displayed as expandable cards containing:

- Article title
- Direct URL

### Active Filters

Displayed above the chat input as status pills.

Examples:

- Source
- Category
- Timeframe

---

# Technology Stack

| Layer | Technology |
|---------|-------------|
| Language | Python 3.10+ |
| Frontend | Streamlit |
| RAG Framework | LangChain (LCEL) |
| LLM Provider | Groq API |
| Model | llama-3.3-70b-versatile |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| RSS Parsing | feedparser |
| Scheduling | APScheduler |
| Containerization | Docker |
| Hosting | Hugging Face Spaces |

---

# Local Setup

## Prerequisites

- Python 3.10+
- Git

---

## 1. Clone Repository

```bash
git clone https://github.com/your-org/nexus-intelligence.git
cd nexus-intelligence
```

---

## 2. Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### CPU-only PyTorch

```txt
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.2+cpu
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key from:

https://console.groq.com

No credit card required.

---

## 5. Run Application

```bash
streamlit run app.py
```

Application URL:

```text
http://localhost:8501
```

First launch downloads:

```text
all-MiniLM-L6-v2 (~90 MB)
```

and caches it locally.

---

# Environment Variables

| Variable | Required | Description |
|-----------|----------|-------------|
| GROQ_API_KEY | Yes | Groq API key used for inference |

No additional API keys are required.

---

# Docker Deployment

## Build and Run

```bash
docker-compose up --build
```



# Hugging Face Spaces Deployment

NEXUS is deployed on Hugging Face Spaces using the Docker SDK.

### Configuration

| Resource | Value |
|-----------|--------|
| Instance | CPU Basic |
| CPU | 2 vCPU |
| RAM | 16 GB |

ChromaDB runs locally inside the container.

Because Space storage is non-persistent across restarts, the vector database is rebuilt on each cold start using the background threading architecture.

---

# Project Structure

```text
nexus-intelligence/
├── app.py
├── rss_fetcher.py
├── chunker.py
├── embedder.py
├── retriever.py
├── rag_chain.py
├── db_manager.py
├── scheduler.py
├── news_sources.json
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── chroma_db/
```

### Description

| File | Purpose |
|--------|---------|
| app.py | Streamlit frontend and session management |
| rss_fetcher.py | RSS ingestion, cleaning, deduplication |
| chunker.py | Text splitting |
| embedder.py | Embeddings + ChromaDB upsert |
| retriever.py | Similarity search |
| rag_chain.py | LangChain LCEL chain |
| db_manager.py | Retention policy |
| scheduler.py | APScheduler ingestion |
| news_sources.json | RSS source configuration |
| Dockerfile | Container definition |
| docker-compose.yml | Local deployment |

---

# Adding News Sources

Open:

```text
news_sources.json
```

Append a new object:

```json
{
  "name": "Display Name",
  "url": "https://example.com/feed.rss",
  "category": "CategoryName",
  "icon": "📰",
  "enabled": true
}
```

Save the file.

The application will detect changes within:

```text
30 seconds
```

The scheduler will include the new source during its next ingestion cycle.


---
