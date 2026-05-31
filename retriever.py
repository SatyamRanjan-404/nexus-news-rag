from datetime import datetime, timedelta
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings



def retrieve_multi_source_context(query, top_k=4, source_filter=None, days_filter=None):
    """
    Searches ChromaDB for the top_k relevant snippets.
    Combines source filters and time-window range constraints using ChromaDB's '$and' syntax.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="news_collection")

    query_vector = embedding_model.embed_query(query)

    # 1. Build individual sub-filters
    conditions = []
    
    # Add source condition if specified
    if source_filter and source_filter != "All Feeds":
        conditions.append({"source": source_filter})
        
    # Add time condition if specified (e.g., last 1, 7, or 14 days)
    if days_filter and days_filter != "All Time":
        # Calculate the starting date boundary string
        start_date = (datetime.now() - timedelta(days=int(days_filter))).isoformat()
        # We look for records Greater Than or Equal To ($gte) that start date
        conditions.append({"published": {"$gte": start_date}})

    # 2. Package filters together for ChromaDB
    where_clause = None
    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        # If both filters are active, wrap them in an '$and' conditional list
        where_clause = {"$and": conditions}

    # 3. Query the database
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_clause
    )

    if not results['documents'] or len(results['documents'][0]) == 0:
        return None

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    context_blocks = []
    citations = []

    for i in range(len(documents)):
        title = metadatas[i]['title']
        link = metadatas[i]['link']
        text = documents[i]

        block = f"--- Source {i+1}: {title} ---\nContext: {text}\n"
        context_blocks.append(block)
        
        if {"title": title, "link": link} not in citations:
            citations.append({"title": title, "link": link})

    full_context = "\n".join(context_blocks)
    
    return {
        "context": full_context,
        "citations": citations
    }