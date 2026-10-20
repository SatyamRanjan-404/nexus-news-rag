import json
import chromadb
# Import the tool that lets LangChain use local sentence-transformers models
from langchain_community.embeddings import HuggingFaceEmbeddings


def embed_and_store():
    try:
        with open('chunked_articles.json','r',encoding='utf-8') as f:
            chunks=json.load(f)
    except FileNotFoundError:
        print("Error: chunked_articles.json not found !")
    
    print(f"Loaded {len(chunks)} text chunks to embed.")

    # 2. Initialize our free embedding model
    # 'all-MiniLM-L6-v2' is a fast, accurate, and light model running locally on your CPU
    print("Loading embedding model ...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Initialize ChromaDB
    # named 'chroma_db' and save your data permanently inside it.
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # 4. Create or fetch our specific database drawer (Collection)
    # get_or_create_collection looks for a drawer named 'news_collection'. 
    # If it doesn't exist, it builds it.
    collection = chroma_client.get_or_create_collection(name="news_collection")

    # list for chromodb 
    ids=[]
    documents=[]
    metadatas=[]

    for chunk in chunks:
        ids.append(chunk['chunk_id'])
        documents.append(chunk['text'])

        metadatas.append({
            "title":chunk['title'],
            "link":chunk['link'],
            "published":chunk['published'],
            "published_ts": chunk['published_ts'], # Add this exact line!
            "source":chunk['source']
        })

    print(f"Generating vectors and saving to ChromoDB...")

    # We use batching because generating vectors takes processing power.
    # We will process 100 items at a time to keep it safe and stable
    batch_size=100
    for i in range(0,len(documents),batch_size):
        end_idx=i+batch_size
        batch_ids=ids[i:end_idx]
        batch_docs=documents[i:end_idx]
        batch_meta=metadatas[i:end_idx]

        batch_embeddings=embedding_model.embed_documents(batch_docs)

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta
        )

        print(f"Processed batch {i // batch_size+1}....")
    total_count=collection.count()
    print(f"\nSuccess! Your vector database now contains {total_count} embedded news chunks.")

if __name__ == "__main__":
    embed_and_store()