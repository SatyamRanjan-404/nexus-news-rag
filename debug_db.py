# debug_db.py — paste and run this to see what topics you actually have stored
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("news_collection")

# Get a sample of 20 stored chunks and print their article titles
results = collection.get(limit=20, include=["metadatas"])

print(f"Total chunks in DB: {collection.count()}\n")
print("Sample of stored articles:")
seen_titles = set()
for meta in results["metadatas"]:
    title = meta.get("title", "unknown")
    if title not in seen_titles:
        seen_titles.add(title)
        print(f"  - {title}")