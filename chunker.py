import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_articles():
    """
    Loads fetched_articles.json, cuts text into overlapping semantic snippets,
    and preserves all metadata tags (including the unix timestamp) for the embedder.
    """
    # 1. Load the raw fetched articles
    with open("fetched_articles.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    # 2. Configure our smart text cutting tool
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Targets ~500 characters per text card
        chunk_overlap=50      # Maintains context continuity across boundaries
    )

    all_chunks = []

    # 3. Process every article
    for idx, article in enumerate(articles):
        # We split based on the article summary text field
        text_to_split = article.get("summary", "")
        if not text_to_split.strip():
            continue

        chunks = text_splitter.split_text(text_to_split)

        # 4. Loop through the broken up pieces of this specific article
        for chunk_idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"art_{idx}_chk_{chunk_idx}",
                "text": chunk_text,
                "title": article.get("title", "Untitled"),
                "link": article.get("link", ""),
                "published": article.get("published", ""),
                # --- THE FIX: Pass the timestamp integer safely down the stream! ---
                "published_ts": article.get("published_ts", 0), 
                "source": article.get("source", "Unknown")
            })

    # 5. Write out the completed structural list
    with open("chunked_articles.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)
        
    print(f"✨ Successfully split {len(articles)} articles into {len(all_chunks)} chunks.")

if __name__ == "__main__":
    chunk_articles()