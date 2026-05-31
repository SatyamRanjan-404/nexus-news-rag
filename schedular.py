import time
import os
from datetime import datetime, timedelta
import chromadb
from apscheduler.schedulers.background import BackgroundScheduler

from rss_fetcher import fetch_news
from chunker import chunk_articles
from embedder import embed_and_store


def run_retention_cleanup(days_to_keep=14):
    """
    Connects to ChromaDB and purges all records older than the retention threshold
    using integer timestamp math.
    """
    print(f"Initiating database maintenance. Retention policy: {days_to_keep} days.")
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="news_collection")
    
    # Calculate the threshold timestamp (Current Time minus 14 Days)
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    cutoff_ts = int(cutoff_date.timestamp())
    
    # Query ChromaDB to delete any chunk where 'published_ts' is less than (<) our threshold
    collection.delete(
        where={"published_ts": {"$lt": cutoff_ts}}
    )
    print(f" Maintenance complete. All vector chunks older than timestamp {cutoff_ts} pruned.")

def run_full_pipeline():
    print(f"\n⚡ [PIPELINE STARTED] Triggering scheduled data refresh at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    try:
        print(" -> Step 1/4: Fetching latest RSS entries...")
        fetch_news()
        
        print(" -> Step 2/4: Segmenting text summaries into semantic chunks...")
        chunk_articles()
        
        print(" -> Step 3/4: Vectorizing chunks and updating ChromaDB...")
        embed_and_store()
        
        # New Step 4: Run the deletion script immediately after storage updates!
        print(" -> Step 4/4: Executing data retention policy cleanup...")
        run_retention_cleanup(days_to_keep=14)
        
        print(" [PIPELINE SUCCESS] Vector database completely synchronized and pruned.\n")
        
    except Exception as e:
        print(f"❌ [PIPELINE ERROR] Automated ingestion failed: {str(e)}")

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=run_full_pipeline, trigger="interval", minutes=15, id="news_sync_job",max_instances=2)
    scheduler.start()
    
    print(" News Ingestion & Pruning Scheduler Activated! ")
    run_full_pipeline()
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("\n👋 Scheduler stopped successfully.")