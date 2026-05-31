import feedparser
import json
import re
from datetime import datetime
import calendar
import os
import socket  # Import Python's core socket library

# CRITICAL FIX: Set a global timeout of 10 seconds for ALL network operations.
# This prevents feedparser from hanging indefinitely on a dead or lagging server.
socket.setdefaulttimeout(10)

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.strip()

def fetch_news():
    # Load our single source of truth configuration
    with open("news_sources.json", "r", encoding="utf-8") as f:
        all_sources = json.load(f)
    
    active_feeds = [(s["name"], s["url"]) for s in all_sources if s.get("enabled", True)]
    articles = []
    
    # Track metrics for validation logging
    successful_feeds = 0
    failed_feeds = 0
    
    for source_name, url in active_feeds:
        print(f"Fetching: {source_name}...")
        
        # FIX: Wrap each individual feed fetch operation inside its own safety block
        try:
            parsed_feed = feedparser.parse(url)
            
            # Check if feedparser encountered a systemic error or empty parse
            if parsed_feed.bozo:
                # bozo = 1 means the XML formatting was broken or connection timed out
                print(f"⚠️  [FEED WARNING] Non-fatal issue parsing {source_name}. Attempting to read anyway...")
                
            for entry in parsed_feed.entries:
                time_struct = entry.get("published_parsed", None)
                
                if time_struct:
                    published_str = datetime(*time_struct[:6]).isoformat()
                    published_ts = int(calendar.timegm(time_struct))
                else:
                    published_str = datetime.now().isoformat()
                    published_ts = int(datetime.now().timestamp())
                    
                articles.append({
                    "title": entry.get("title", "No Title"),
                    "summary": clean_html(entry.get("summary", "")),
                    "link": entry.get("link", ""),
                    "published": published_str,
                    "published_ts": published_ts,
                    "source": source_name 
                })
            
            successful_feeds += 1
            
        except Exception as feed_err:
            # If a remote end closes a connection, we log it here and seamlessly continue our loop!
            print(f"❌ [FEED ERROR] Skipped network stream '{source_name}' due to connectivity issue: {str(feed_err)}")
            failed_feeds += 1
            continue
            
    # Save whatever successful data we captured
    with open('fetched_articles.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4)
        
    print(f"\n📊 [FETCH SUMMARY] Successfully parsed: {successful_feeds}/{len(active_feeds)} feeds. Failed/Skipped: {failed_feeds}.")