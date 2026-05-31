import streamlit as st
import json
from rag_chain import get_rag_stream_and_sources
import threading # Ensure threading is imported

st.set_page_config(page_title="NEXUS Intelligence", page_icon="logo.png", layout="wide")

from apscheduler.schedulers.background import BackgroundScheduler
from schedular import run_full_pipeline


# Initialize a session state token to check if data is ready
if "pipeline_complete" not in st.session_state:
    st.session_state.pipeline_complete = False

# We use @st.cache_resource so Streamlit ONLY runs this setup code once ever
@st.cache_resource
def start_background_crawler():
    print("⏰ [LAUNCH] Activating background news ingestion engine...")
    bg_scheduler = BackgroundScheduler()
    
    # We schedule the recurring 15-minute sync job
    bg_scheduler.add_job(
        func=run_full_pipeline, 
        trigger="interval", 
        minutes=15, 
        id="nexus_sync_job"
    )
    bg_scheduler.start()
    
    # Launch the initial sync in an independent background thread.
    # This makes this function finish in 0.01 seconds, completely preventing the infinite log looping!
    print("⚡ [STARTUP] Offloading initial ingestion to isolated background thread...")
    threading.Thread(target=run_full_pipeline, daemon=True).start()
    
    return bg_scheduler

# Trigger the background worker cleanly (Returns instantly, locking the cache)
scheduler_client = start_background_crawler()


# 2. INJECT CUSTOM CSS THEME
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500&family=Syne:wght@400..800&display=swap');
    
    /* Global Background and Fonts */
    .stApp {
        background-color: #0A1128; /* Deep Navy */
        color: #FFFFFF;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #e8ff47 !important; /* Yellow-Green accent */
    }
    p, span, div, input {
        font-family: 'DM Mono', monospace !important;
    }
    
    /* The Pulsing Red Dot */
    .pulse-dot {
        height: 12px;
        width: 12px;
        background-color: #ff3333;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(255, 51, 51, 1);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 51, 51, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(255, 51, 51, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 51, 51, 0); }
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. LOAD THE MENU (SSOT)
@st.cache_data(ttl=30)
def load_sources():
    with open("news_sources.json", "r", encoding="utf-8") as f:
        return [s for s in json.load(f) if s.get("enabled", True)]

sources_data = load_sources()

# 4. MEMORY SETUP (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: TWO-STEP FILTER ---
with st.sidebar:
    st.markdown("### 🔴 LIVE FEEDS")
    
    categories = ["All Categories"] + sorted(list(set([s["category"] for s in sources_data])))
    selected_category = st.selectbox("1. Filter by Category:", options=categories)
    
    if selected_category == "All Categories":
        available_feeds = ["All Feeds"] + [s["name"] for s in sources_data]
    else:
        available_feeds = ["All Feeds"] + [s["name"] for s in sources_data if s["category"] == selected_category]
        
    selected_feed_name = st.selectbox("2. Select Source:", options=available_feeds)
    
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN LAYOUT ---
st.markdown("<h1><span class='pulse-dot'></span> NEXUS</h1>", unsafe_allow_html=True)
st.caption("GLOBAL NEWS INTELLIGENCE SYNTHESIS")
st.markdown("---")

# check if our pipeline has successfully saved data to disk yet
import os
data_ready = os.path.exists("chunked_articles.json") and os.path.getsize("chunked_articles.json") > 0

if not data_ready:
    # This alert will show nicely on the UI while the background thread fetches data
    st.info("😁 Collecting news from across the internet for you... Building target index matrices. Please stand by.")
    st.toast("Nexus initialization protocol active...")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "citations" in msg and msg["citations"]:
            for idx, cite in enumerate(msg["citations"]):
                with st.expander(f"[{idx+1}] {cite['title']}"):
                    st.markdown(f"[Read Source]({cite['link']})")

user_query = None
if len(st.session_state.messages) == 0:
    st.markdown("**Suggested Briefings:**")
    col1, col2, col3 = st.columns(3)
    if col1.button("Summarize today's World news"): user_query = "Summarize today's World news"
    if col2.button("Latest Market movements"): user_query = "What are the latest Market movements?"
    if col3.button("Updates in Technology"): user_query = "What are the major updates in Technology?"

chat_input = st.chat_input("Query the global feed...")
if chat_input:
    user_query = chat_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Synthesizing active networks..."):
            stream, citations = get_rag_stream_and_sources(user_query, source_filter=selected_feed_name)
        
        full_response = ""
        if stream is None:
            full_response = "I cannot find explicit details about that topic in the selected active networks."
            response_placeholder.write(full_response)
        else:
            for chunk in stream:
                full_response += chunk
                response_placeholder.write(full_response + "▌")
            response_placeholder.write(full_response)

        if citations:
            st.markdown("---")
            for idx, cite in enumerate(citations):
                with st.expander(f"[{idx+1}] {cite['title']}"):
                    st.markdown(f"[Read Source]({cite['link']})")
                    
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "citations": citations
        })