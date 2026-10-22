import pandas as pd
import re
import time
import feedparser
import trafilatura
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 1. RSS Feed Configuration
# ==========================================
rss_dict = {
    'malaymail': 'https://www.malaymail.com/feed/rss/malaysia',
    'fmt': 'https://cms.freemalaysiatoday.com/feed',
    'bernama': 'https://www.bernama.com/en/rssfeed.php/',
    'malaysiakini': 'https://www.malaysiakini.com/en/rss/news.rss',
    # 'astroawani': 'https://freenewsapi.ai/v1/rss?host=luminews.my&size=50'  # Temporarily disabled if unstable
}

# ==========================================
# 2. Retrieve Links and Parse Article Content
# ==========================================
all_articles = []

for source, rss_url in rss_dict.items():
    print(f"\n--- Fetching from {source} ---")
    try:
        feed = feedparser.parse(rss_url)
        # Extract article URLs from RSS
        links = [entry.link for entry in feed.entries[:20] if hasattr(entry, 'link')]
        print(f"Found {len(links)} links.")
    except Exception as e:
        print(f"Failed to parse RSS for {source}: {e}")
        continue

    for url in links:
        print(f"  Extracting: {url[:60]}...")
        text = None
        
        # Primary extraction with trafilatura (best for news)
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        except Exception as e:
            print(f"    trafilatura failed: {e}")

        if text and len(text.strip()) > 200:  # Filter out too-short content (likely navigation bars, etc.)
            all_articles.append({
                "source": source, 
                "url": url, 
                "text": text
            })
            print(f"    ✅ Extracted ({len(text)} chars)")
        else:
            print(f"    ❌ Failed or too short")
        
        # Be polite, avoid hammering servers
        time.sleep(2) 

# ==========================================
# 3. Store to CSV and Clean
# ==========================================
if not all_articles:
    print("\n⚠️ No articles extracted. Please check RSS links or network.")
else:
    df = pd.DataFrame(all_articles)
    
    # Deduplication (some RSS feeds may contain duplicate articles)
    df = df.drop_duplicates(subset=['url'])
    
    df.to_csv("./data/raw/malaysian_news/malaysian_news.csv", index=False)
    print(f"\n✅ Saved {len(df)} articles to malaysian_news.csv")

    # Clean text: remove excess whitespace, keep normal characters
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)  # Merge multiple spaces/newlines
        return text.strip()

    df['text'] = df['text'].apply(clean_text)
    
    # Filter out empty text again just in case
    df = df[df['text'].str.len() > 0]
    df.to_csv("./data/raw/malaysian_news/malaysian_news_clean.csv", index=False)
    print(f"✅ Saved {len(df)} cleaned articles to malaysian_news_clean.csv")

    # ==========================================
    # 4. Build Chroma Vector Store
    # ==========================================
    print("\n--- Building Vector Database ---")
    loader = CSVLoader(file_path="./data/raw/malaysian_news/malaysian_news_clean.csv", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./data/db/malaysian_news_db"
    )

    print(f"✅ Vector database built with {len(chunks)} chunks")