import feedparser, trafilatura, sqlite3, os, time, re
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
DB_PATH = "./data/db/malaysian_news_db/malaysian_news.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
RSS_FEEDS = {
    'malaymail': 'https://www.malaymail.com/feed/rss/malaysia',
    'fmt': 'https://cms.freemalaysiatoday.com/feed',
    'bernama': 'https://www.bernama.com/en/rssfeed.php/',
    'malaysiakini': 'https://www.malaysiakini.com/en/rss/news.rss',
}

# ==========================================
# 2. DATABASE SETUP
# ==========================================
def init_db():
    """Create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            publish_date TEXT,
            scraped_at TEXT,
            text TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized.")

def is_url_exists(url):
    """Check if the URL is already in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def save_article(url, source, title, publish_date, text):
    """Insert a new article into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO articles (url, source, title, publish_date, scraped_at, text)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (url, source, title, publish_date, datetime.now().isoformat(), text))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # URL already exists, skip silently
    finally:
        conn.close()

# ==========================================
# 3. SCRAPING LOGIC
# ==========================================
def clean_text(text):
    """Basic cleaning to remove excessive whitespace."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_daily():
    print(f"\n🚀 Starting daily scrape at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_new = 0

    for source, rss_url in RSS_FEEDS.items():
        print(f"\n--- Fetching from {source} ---")
        try:
            feed = feedparser.parse(rss_url)
            # Extract up to 20 links (or however many are available)
            links = [entry.link for entry in feed.entries if hasattr(entry, 'link')]
            print(f"Found {len(links)} links in RSS.")
        except Exception as e:
            print(f"❌ Failed to parse RSS for {source}: {e}")
            continue

        for entry in feed.entries:
            url = getattr(entry, 'link', None)
            if not url:
                continue

            # 1. CHECK FOR DUPLICATES
            if is_url_exists(url):
                continue

            # 2. EXTRACT TEXT
            print(f"  Extracting new article: {url[:60]}...")
            text = None
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            except Exception:
                pass

            if text and len(text.strip()) > 200:
                text = clean_text(text)
                title = getattr(entry, 'title', 'Unknown Title')
                publish_date = getattr(entry, 'published', 'Unknown Date')
                
                # 3. SAVE TO DB
                save_article(url, source, title, publish_date, text)
                print(f"    ✅ Saved ({len(text)} chars)")
                total_new += 1
            else:
                print(f"    ❌ Failed or too short")

            # Be polite
            time.sleep(1)

    print(f"\n🎉 Finished. Added {total_new} new articles to the database.")

# ==========================================
# 4. RUN
# ==========================================
if __name__ == "__main__":
    init_db()
    scrape_daily()