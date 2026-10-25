import pandas as pd
import spacy, os, feedparser, trafilatura, sqlite3, time, re
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from config import CACHE_DIR, RPATH, SPATH, DB_PATH, RSS_FEEDS
from pathlib import Path

NLP = spacy.load("en_core_web_sm")
tqdm.pandas(' --- Processing...')

class DataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.train_cache = f'{CACHE_DIR}/exploded_train_cache.pkl'
        self.test_cache = f'{CACHE_DIR}/exploded_test_cache.pkl'
        self.hb_cache = f'{CACHE_DIR}/hb_exploded_cache.pkl'

    def chunk_into_sentence(self, text):
        doc = NLP(text)
        return [(sent.text.strip(), sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()]

    def split_dataset(self, data, split_type):
        return data[(data['split'] == split_type) & (data['quality']=='good')]

    def merge_response(self, response, source):
        return pd.merge(response, source, on = 'source_id', how = 'inner')

    def parse_rows(self, merged_df) -> pd.DataFrame:
        parsed_set = []
        for _, row in merged_df.iterrows():
            d = row.to_dict()
            d.pop('prompt', None)
            d['type'] = 'response'
            label = sorted(d['labels'], key=lambda x: x['start'])
            format_label = {'baseless info': [], 'conflict': []}
            for l in label:
                if l['label_type'].lower().find('baseless')>=0:
                    format_label['baseless info'].append(l['text'])
                else:
                    format_label['conflict'].append(l['text'])
            d['format_label'] = format_label
            if row['task_type']=='QA':
                d['reference'] = row['source_info']['passages']
                d['question'] = row['source_info']['question']
            elif row['task_type']=='Summary':
                d['reference'] = row['source_info']
            else:
                d['reference'] = f"{row['source_info']}"
            parsed_set.append(d)
        return pd.DataFrame(self.parsed_set)

    def load_ragtruth(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and merges response.jsonl and source_info.jsonl"""

        if os.path.exists(self.train_cache) and os.path.exists(self.test_cache):
            print('\n --- Loading cached data')
            exploded_train = pd.read_pickle(self.train_cache)
            exploded_test = pd.read_pickle(self.test_cache)
            return (exploded_train, exploded_test)

        print('\n --- Chunking and extarcting features.')
        responses = pd.read_json(RPATH, lines=True)
        sources = pd.read_json(SPATH, lines=True)

        train = self.merge_response(self.split_dataset(responses, 'train'), sources)
        test = self.merge_response(self.split_dataset(responses, 'test'), sources)

        train = self.parse_rows(train)
        test = self.parse_rows(test)

        train['sentence_chunks'] = train['response'].progress_apply(self.chunk_into_sentence)
        test['sentence_chunks'] = test['response'].progress_apply(self.chunk_into_sentence)

        exploded_train = train.explode('sentence_chunks').dropna(subset=['sentence_chunks']).reset_index(drop=True)
        exploded_test = test.explode('sentence_chunks').dropna(subset=['sentence_chunks']).reset_index(drop=True)

        exploded_train[['sentence_chunks', 'start_char', 'end_char']] = pd.DataFrame(exploded_train['sentence_chunks'].to_list(), index=exploded_train.index)
        exploded_test[['sentence_chunks', 'start_char', 'end_char']] = pd.DataFrame(exploded_test['sentence_chunks'].to_list(), index=exploded_test.index)

        exploded_train.to_pickle(self.train_cache)
        exploded_test.to_pickle(self.test_cache)
        print(f"--- Saved to cache: {self.train_cache} & {self.test_cache}.")

        return (exploded_train, exploded_test)
        
    def load_halubench(self) -> pd.DataFrame:
        """Loads HaluBench from HuggingFace"""

        if os.path.exists(self.hb_cache):
            print('\n --- Loading cached data')
            return pd.read_pickle(self.hb_cache)
        
        else:
            print('\n --- Chunking and extarcting features.')
            df = pd.DataFrame(load_dataset("PatronusAI/HaluBench")['test'])
            df = df[df['source_ds'] != 'RAGTruth'].copy()

            # Renaming columns in HaluBench dataset to make sure the columns can fit into the functions
            df = df.rename(
                columns = {'passage':'reference', 'answer':'response', 'id':'source_id', 'source_ds':'task_type'}
            )

            df['is_hallucination'] = (df['label'] == 'FAIL').astype(int) # FAIL = 1 (hallucinated), PASS = 0
            df['id'] = df.index.astype(str)

            df['sentence_chunks'] = df['response'].progress_apply(self.chunk_into_sentence)
            exploded_df = df.explode('sentence_chunks').dropna(subset = ['sentence_chunks']).reset_index(drop=True)
            unpacked = pd.DataFrame(exploded_df['sentence_chunks'].to_list(), index=exploded_df.index)
            exploded_df['sentence_chunks'] = unpacked[0]

            # start and end char is the index of the words in the response sentences where the hallucination begins and ends.
            exploded_df['start_char'] = unpacked[1]
            exploded_df['end_char'] = unpacked[2]

            exploded_df.to_pickle(self.hb_cache)

            return exploded_df
        
    def load_malaysian_news(self) -> pd.DataFrame:
        """Loads local CSV"""
        return pd.read_csv(self.data_dir / "malaysian_news_clean.csv")


class NewsScraper:
    def __init__(self):
        self.db_path = DB_PATH
        self.rss_feeds = RSS_FEEDS
        
    def init_db(self):
        """Creates SQLite tables"""
        conn = sqlite3.connect(self.db_path)
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

    def is_url_exists(self, url):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
        exist = cursor.fetchone() is not None
        conn.close()
        return exist

    def save_article(self, url, source, title, publish_date, text):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO articles (url, source, title, publish_date, scraped_at, text)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (url, source, title, publish_date, datetime.now().isoformat(), text))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        finally:
            conn.close()

    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def scrape_all(self):
        self.init_db()
        """Loops through RSS feeds, fetches URLs, extracts text, saves to DB"""
        print(f'\n --- Scraping at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        total_new = 0

        for source, rss_url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(rss_url)
                links = [entry.link for entry in feed.entries if hasattr(entry, 'link')]
                print(f' --- Found {len(links)} links in RSS.')
            except Exception as e:
                print(f" --- Error: {e}.")
                continue

            for entry in feed.entries:
                url = getattr(entry, 'link', None)
                if not url:
                    continue

                if self.is_url_exists(url):
                    continue

                text = None
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                except Exception as e:
                    pass

                if text and len(text.strip()) > 200:
                    text = self.clean_text(text)
                    title = getattr(entry, 'title', 'Unknown Title')
                    publish_date = getattr(entry, 'published', 'Unknown Date')
                    self.save_article(url, source, title, publish_date, text)
                    print(f" --- Saved {len(text)} chars.")
                    total_new += 1
                else:
                    print(f" --- Error saving.")
                time.sleep(2)
        print(f"\n --- Added {total_new} articles to the database.")