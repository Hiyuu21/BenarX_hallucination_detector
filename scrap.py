import trafilatura
import pandas as pd
import re

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Put your Malaysian news URLs here
urls = [
    "https://www.thestar.com.my/news/nation/2026/10/21/singapore-and-malaysia-lead-the-largest-appliededu-asean-rankings-to-date",
    "https://www.thestar.com.my/news/nation/2026/10/21/sabak-bernam-mp-switches-to-javanese-to-urge-prabowo-to-tackle-trans-boundary-haze",
    "https://www.thestar.com.my/news/nation/2026/10/21/akmal-to-sell-motorbike-farmed-fish-to-help-settle-teresa-kok-lawsuit",
    "https://www.thestar.com.my/news/nation/2026/10/21/pm-expresses-concern-over-prolonged-middle-east-crisis-condemns-houthi-attacks",
    "https://www.thestar.com.my/news/nation/2026/10/21/mcmc-records-statement-of-individual-over-fb-content-allegedly-insulting-to-negri-royal-institution",
    "https://www.thestar.com.my/news/nation/2026/10/21/pasir-gudang-mp-laments-absence-of-ministers-during-his-budget-2027-debate",
    "https://www.thestar.com.my/news/nation/2026/10/21/kuching-cops-arrest-student-for-injuring-teacher-with-axe",
    "https://www.thestar.com.my/news/nation/2026/10/21/haze-selangor-and-kl-schools-to-reopen-on-oct-22"
]

articles = []
for url in urls:
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if text:
        articles.append({"url": url, "text": text})
        print(f"✅ Extracted: {url[:60]}... ({len(text)} chars)")
    else:
        print(f"❌ Failed: {url}")

df = pd.DataFrame(articles)
df.to_csv("malaysian_news.csv", index=False)
print(f"\nSaved {len(df)} articles to malaysian_news.csv")


def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove weird unicode characters (optional)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()

df['text'] = df['text'].apply(clean_text)
df.to_csv("malaysian_news_clean.csv", index=False)

# 1. Load the CSV
loader = CSVLoader(file_path="malaysian_news_clean.csv", encoding="utf-8")
docs = loader.load()

# 2. Split into chunks (so retrieval can find specific passages)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create embeddings (turn text into vectors)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./malaysian_news_db"
)

print(f"✅ Vector database built with {len(chunks)} chunks")