# RAG Hallucination Detection & Explanation Pipeline

This repository contains a complete pipeline for detecting and explaining hallucinations in Retrieval-Augmented Generation (RAG) systems. It trains a model on the **RAGTruth** dataset, evaluates its zero-shot transfer capability on **HaluBench**, and provides a fully local, end-to-end demonstration using Malaysian news articles. 

The system not only flags hallucinated sentences but also uses SHAP to provide explainable AI (XAI) insights into *why* a sentence was flagged.

## 🚀 Quick Start
### 1. Installation
Clone the repository and install the required packages:

`git clone <your-repo-url>
cd detector
pip install -r requirements.txt
python -m spacy download en_core_web_sm`

### 2. Setup Environment
Create a .env file in the root directory and add your Hugging Face token (required for downloading the embedder and NLI models):
`HF_TOKEN=your_huggingface_token_here`

### 3. Download Base Data
Place the RAGTruth raw files (response.jsonl and source_info.jsonl) into data/raw/ragtruth/.

### 4. Train the Detector
To train the XGBoost model on the RAGTruth dataset and freeze it for production:
`python -m src.training.detector_v2`
This will save the frozen models to the models/ (or frozen/) directory and generate cache files.

### 5. Run the Local Malaysian News Demo
To test the pipeline on local Malaysian news using Ollama:

1. Ensure Ollama is running and pull the required models:
`ollama pull qwen2.5:7b
ollama pull llama3.2:latest`

2. Run the pipeline:
`python -m src.pipeline.response_generate`

This script will:
- Load the local news dataset.
- Use Qwen to generate 3 questions per article.
- Use Llama 3.2 to answer them based only on the context.
- Run the frozen detector to flag hallucinations.
- Generate a SHAP explanation for the most suspicious sentence.

☁️ Automated Scraping (GitHub Actions)
The repository includes a GitHub Action (.github/workflows/scrape.yml) that runs daily_scraper.py every day at 00:00 UTC (8:00 AM MYT). It automatically scrapes new Malaysian news from RSS feeds (Malay Mail, FMT, Bernama, Malaysiakini), deduplicates them, and commits the updated SQLite database back to the repository.
You can also trigger it manually via the "Actions" tab on GitHub or the GitHub Mobile app.

📊 Datasets Used
- RAGTruth: For training the core sentence-level hallucination detector.
- HaluBench: For evaluating the model's zero-shot transfer to unseen domains.
- Malaysian News: Custom local dataset scraped daily for real-world RAG testing.

📝 Citation
If you use this codebase or the underlying methodologies, please refer to the original papers:

- RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models
- Lynx: An Open Source Hallucination Evaluation Model (HaluBench)
