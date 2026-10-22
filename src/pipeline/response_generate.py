import sys
import os
# This tells Python to look 2 folders up (the 'detector' folder) for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ============================================================
# IMPORTS & SETUP
# ============================================================
import re, json, pickle, joblib, spacy, shap
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# LangChain imports
from langchain_ollama import OllamaLLM

# Import custom feature builder from your project
from src.training.detector_v2 import build_features_v2

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load resources
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer('all-MiniLM-L6-v2', token=os.getenv("HF_TOKEN"))

# ============================================================
# HELPER FUNCTIONS
# ============================================================
SKIP_ENT_LABELS = {"CARDINAL", "ORDINAL"}

def get_best_source_sentence(answer_sentence, source_text):
    """Find the source sentence most similar to the answer sentence."""
    source_doc = nlp(source_text)
    source_sents = [s.text.strip() for s in source_doc.sents if s.text.strip()]
    if not source_sents:
        return "", 0.0

    source_embs = embedder.encode(source_sents, show_progress_bar=False)
    answer_emb = embedder.encode([answer_sentence], show_progress_bar=False)
    sims = cosine_similarity(answer_emb, source_embs)[0]
    
    best_idx = int(np.argmax(sims))
    return source_sents[best_idx], float(sims[best_idx])

def get_entity_number_evidence(answer_sentence, source_text):
    """Extract matched and missing entities/numbers."""
    doc = nlp(answer_sentence)
    entities = [e.text for e in doc.ents if e.label_ not in SKIP_ENT_LABELS]
    numbers = re.findall(r'\d+(?:\.\d+)?', answer_sentence)
    source_lower = source_text.lower()

    return {
        "matched_entities": [e for e in entities if e.lower() in source_lower],
        "missing_entities": [e for e in entities if e.lower() not in source_lower],
        "matched_numbers": [n for n in numbers if n in source_text],
        "missing_numbers": [n for n in numbers if n not in source_text]
    }

def explain_feature(feature_name, value, shap_value, best_source_sentence, evidence):
    """Generate a human-readable explanation for a single feature."""
    # Similarity features
    if feature_name in ["semantic_similarity", "content_overlap_best", "content_overlap_top3",
                        "content_overlap_all", "bigram_overlap_best", "bm25_best_norm", "sim_top3_mean"]:
        if value > 0.6:
            return f"✅ High text similarity ({value:.2f}) — source: \"{best_source_sentence[:100]}...\""
        elif value > 0.3:
            return f"⚠️ Partial text similarity ({value:.2f}) — source only loosely matches."
        else:
            return f"❌ Low text similarity ({value:.2f}) — source does NOT contain this claim."

    # NLI features
    if feature_name == "nli_entail_top1":
        tag = "✅" if value > 0.7 else "⚠️"
        return f"{tag} Source entails claim (p={value:.2f})"
    if feature_name == "nli_contra_top1":
        tag = "🚨" if value > 0.5 else "✅"
        word = "CONTRADICTS" if value > 0.5 else "does NOT contradict"
        return f"{tag} Source {word} claim (p={value:.2f})"

    # Entity/Number features
    if feature_name in ["entity_match_ratio", "n_entities_unmatched"]:
        if evidence["missing_entities"]:
            return f"❌ Entity NOT in source: {', '.join(evidence['missing_entities'])}"
        elif evidence["matched_entities"]:
            return f"✅ Entities matched: {', '.join(evidence['matched_entities'])}"
        return "➖ No named entities."
        
    if feature_name in ["number_match_ratio", "unmatched_numbers_count", "n_numbers_unmatched"]:
        if evidence["missing_numbers"]:
            return f"❌ Fabricated number(s): {', '.join(evidence['missing_numbers'])}"
        return "✅ All numbers supported."

    if feature_name == "sentence_length":
        return f"ℹ️ {int(value)} words."

    return f"ℹ️ {feature_name} = {value:.2f}"

# ============================================================
# STEP 1: GENERATE QA PAIRS (Using Local Models)
# ============================================================
print("\n--- [1] Generating QA pairs from Malaysian News ---")
question_gen = OllamaLLM(model="qwen2.5:7b", temperature=0.7) 
answer_gen = OllamaLLM(model="llama3.2:latest", temperature=0.3)

df = pd.read_csv("./data/raw/malaysian_news/malaysian_news_clean.csv")
generated_data = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating QA"):
    article_text = row['text']
    article_url = row['url']
    
    # Truncate if too long
    if len(article_text.split()) > 800:
        article_text = " ".join(article_text.split()[:800])

    q_prompt = f"""Generate exactly 3 questions based ONLY on the facts in the article.
    Output in valid JSON list format: ["Question 1", "Question 2", "Question 3"]
    Article: {article_text}"""

    try:
        q_response = question_gen.invoke(q_prompt)
        match = re.search(r'\[.*\]', q_response.replace('\n', ' '))
        if not match: continue
        questions = json.loads(match.group(0))
    except Exception:
        continue

    for q in questions:
        a_prompt = f"""Answer the question based ONLY on the provided context. 
        If the context doesn't contain the answer, say "I don't know".
        Context: {article_text}
        Question: {q}
        Answer:"""
        
        try:
            answer = answer_gen.invoke(a_prompt).strip()
            generated_data.append({
                "id": f"news_{idx}",           # Unique ID for the article
                "source": row['source'],
                "url": article_url,
                "context": article_text,
                "question": q,
                "answer": answer
            })
        except Exception:
            pass

# Save the QA pairs
qa_df = pd.DataFrame(generated_data)
qa_df.to_json("./data/raw/malaysian_news/malaysian_news_qa.jsonl", orient="records", lines=True)
print(f"✅ Saved {len(qa_df)} QA pairs.")

# ============================================================
# STEP 2: FEATURE EXTRACTION (Looping over the generated data)
# ============================================================
print("\n--- [2] Extracting Features ---")
all_sentences_data = []

# Loop through every generated QA pair to build the sentence-level DataFrame
for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Extracting features"):
    answer = row['answer']
    context = row['context']
    doc_id = row['id']
    
    doc = nlp(answer)
    sentences = [(s.text.strip(), s.start_char, s.end_char) for s in doc.sents if s.text.strip()]
    
    for i, (sent, start, end) in enumerate(sentences):
        all_sentences_data.append({
            'id': f"{doc_id}_s{i}", 
            'source_id': doc_id, 
            'task_type': 'QA',
            'sentence_chunks': sent, 
            'start_char': start, 
            'end_char': end,
            'reference': context, 
            'labels': []
        })

detector_input = pd.DataFrame(all_sentences_data)

# Compute 5 BASE features
def calc_overlap(row):
    s_set = set(re.findall(r'\b\w+\b', str(row['sentence_chunks']).lower()))
    r_set = set(re.findall(r'\b\w+\b', str(row['reference']).lower()))
    return len(s_set & r_set) / len(s_set) if s_set else 0.0

def count_bad_numbers(row):
    s_nums = re.findall(r'\d+', str(row['sentence_chunks']).lower())
    r_nums = set(re.findall(r'\d+', str(row['reference']).lower()))
    return sum(1 for n in s_nums if n not in r_nums)

with open('./cache/tfidf_vectorizer_cache.pkl', 'rb') as f:
    ragtruth_tfidf = pickle.load(f)

def calc_tfidf(sentence, reference):
    s_vec = ragtruth_tfidf.transform([sentence])
    r_vec = ragtruth_tfidf.transform([reference])
    return float(s_vec.multiply(r_vec).sum())

# 1. Pre-compute reference embeddings ONCE
ref_sents = [s.text.strip() for s in nlp(context).sents if s.text.strip()] or [context]
ref_embs = embedder.encode(ref_sents, show_progress_bar=False)

# 2. Helper function using the pre-computed embeddings
def calc_semantic_sim(sentence):
    sent_emb = embedder.encode([sentence], show_progress_bar=False)
    return float(cosine_similarity(sent_emb, ref_embs).max())

# Apply BASE features
detector_input['sentence_length'] = detector_input['sentence_chunks'].str.split().str.len()
detector_input['word_overlap_ratio'] = detector_input.apply(calc_overlap, axis=1)
detector_input['unmatched_numbers_count'] = detector_input.apply(count_bad_numbers, axis=1)
detector_input['semantic_similarity'] = detector_input['sentence_chunks'].apply(calc_semantic_sim)
detector_input['tfidf_similarity'] = detector_input.apply(
    lambda r: calc_tfidf(r['sentence_chunks'], r['reference']), axis=1
)

# Add V2 features
features_v2 = build_features_v2(detector_input, nlp, embedder, use_nli=True)
detector_input = pd.concat([detector_input, features_v2], axis=1)

# ============================================================
# STEP 3: DETECTION
# ============================================================
print("\n--- [3] Running Detector ---")
bundle = joblib.load('./frozen/frozen_full.joblib')
model, feature_cols, threshold = bundle['model'], bundle['feature_cols'], bundle['threshold_sentence']

for col in feature_cols:
    if col not in detector_input.columns:
        detector_input[col] = 0.0

X = detector_input[feature_cols].values
probs = model.predict_proba(X)[:, 1]
detector_input['prob'] = probs
detector_input['is_hallucinated'] = (probs >= threshold).astype(int)

# Summary printout
print(f"Analyzed {len(detector_input)} sentences.")
print(f"Flagged {detector_input['is_hallucinated'].sum()} as hallucinated.")

# ============================================================
# STEP 4: XAI (SHAP + EXPLANATION) for the top flagged sentence
# ============================================================
print("\n--- [4] Generating XAI Explanations ---")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
sv = np.asarray(shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if np.asarray(shap_values).ndim == 3 else shap_values))

# Find the most hallucinated sentence
flagged_idx = int(np.argmax(probs))
row_dict = detector_input.iloc[flagged_idx].to_dict()
best_sent, best_sim = get_best_source_sentence(row_dict['sentence_chunks'], row_dict['reference'])
evidence = get_entity_number_evidence(row_dict['sentence_chunks'], row_dict['reference'])

contributions = sorted(
    [(feat, row_dict[feat], sv[flagged_idx][j]) for j, feat in enumerate(feature_cols)],
    key=lambda t: -abs(t[2])
)

print(f"\n{'='*70}")
print(f"TOP EXPLANATION FOR: \"{row_dict['sentence_chunks']}\"")
print(f"{'='*70}")
print(f"Verdict: {'🚨 HALLUCINATED' if probs[flagged_idx] >= threshold else '✅ FAITHFUL'} (p={probs[flagged_idx]:.3f})")
print(f"\nBest-matching source sentence:\n  \"{best_sent[:200]}...\"")
print("\nTop reasons:")
for feat, value, contrib in contributions[:5]:
    print(f"  • {explain_feature(feat, value, contrib, best_sent, evidence)}")