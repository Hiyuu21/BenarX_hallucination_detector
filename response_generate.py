# ============================================================
# IMPORTS & SETUP
# ============================================================
import os
import re
import pickle
import joblib
import numpy as np
import pandas as pd
import spacy
import shap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import custom feature builder from your project
from detector_v2 import build_features_v2


load_dotenv()
print(f"HF_TOKEN: {'✅ Set' if os.getenv('HF_TOKEN') else '❌ Not set'}")
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
# STEP 1: RAG PIPELINE (RETRIEVE & GENERATE)
# ============================================================
print("\n--- [1] Running RAG Pipeline ---")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./malaysian_news_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="llama3.2")

question = "Did the news mention anything about haze?"
relevant_docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in relevant_docs])

prompt = f"""Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

answer = llm.invoke(prompt)
print(f"Llama 3 Answer:\n{answer}\n")

# ============================================================
# STEP 2: FEATURE EXTRACTION
# ============================================================
print("--- [2] Extracting Features ---")
doc = nlp(answer)
sentences = [(s.text.strip(), s.start_char, s.end_char) for s in doc.sents if s.text.strip()]

# Build DataFrame for detector
detector_input = pd.DataFrame([
    {
        'id': f'news_{i}', 'source_id': 'malaysian_news_db', 'task_type': 'Summary',
        'sentence_chunks': sent, 'start_char': start, 'end_char': end,
        'reference': context, 'labels': []
    } for i, (sent, start, end) in enumerate(sentences)
])

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

# 3. Apply features to the DataFrame
detector_input['sentence_length'] = detector_input['sentence_chunks'].str.split().str.len()
detector_input['word_overlap_ratio'] = detector_input.apply(calc_overlap, axis=1)
detector_input['unmatched_numbers_count'] = detector_input.apply(count_bad_numbers, axis=1)
detector_input['semantic_similarity'] = detector_input['sentence_chunks'].apply(calc_semantic_sim)
detector_input['tfidf_similarity'] = detector_input.apply(
    lambda r: calc_tfidf(r['sentence_chunks'], r['reference']), axis=1
)

# Add v2 features
features_v2 = build_features_v2(detector_input, nlp, embedder, use_nli=True)
detector_input = pd.concat([detector_input, features_v2], axis=1)

# ============================================================
# STEP 3: DETECTION
# ============================================================
print("--- [3] Running Detector ---")
bundle = joblib.load('./frozen/frozen_full.joblib')
model, feature_cols, threshold = bundle['model'], bundle['feature_cols'], bundle['threshold_sentence']

# Safety alignment
for col in feature_cols:
    if col not in detector_input.columns:
        detector_input[col] = 0.0
print("Missing features: ✅ None" if all(c in detector_input.columns for c in feature_cols) else "⚠️ Features missing")

X = detector_input[feature_cols].values
probs = model.predict_proba(X)[:, 1]

for sent, p in zip(detector_input['sentence_chunks'], probs):
    flag = "🚨 HALLUCINATED" if p >= threshold else "✅ FAITHFUL"
    print(f"[{flag}] p={p:.3f} | {sent}")

response_flag = (probs.max() >= threshold)
print(f"\nRESPONSE-LEVEL: {'🚨 HALLUCINATED' if response_flag else '✅ FAITHFUL'} (max p={probs.max():.3f})\n")

# ============================================================
# STEP 4: XAI (SHAP + EXPLANATION)
# ============================================================
print("--- [4] Generating XAI Explanations ---")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Robustly handle SHAP shapes
sv = np.asarray(shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if np.asarray(shap_values).ndim == 3 else shap_values))
assert sv.ndim == 2 and sv.shape[0] == len(detector_input) and sv.shape[1] == len(feature_cols)

# Show explanation for the highest-confidence flag
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
print(f"  (similarity: {best_sim:.2f})")
print("\nTop reasons:")
for feat, value, contrib in contributions[:5]:
    explanation = explain_feature(feat, value, contrib, best_sent, evidence)
    print(f"  • {explanation}")

print(f"\n📋 SUMMARY: Strongest signal is '{contributions[0][0]}'.")