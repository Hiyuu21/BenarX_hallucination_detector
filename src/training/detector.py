# --------------------------------------
# Libraries
# --------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
import os
from tqdm import tqdm
import pickle

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import shap

from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --------------------------------------
# 0. Helper Functions
# --------------------------------------

nlp = spacy.load("en_core_web_sm")
tqdm.pandas(desc="Processing..")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_into_sentence(text):
    doc = nlp(text)
    return [(sent.text.strip(), sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()]

def split_dataset(data, split_type):
    dataset = data[(data['split'] == split_type) & (data['quality']=='good')]
    return dataset

def merge_response(response, source):
    merged = pd.merge(response, source, on = 'source_id', how = 'inner')
    return merged

def parse_rows(merged_df):
    parsed_set = []
    for idx, row in merged_df.iterrows():
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
    return parsed_set

def get_data(responses, sources):
    train = split_dataset(responses, 'train')
    merged_train = merge_response(train, sources)

    test = split_dataset(responses, 'test')
    merged_test = merge_response(test, sources)

    return parse_rows(merged_train), parse_rows(merged_test)

def calculate_overlap(row):
    sentence = str(row['sentence_chunks']).lower()
    reference = str(row['reference']).lower()

    sentence_set = set(re.findall(r'\b\w+\b', str(row['sentence_chunks']).lower()))
    reference_set = set(re.findall(r'\b\w+\b', str(row['reference']).lower()))

    intersection = sentence_set & reference_set
    return len(intersection)/ len(sentence_set) if len(sentence_set) > 0 else 0.0

def count_unmatched_numbers(row):
    sentence = str(row['sentence_chunks']).lower()
    reference = str(row['reference']).lower()

    sentence_nums = re.findall(r'\d+', sentence)
    reference_nums = set(re.findall(r'\d+', reference))

    counter = 0
    for num in sentence_nums:
        if num not in reference_nums:
            counter += 1

    return counter

def precompute_source_embeddings(df):
    unique_refs = df[['source_id', 'reference']].drop_duplicates()
    ref_emb_dict = {}

    for _, row in unique_refs.iterrows():
        sid = row['source_id']
        ref_text = str(row['reference'])
        ref_doc = nlp(ref_text)
        ref_sents = [s.text.strip() for s in ref_doc.sents if s.text.strip()]

        if not ref_sents:
            ref_sents = [ref_text]

        ref_emb_dict[sid] = embedder.encode(ref_sents, show_progress_bar=False, batch_size=64)

    return ref_emb_dict

def compute_batch_cosine_similarity(df):
    ref_emb_dict = precompute_source_embeddings(df)
    
    sentence_list = df['sentence_chunks'].astype(str).tolist()
    sent_embs = embedder.encode(sentence_list, show_progress_bar=True, batch_size=128)

    # Pre-allocate array to GUARANTEE shape matching
    similarities = np.zeros(len(df))
    
    for idx, row in enumerate(df.itertuples()):
        s_emb = sent_embs[idx].reshape(1, -1)
        r_embs = ref_emb_dict.get(row.source_id)

        if r_embs is not None and len(r_embs) > 0:
            sims = cosine_similarity(s_emb, r_embs)
            similarities[idx] = float(np.max(sims))
            
    return similarities

def compute_tfidf_overlap(df):
    sent_vecs = global_tfidf.transform(df['sentence_chunks'].astype(str))
    ref_vecs = global_tfidf.transform(df['reference'].astype(str))
    
    # Efficient row-wise sparse dot product (cosine similarity since L2-normalized)
    sims = sent_vecs.multiply(ref_vecs).sum(axis=1)
    return np.asarray(sims).ravel()

def tag_hallucination(row):
    s_start = row['start_char']
    s_end = row['end_char']
    labels = row['labels']

    for l in labels:
        l_start = l['start']
        l_end = l['end']
        if max(s_start, l_start) < min(s_end, l_end):
            return 1
    return 0

# ------------------------------------------------
# 1. Import RAGTruth Dataset & Chunk the response
# ------------------------------------------------

RPATH = 'C:/Users/jinxl/Downloads/RAG Hallucination Detection/detector/RAGTruth/dataset/response.jsonl' 
SPATH= 'C:/Users/jinxl/Downloads/RAG Hallucination Detection/detector/RAGTruth/dataset/source_info.jsonl'

train_cache = './exploded_train_cache.pkl'
test_cache = './exploded_test_cache.pkl'
tfidf_cache = './tfidf_vectorizer_cache.pkl'

if os.path.exists(train_cache) and os.path.exists(test_cache) and os.path.exists(tfidf_cache):
    print("Found cached data! Bypassing spaCy chunking and loading directly...")
    exploded_train = pd.read_pickle(train_cache)
    exploded_test = pd.read_pickle(test_cache)

    with open(tfidf_cache, 'rb') as f:
        global_tfidf = pickle.load(f)

else:
    print("No cache found. Running full chunking and feature extraction pipeline...")
    
    responses = pd.read_json(RPATH, lines=True)
    sources = pd.read_json(SPATH, lines=True)

    train_set, test_set = get_data(responses, sources)

    train_set = pd.DataFrame(train_set)
    test_set = pd.DataFrame(test_set)

    random.seed(2024)
    dev_source_id = []
    for task in['QA', 'Summary', 'Data2txt']:
        source_ids = train_set[train_set['task_type'] == task]['source_id'].unique().tolist()
        dev_source_id.extend(random.sample(source_ids, 50))

    dev_set = train_set[train_set['source_id'].isin(dev_source_id)]
    train_set = train_set[~train_set['source_id'].isin(dev_source_id)].reset_index(drop=True)

    train_set.to_json('./train.jsonl', orient='records', lines=True)
    test_set.to_json('./test.jsonl', orient='records', lines=True)
    dev_set.to_json('./dev.jsonl', orient='records', lines=True)

    train_set['sentence_chunks'] = train_set['response'].progress_apply(chunk_into_sentence)
    exploded_train = train_set.explode('sentence_chunks').dropna(subset=['sentence_chunks']).reset_index(drop=True)
    exploded_train[['sentence_chunks', 'start_char', 'end_char']] = pd.DataFrame(
        exploded_train['sentence_chunks'].tolist(), index=exploded_train.index
    )

    test_set['sentence_chunks'] = test_set['response'].progress_apply(chunk_into_sentence)
    exploded_test = test_set.explode('sentence_chunks').dropna(subset=['sentence_chunks']).reset_index(drop=True)
    exploded_test[['sentence_chunks', 'start_char', 'end_char']] = pd.DataFrame(
        exploded_test['sentence_chunks'].tolist(), index=exploded_test.index
    )


    # ------------------------------------------------
    # 2. Add in new features
    # ------------------------------------------------

    unique_train_sources = train_set['reference'].astype(str).unique().tolist()
    global_tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    global_tfidf.fit(unique_train_sources)

    exploded_train['sentence_length'] = exploded_train['sentence_chunks'].str.split().str.len()
    exploded_train['word_overlap_ratio'] = exploded_train.apply(calculate_overlap, axis=1)
    exploded_train['unmatched_numbers_count'] = exploded_train.apply(count_unmatched_numbers, axis=1)
    exploded_train['semantic_similarity'] = compute_batch_cosine_similarity(exploded_train)
    exploded_train['tfidf_similarity'] = compute_tfidf_overlap(exploded_train)
    exploded_train['is_hallucination'] = exploded_train.apply(tag_hallucination, axis=1)

    exploded_test['sentence_length'] = exploded_test['sentence_chunks'].str.split().str.len()
    exploded_test['word_overlap_ratio'] = exploded_test.apply(calculate_overlap, axis=1)
    exploded_test['unmatched_numbers_count'] = exploded_test.apply(count_unmatched_numbers, axis=1)
    exploded_test['semantic_similarity'] = compute_batch_cosine_similarity(exploded_test)
    exploded_test['tfidf_similarity'] = compute_tfidf_overlap(exploded_test)
    exploded_test['is_hallucination'] = exploded_test.apply(tag_hallucination, axis=1)


    print(exploded_train[['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count']].describe())
    print()
    print(exploded_test[['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count']].describe())
    print()
    
    # Save the finished DataFrames for next time
    print("Saving processed data to cache...")
    exploded_train.to_pickle(train_cache)
    exploded_test.to_pickle(test_cache)

    with open(tfidf_cache, 'wb') as f:
        pickle.dump(global_tfidf, f)

# ------------------------------------------------
# 3. Model Training
# ------------------------------------------------
print("\n --- Data loaded for model training...")

models = {
    'Dummy (Baseline)': DummyClassifier(strategy='prior'),
    'Decision Tree (Pruned)': DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight='balanced', random_state=2024),
    'Logistic Regression (Scaled)': make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000)),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=2024, n_jobs=-1),
    'XGBoost': XGBClassifier(scale_pos_weight=8, random_state=2024, n_jobs=-1)
}
        
feature_cols = ['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count', 'semantic_similarity', 'tfidf_similarity']
X = exploded_train[feature_cols]
Y = exploded_train['is_hallucination']
groups = exploded_train['source_id']

gkf = GroupKFold(n_splits=5)

global_best_model_name = 'model'
global_best_f1 = 0.0
global_best_threshold = 0.5

for model_name, model in models.items():
    print(f"\n --- Evaluating {model_name}.")

    probabilities = np.zeros(len(X))
    targets = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, Y, groups=groups)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        Y_train_fold, Y_val_fold = Y.iloc[train_idx], Y.iloc[val_idx]        
        model.fit(X_train_fold, Y_train_fold)

        if model_name == 'Dummy (Baseline)':
            probabilities[val_idx] = 0.0
        else:
            probabilities[val_idx] = model.predict_proba(X_val_fold)[:,1]

        targets[val_idx] = Y_val_fold

    best_threshold = 0.5
    best_f1 = 0

    if model_name != 'Dummy (Baseline)':
        for i in np.arange(0.01, 1.0, 0.01):
            predictions_temp = (probabilities >= i).astype(int)
            current_f1 = f1_score(targets, predictions_temp, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = i

    if best_f1 > global_best_f1:
        global_best_f1 = best_f1
        global_best_model_name = model_name
        global_best_threshold = best_threshold

    best_predictions = (probabilities >= best_threshold).astype(int)

    print(f"Optimal Threshold:   {best_threshold:.2f}")
    print(f"Precision:           {precision_score(targets, best_predictions, zero_division=0):.4f}")
    print(f"Recall:              {recall_score(targets, best_predictions, zero_division=0):.4f}")
    print(f"F1-Score:            {best_f1:.4f}")
    print(f"ROC-AUC Score:       {roc_auc_score(targets, probabilities):.4f}")
    print(f"PR-AUC (Avg Prec):   {average_precision_score(targets, probabilities):.4f}")

    matrix = confusion_matrix(targets, best_predictions)    
    print("\n ---> F: Fact  H: Hallucination")
    print("         Predicted F | Predicted H")
    print(f"Actual F: {matrix[0][0]:<10d}|{matrix[0][1]:<10d}")
    print(f"Actual H: {matrix[1][0]:<10d}|{matrix[1][1]:<10d}\n")

print(f"\n --- Best model: {global_best_model_name} | F1: {global_best_f1:.4f} at threshold {global_best_threshold:.4f}")

print("\n=== SINGLE-FEATURE BASELINE DIAGNOSTICS ===")
for col in feature_cols:
    # Use negative values if higher feature value implies lower hallucination risk
    score = roc_auc_score(Y, -X[col] if 'overlap' in col or 'similarity' in col else X[col])
    print(f"Feature: {col:<25} | Individual AUROC: {score:.4f}")

# ------------------------------------------------
# 4. Final Test 
# ------------------------------------------------
print("\n --- Model Testing.")

feature_sets = {
    'Base Only': ['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count'],
    'Base + TF-IDF': ['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count', 'tfidf_similarity'],
    'Base + Semantics': ['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count', 'semantic_similarity'],
    'All Features': ['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count', 'tfidf_similarity', 'semantic_similarity']
}

best_model = models[global_best_model_name]
Y_train_full = exploded_train['is_hallucination']
Y_test_full = exploded_test['is_hallucination']

for set_name, columns in feature_sets.items():
    print(f"\n--- Evaluating: {set_name} ---")

    X_train_abl = exploded_train[columns]
    X_test_abl = exploded_test[columns]

    best_model.fit(X_train_abl, Y_train_full)
    test_probs = best_model.predict_proba(X_test_abl)[:, 1]
    
    test_predictions = (test_probs >= global_best_threshold).astype(int)

    test_prec = precision_score(Y_test_full, test_predictions, zero_division=0)
    test_rec = recall_score(Y_test_full, test_predictions, zero_division=0)
    test_f1 = f1_score(Y_test_full, test_predictions, zero_division=0)
    test_auc = roc_auc_score(Y_test_full, test_probs)
    test_pr_auc = average_precision_score(Y_test_full, test_probs)

    print(f"Test ROC-AUC:  {test_auc:.4f}")
    print(f"Test PR-AUC:   {test_pr_auc:.4f}")
    print(f"Test Precision:{test_prec:.4f}")
    print(f"Test Recall:   {test_rec:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")


# ------------------------------------------------
# 5. Reporting: Task, Response, and Baselines 
# ------------------------------------------------
exploded_test['pred_prob'] = test_probs
exploded_test['pred_label'] = test_predictions
# Uncovered fraction baseline (1 - overlap)
exploded_test['baseline_prob'] = 1.0 - exploded_test['word_overlap_ratio']

# 1. Per-Task Sentence-Level
for task in ['QA', 'Summary', 'Data2txt']:
    subset = exploded_test[exploded_test['task_type'] == task]
    auc = roc_auc_score(subset['is_hallucination'], subset['pred_prob'])
    base_auc = roc_auc_score(subset['is_hallucination'], subset['baseline_prob'])
    print(f"Sentence | Task: {task:<8} | Model AUROC: {auc:.4f} | Baseline AUROC: {base_auc:.4f}")

# 2. Response-Level Aggregation
response_df = exploded_test.groupby('id').agg({
    'is_hallucination': 'max',  # 1 if ANY sentence is hallucinated
    'pred_prob': 'max',         # Highest sentence hallucination probability
    'baseline_prob': 'max',     # Highest sentence baseline risk
    'task_type': 'first'
}).reset_index()

# 3. Overall Response-Level Metrics
resp_auc = roc_auc_score(response_df['is_hallucination'], response_df['pred_prob'])
resp_base_auc = roc_auc_score(response_df['is_hallucination'], response_df['baseline_prob'])

# Recalculate discrete labels using the global threshold
response_df['pred_label'] = (response_df['pred_prob'] >= global_best_threshold).astype(int)
resp_f1 = f1_score(response_df['is_hallucination'], response_df['pred_label'], zero_division=0)

print(f"\nResponse | OVERALL      | Model AUROC: {resp_auc:.4f} | Baseline AUROC: {resp_base_auc:.4f}")
print(f"Response | Overall F1   : {resp_f1:.4f} (at global threshold {global_best_threshold:.2f})")


# ------------------------------------------------------
# 6. XAI Features - SHAP on current best model (XGBoost)
# ------------------------------------------------------

explainer = shap.TreeExplainer(best_model)
X_test_sample = X_test_abl.sample(n=500, random_state=2024)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample, feature_names=X_test_abl.columns)