# --------------------------------------
# Libraries
# --------------------------------------

import pandas as pd
import numpy as np
import re
import random
import os
from tqdm import tqdm
import pickle
import math
from collections import Counter

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import shap
from scipy.sparse import csc_matrix

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

# --------------------------------------
# Feature groups
# --------------------------------------
BASE = ['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count',
        'semantic_similarity', 'tfidf_similarity']
 
LEXICAL_BEST = ['content_overlap_best', 'content_overlap_top3', 'content_overlap_all',
                'bigram_overlap_best', 'bm25_best_norm', 'sim_top3_mean']
 
ENTITY_NUMBER = ['n_entities', 'n_entities_unmatched', 'entity_match_ratio',
                 'n_numbers', 'n_numbers_unmatched', 'number_match_ratio']
 
CONTEXT = ['is_QA', 'is_Summary', 'is_Data2txt',
           'sent_pos_rel', 'n_sents_resp', 'resp_len_words']
 
NLI = ['nli_entail_top1', 'nli_contra_top1',
       'nli_entail_ctx', 'nli_contra_ctx', 'nli_neutral_ctx']
 
TOKEN_RE = re.compile(r"\b\w+\b")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
SKIP_ENT_LABELS = {"CARDINAL", "ORDINAL"}


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
    return data[(data['split'] == split_type) & (data['quality']=='good')].copy()

def merge_response(response, source):
    return pd.merge(response, source, on='source_id', how='inner')

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
    sentence_set = set(re.findall(r'\b\w+\b', str(row['sentence_chunks']).lower()))
    reference_set = set(re.findall(r'\b\w+\b', str(row['reference']).lower()))
    intersection = sentence_set & reference_set
    return len(intersection)/ len(sentence_set) if len(sentence_set) > 0 else 0.0

def count_unmatched_numbers(row):
    sentence_nums = re.findall(r'\d+', str(row['sentence_chunks']).lower())
    reference_nums = set(re.findall(r'\d+', str(row['reference']).lower()))
    counter = sum(1 for num in sentence_nums if num not in reference_nums)
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
    similarities = np.zeros(len(df))
    for idx, row in enumerate(df.itertuples()):
        s_emb = sent_embs[idx].reshape(1, -1)
        r_embs = ref_emb_dict.get(row.source_id)
        if r_embs is not None and len(r_embs) > 0:
            sims = cosine_similarity(s_emb, r_embs)
            similarities[idx] = float(np.max(sims))
    return similarities

def compute_tfidf_overlap(df, vectorizer):
    sent_vecs = vectorizer.transform(df['sentence_chunks'].astype(str))
    ref_vecs = vectorizer.transform(df['reference'].astype(str))
    sims = sent_vecs.multiply(ref_vecs).sum(axis=1)
    return np.asarray(sims).ravel()

def tag_hallucination(row):
    s_start, s_end = row['start_char'], row['end_char']
    for l in row['labels']:
        if max(s_start, l['start']) < min(s_end, l['end']):
            return 1
    return 0

def content_tokens(text):
    return [t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOP_WORDS]
 
def _norm_num(s):
    v = float(s)
    return str(int(v)) if v.is_integer() else repr(v)
 
def extract_numbers(text):
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', str(text))
    return [_norm_num(x) for x in NUM_RE.findall(t)]
 
def _units_from_doc(doc, max_words=60, overlap=10):
    units = []
    for sent in doc.sents:
        words = sent.text.split()
        if not words: continue
        if len(words) <= max_words:
            units.append(" ".join(words))
        else:
            step = max_words - overlap
            for s in range(0, len(words), step):
                units.append(" ".join(words[s:s + max_words]))
                if s + max_words >= len(words): break
    return units or [(doc.text.strip() or " ")]
 
def build_source_index(df, nlp, embedder, max_words=60):
    uniq = df[['source_id', 'reference']].drop_duplicates('source_id')
    ids, texts = uniq['source_id'].tolist(), [str(t) for t in uniq['reference']]
    units_per_source, all_units = {}, []
    docs = nlp.pipe(texts, batch_size=32, disable=["ner", "lemmatizer"])
    for sid, doc in tqdm(zip(ids, docs), total=len(ids), desc="Splitting sources"):
        units = _units_from_doc(doc, max_words)
        units_per_source[sid] = units
        all_units.extend(units)
    emb_all = embedder.encode(all_units, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
 
    index, offset = {}, 0
    for sid, text in zip(ids, texts):
        units = units_per_source[sid]
        emb = emb_all[offset:offset + len(units)]
        offset += len(units)
        tokens_per_unit = [content_tokens(u) for u in units]
        vocab, rows, cols, vals, df_counter = {}, [], [], [], Counter()
        for r, toks in enumerate(tokens_per_unit):
            for term, c in Counter(toks).items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(r); cols.append(j); vals.append(c)
                df_counter[term] += 1
        tf = csc_matrix((vals, (rows, cols)), shape=(len(units), max(len(vocab), 1)), dtype=np.float32)
        dl = np.array([len(t) for t in tokens_per_unit], dtype=float)
        unit_tokset = [set(t) for t in tokens_per_unit]
        index[sid] = {
            'units': units, 'emb': emb, 'unit_tokset': unit_tokset,
            'unit_bigrams': [set(zip(t, t[1:])) for t in tokens_per_unit],
            'tok_all': set().union(*unit_tokset), 'lower_text': text.lower(),
            'numbers': set(extract_numbers(text)), 'vocab': vocab, 'df': df_counter, 'tf': tf,
            'dl': dl, 'avgdl': float(dl.mean()) if dl.mean() > 0 else 1.0,
        }
    return index
 
def bm25_best_norm(q_terms, idx, k1=1.5, b=0.75):
    if not q_terms: return 0.0
    n_units = len(idx['units'])
    max_possible, cols, idfs = 0.0, [], []
    for t in q_terms:
        n = idx['df'].get(t, 0)
        idf = math.log(1 + (n_units - n + 0.5) / (n + 0.5))
        max_possible += idf * (k1 + 1)
        j = idx['vocab'].get(t)
        if j is not None:
            cols.append(j)
            idfs.append(idf)
    if not cols or max_possible == 0: return 0.0
    tf = idx['tf'][:, cols].toarray()
    denom = tf + k1 * (1 - b + b * idx['dl'][:, None] / idx['avgdl'])
    score = ((tf * (k1 + 1)) / denom * np.array(idfs)).sum(axis=1)
    return float(score.max() / max_possible)
 
def _entity_matched(ent, idx):
    if ent in idx['lower_text']: return True
    toks = content_tokens(ent)
    return bool(toks) and all(t in idx['tok_all'] for t in toks)

def compute_lexical_features(df, index, nlp, embedder):
    sentences = df['sentence_chunks'].astype(str).tolist()
    source_ids = df['source_id'].tolist()
    sent_emb = embedder.encode(sentences, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    ents_per_sentence = [
        [e.text.lower() for e in doc.ents if e.label_ not in SKIP_ENT_LABELS]
        for doc in tqdm(nlp.pipe(sentences, batch_size=256, disable=["parser", "lemmatizer"]),
                        total=len(sentences), desc="NER on sentences")
    ]
    feats, evidence = [], []
    for i, (sid, sent) in enumerate(tqdm(zip(source_ids, sentences), total=len(sentences), desc="Lexical features")):
        idx = index[sid]
        sims = idx['emb'] @ sent_emb[i]
        order = np.argsort(-sims)
        best, top3 = order[0], order[:3]
        toks = content_tokens(sent)
        tset = set(toks)
        n_tok = len(tset)
        bigrams = set(zip(toks, toks[1:]))
        ov_best = len(tset & idx['unit_tokset'][best]) / n_tok if n_tok else 0.0
        top3_tokens = set().union(*[idx['unit_tokset'][j] for j in top3])
        f = {
            'sim_top3_mean': float(sims[top3].mean()),
            'content_overlap_best': ov_best,
            'content_overlap_top3': len(tset & top3_tokens) / n_tok if n_tok else 0.0,
            'content_overlap_all': len(tset & idx['tok_all']) / n_tok if n_tok else 0.0,
            'bigram_overlap_best': (len(bigrams & idx['unit_bigrams'][best]) / len(bigrams) if bigrams else ov_best),
            'bm25_best_norm': bm25_best_norm(tset, idx),
        }
        ents = ents_per_sentence[i]
        n_unmatched = sum(0 if _entity_matched(e, idx) else 1 for e in ents)
        f['n_entities'] = len(ents)
        f['n_entities_unmatched'] = n_unmatched
        f['entity_match_ratio'] = 1.0 - n_unmatched / len(ents) if ents else 1.0
        nums = extract_numbers(sent)
        n_bad = sum(1 for x in nums if x not in idx['numbers'])
        f['n_numbers'] = len(nums)
        f['n_numbers_unmatched'] = n_bad
        f['number_match_ratio'] = 1.0 - n_bad / len(nums) if nums else 1.0
        feats.append(f)
        evidence.append({'evidence_top1': idx['units'][best], 'evidence_top3': " ".join(idx['units'][j] for j in sorted(top3))})
    return pd.DataFrame(feats, index=df.index), pd.DataFrame(evidence, index=df.index)
 
def compute_context_features(df):
    assert 'id' in df.columns, "need the response 'id' column (one id per response)"
    out = pd.DataFrame(index=df.index)
    for t in ['QA', 'Summary', 'Data2txt']:
        out[f'is_{t}'] = (df['task_type'] == t).astype(int)
    grp = df.groupby('id')
    size = grp['id'].transform('size')
    pos = grp.cumcount()
    out['sent_pos_rel'] = np.where(size > 1, pos / (size - 1).clip(lower=1), 0.0)
    out['n_sents_resp'] = size
    words = df['sentence_chunks'].astype(str).str.split().str.len()
    out['resp_len_words'] = words.groupby(df['id']).transform('sum')
    return out
 
def _nli_probs(premises, hypotheses, tok, model, device, batch_size, max_length):
    import torch
    order = np.argsort([len(p) + len(h) for p, h in zip(premises, hypotheses)])
    out = np.zeros((len(premises), model.config.num_labels), dtype=np.float32)
    use_amp = (device == 'cuda')
    for start in tqdm(range(0, len(order), batch_size), desc="NLI batches"):
        ids = order[start:start + batch_size]
        enc = tok([premises[k] for k in ids], [hypotheses[k] for k in ids],
                  truncation='only_first', max_length=max_length,
                  padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits if not use_amp else torch.autocast(device_type='cuda', dtype=torch.float16)(lambda: model(**enc).logits)()
        out[ids] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out
 
def compute_nli_features(evidence_df, hypotheses, model_name="cross-encoder/nli-deberta-v3-small", batch_size=32, max_length=320, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
    names = {int(k): v.lower() for k, v in model.config.id2label.items()}
    try:
        i_ent = [k for k, v in names.items() if 'entail' in v][0]
        i_con = [k for k, v in names.items() if 'contradict' in v][0]
        i_neu = [k for k, v in names.items() if 'neutral' in v][0]
    except IndexError:
        i_con, i_ent, i_neu = 0, 1, 2
    hyps = [" ".join(str(h).split()[:120]) for h in hypotheses]
    p1 = _nli_probs(evidence_df['evidence_top1'].tolist(), hyps, tok, model, device, batch_size, max_length)
    p3 = _nli_probs(evidence_df['evidence_top3'].tolist(), hyps, tok, model, device, batch_size, max_length)
    return pd.DataFrame({
        'nli_entail_top1': p1[:, i_ent], 'nli_contra_top1': p1[:, i_con],
        'nli_entail_ctx': p3[:, i_ent], 'nli_contra_ctx': p3[:, i_con], 'nli_neutral_ctx': p3[:, i_neu],
    }, index=evidence_df.index)
 
def build_features_v2(df, nlp, embedder, use_nli=True, nli_model="cross-encoder/nli-deberta-v3-small"):
    index = build_source_index(df, nlp, embedder)
    lexical, evidence = compute_lexical_features(df, index, nlp, embedder)
    parts = [lexical, compute_context_features(df)]
    if use_nli:
        parts.append(compute_nli_features(evidence, df['sentence_chunks'].astype(str).tolist(), model_name=nli_model))
    return pd.concat(parts, axis=1)
 
def load_or_build_v2(df, cache_path, nlp, embedder, **kwargs):
    if os.path.exists(cache_path):
        print(f"Loading cached v2 features from {cache_path}")
        new = pd.read_pickle(cache_path)
    else:
        new = build_features_v2(df, nlp, embedder, **kwargs)
        new.to_pickle(cache_path)
    return pd.concat([df.drop(columns=[c for c in new.columns if c in df.columns]), new], axis=1)
 
def cv_evaluate(model, X, y, groups, seeds=(42, 2024, 999), n_splits=5):
    y = np.asarray(y)
    rows = []
    best_oof = None
    for seed in seeds:
        oof = np.zeros(len(y))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, va in cv.split(X, y, groups):
            m = clone(model).fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        
        if seed == seeds[0]:
            best_oof = oof.copy()
            
        thresholds = np.arange(0.01, 1.0, 0.01)
        f1s = [f1_score(y, (oof >= t).astype(int), zero_division=0) for t in thresholds]
        rows.append({'auroc': roc_auc_score(y, oof),
                     'pr_auc': average_precision_score(y, oof),
                     'best_f1': max(f1s),
                     'best_thr': thresholds[int(np.argmax(f1s))]})
    res = pd.DataFrame(rows)
    return res.mean().to_dict(), res.std().to_dict(), best_oof

# ------------------------------------------------
# 1. Import RAGTruth Dataset & Chunk the response
# ------------------------------------------------
RPATH = 'C:/Users/jinxl/Downloads/RAG Hallucination Detection/detector/RAGTruth/dataset/response.jsonl' 
SPATH= 'C:/Users/jinxl/Downloads/RAG Hallucination Detection/detector/RAGTruth/dataset/source_info.jsonl'

train_cache = './exploded_train_cache.pkl'
test_cache = './exploded_test_cache.pkl'
v2_train_cache = './v2_train_features.pkl'
v2_test_cache = './v2_test_features.pkl'
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
    train_set, test_set = pd.DataFrame(train_set), pd.DataFrame(test_set)

    random.seed(2024)
    dev_source_id = []
    for task in['QA', 'Summary', 'Data2txt']:
        source_ids = train_set[train_set['task_type'] == task]['source_id'].unique().tolist()
        dev_source_id.extend(random.sample(source_ids, 50))

    dev_set = train_set[train_set['source_id'].isin(dev_source_id)]
    train_set = train_set[~train_set['source_id'].isin(dev_source_id)].reset_index(drop=True)

    train_set['sentence_chunks'] = train_set['response'].progress_apply(chunk_into_sentence)
    exploded_train = train_set.explode('sentence_chunks').dropna(subset=['sentence_chunks']).reset_index(drop=True)
    exploded_train[['sentence_chunks', 'start_char', 'end_char']] = pd.DataFrame(exploded_train['sentence_chunks'].tolist(), index=exploded_train.index)

    test_set['sentence_chunks'] = test_set['response'].progress_apply(chunk_into_sentence)
    exploded_test = test_set.explode('sentence_chunks').dropna(subset=['sentence_chunks']).reset_index(drop=True)
    exploded_test[['sentence_chunks', 'start_char', 'end_char']] = pd.DataFrame(exploded_test['sentence_chunks'].tolist(), index=exploded_test.index)

    unique_train_sources = train_set['reference'].astype(str).unique().tolist()
    global_tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    global_tfidf.fit(unique_train_sources)

    exploded_train['sentence_length'] = exploded_train['sentence_chunks'].str.split().str.len()
    exploded_train['word_overlap_ratio'] = exploded_train.apply(calculate_overlap, axis=1)
    exploded_train['unmatched_numbers_count'] = exploded_train.apply(count_unmatched_numbers, axis=1)
    exploded_train['semantic_similarity'] = compute_batch_cosine_similarity(exploded_train)
    exploded_train['tfidf_similarity'] = compute_tfidf_overlap(exploded_train, global_tfidf)
    exploded_train['is_hallucination'] = exploded_train.apply(tag_hallucination, axis=1)

    exploded_test['sentence_length'] = exploded_test['sentence_chunks'].str.split().str.len()
    exploded_test['word_overlap_ratio'] = exploded_test.apply(calculate_overlap, axis=1)
    exploded_test['unmatched_numbers_count'] = exploded_test.apply(count_unmatched_numbers, axis=1)
    exploded_test['semantic_similarity'] = compute_batch_cosine_similarity(exploded_test)
    exploded_test['tfidf_similarity'] = compute_tfidf_overlap(exploded_test, global_tfidf)
    exploded_test['is_hallucination'] = exploded_test.apply(tag_hallucination, axis=1)

    print("\n --- Saving processed data to cache...")
    exploded_train.to_pickle(train_cache)
    exploded_test.to_pickle(test_cache)
    with open(tfidf_cache, 'wb') as f:
        pickle.dump(global_tfidf, f)

# ------------------------------------------------
# 2b. V2 features (computed once, then loaded from cache)
# ------------------------------------------------
exploded_train = load_or_build_v2(exploded_train, v2_train_cache, nlp, embedder)
exploded_test  = load_or_build_v2(exploded_test,  v2_test_cache,  nlp, embedder)

all_cols = BASE + LEXICAL_BEST + ENTITY_NUMBER + CONTEXT + NLI

print("\n --- Feature Correlation Check ---")
corr = exploded_train['word_overlap_ratio'].corr(exploded_train['content_overlap_all'])
print(f"Correlation between word_overlap_ratio and content_overlap_all: {corr:.4f}")
if corr > 0.95:
    print("Dropping 'content_overlap_all' due to high collinearity.")
    all_cols.remove('content_overlap_all')
    LEXICAL_BEST.remove('content_overlap_all')

# ------------------------------------------------
# 3. Model Selection
# ------------------------------------------------
print("\n --- Model Selection (3 Seeds) ---")
X = exploded_train[all_cols]
Y = exploded_train['is_hallucination']
groups = exploded_train['source_id']

candidates = {
    'Random Forest (Pruned)': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=10, class_weight='balanced', n_jobs=-1),
    'XGBoost': XGBClassifier(scale_pos_weight=8, max_depth=6, random_state=2024, n_jobs=-1)
}

best_model_name = None
best_model_auc = 0.0
best_oof_probs = None
best_sentence_thr = 0.5

for name, clf in candidates.items():
    mean_metrics, std_metrics, oof = cv_evaluate(clf, X, Y, groups)
    print(f"{name:<25} | AUROC: {mean_metrics['auroc']:.4f}±{std_metrics['auroc']:.4f} | PR-AUC: {mean_metrics['pr_auc']:.4f} | F1: {mean_metrics['best_f1']:.4f}")
    
    if mean_metrics['auroc'] > best_model_auc:
        best_model_auc = mean_metrics['auroc']
        best_model_name = name
        best_oof_probs = oof
        best_sentence_thr = mean_metrics['best_thr']

print(f"\nChampion Model: {best_model_name} (AUROC: {best_model_auc:.4f})")

print("\n --- Tuning Response-Level Threshold (OOF) ---")
exploded_train['oof_prob'] = best_oof_probs
resp_train = exploded_train.groupby('id').agg({
    'is_hallucination': 'max',
    'oof_prob': 'max'
}).reset_index()

best_resp_thr = 0.5
best_resp_f1 = 0
for t in np.arange(0.01, 1.0, 0.01):
    f1 = f1_score(resp_train['is_hallucination'], (resp_train['oof_prob'] >= t).astype(int), zero_division=0)
    if f1 > best_resp_f1:
        best_resp_f1 = f1
        best_resp_thr = t
print(f"Optimal Response-Level Threshold: {best_resp_thr:.2f} (OOF F1: {best_resp_f1:.4f})")
print(f"Train Response-Level Base Rate: {resp_train['is_hallucination'].mean():.4f}")

# ------------------------------------------------
# 4. Leave-One-Out Ablation Test
# ------------------------------------------------
print("\n --- Ablation Study (Leave-One-Out) ---")

experiments = {
    'All Features': all_cols,
    'No NLI': [c for c in all_cols if c not in NLI],
    'No Context': [c for c in all_cols if c not in CONTEXT],
    'No Entity/Num': [c for c in all_cols if c not in ENTITY_NUMBER],
    'No Lexical': [c for c in all_cols if c not in LEXICAL_BEST],
    'Evidence Only (No NLI/Context)': BASE + LEXICAL_BEST + ENTITY_NUMBER
}

for name, cols in experiments.items():
    mean, std, _ = cv_evaluate(candidates[best_model_name], exploded_train[cols], Y, groups)
    print(f"{name:<30} AUROC {mean['auroc']:.4f}±{std['auroc']:.4f} | PR-AUC {mean['pr_auc']:.4f} | F1 {mean['best_f1']:.4f}")


# ------------------------------------------------
# 5. Reporting: Test Set Metrics
# ------------------------------------------------
print("\n=== FINAL REPORT: TEST SET METRICS ===")
final_model = clone(candidates[best_model_name])
final_model.fit(exploded_train[all_cols], Y)
test_probs = final_model.predict_proba(exploded_test[all_cols])[:, 1]
test_predictions = (test_probs >= best_sentence_thr).astype(int)

exploded_test['pred_prob'] = test_probs
exploded_test['pred_label'] = test_predictions
exploded_test['baseline_prob'] = 1.0 - exploded_test['word_overlap_ratio']

test_auc = roc_auc_score(exploded_test['is_hallucination'], test_probs)
test_pr = average_precision_score(exploded_test['is_hallucination'], test_probs)
test_f1 = f1_score(exploded_test['is_hallucination'], test_predictions, zero_division=0)
print(f"Sentence | OVERALL      | Model AUROC: {test_auc:.4f} | PR-AUC: {test_pr:.4f} | F1: {test_f1:.4f}")

for task in ['QA', 'Summary', 'Data2txt']:
    subset = exploded_test[exploded_test['task_type'] == task]
    auc = roc_auc_score(subset['is_hallucination'], subset['pred_prob'])
    base_auc = roc_auc_score(subset['is_hallucination'], subset['baseline_prob'])
    print(f"Sentence | Task: {task:<8} | Model AUROC: {auc:.4f} | Baseline AUROC: {base_auc:.4f}")

response_df = exploded_test.groupby('id').agg({
    'is_hallucination': 'max',  
    'pred_prob': 'max',         
    'baseline_prob': 'max',     
    'task_type': 'first'
}).reset_index()

resp_auc = roc_auc_score(response_df['is_hallucination'], response_df['pred_prob'])
resp_base_auc = roc_auc_score(response_df['is_hallucination'], response_df['baseline_prob'])
response_df['pred_label'] = (response_df['pred_prob'] >= best_resp_thr).astype(int)
resp_f1 = f1_score(response_df['is_hallucination'], response_df['pred_label'], zero_division=0)

print(f"\nResponse | Test Base Rate : {response_df['is_hallucination'].mean():.4f}")
print(f"Response | OVERALL      | Model AUROC: {resp_auc:.4f} | Baseline AUROC: {resp_base_auc:.4f}")
print(f"Response | Overall F1   : {resp_f1:.4f} (at tuned response threshold {best_resp_thr:.2f})")

# ------------------------------------------------------
# 6. XAI Features - SHAP on Champion Model
# ------------------------------------------------------
print("\n=== XAI: GENERATING SHAP VISUALIZATIONS ===")
X_test_abl = exploded_test[all_cols]
explainer = shap.TreeExplainer(final_model)
X_test_sample = X_test_abl.sample(n=500, random_state=2024)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample, feature_names=X_test_abl.columns)