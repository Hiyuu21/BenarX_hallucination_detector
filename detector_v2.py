# --------------------------------------
# Libraries
# --------------------------------------

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import re, random, os, pickle, math, json
import joblib, time, datetime, platform

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from scipy.sparse import csc_matrix

import sklearn, xgboost
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
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
SKIP_ENT_LABELS = {"CARDINAL", "ORDINAL"}   # numbers are handled by the number features


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
    return data[(data['split'] == split_type) & (data['quality']=='good')]

def merge_response(response, source):
    return pd.merge(response, source, on = 'source_id', how = 'inner')

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

    # Pre-allocate array to GUARANTEE shape matching
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
    
    # Efficient row-wise sparse dot product (cosine similarity since L2-normalized)
    sims = sent_vecs.multiply(ref_vecs).sum(axis=1)
    return np.asarray(sims).ravel()

def tag_hallucination(row):
    s_start, s_end = row['start_char'], row['end_char']
    for l in row['labels']:
        if max(s_start, l['start']) < min(s_end, l['end']):
            return 1
    return 0


# ------------------------------------------------
# V2 Update: New helper functions
# ------------------------------------------------ 

# ------------------------------------------------
# 1. Text helpers
# ------------------------------------------------ 

def content_tokens(text):
    """Lower-cased word tokens without stop-words."""
    return [t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOP_WORDS]
 
 
def _norm_num(s):
    v = float(s)
    return str(int(v)) if v.is_integer() else repr(v)      # "4.50" -> "4.5"
 
 
def extract_numbers(text):
    """Normalised numbers: '1,000' -> '1000', '3.50' -> '3.5'."""
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', str(text))
    return [_norm_num(x) for x in NUM_RE.findall(t)]
 
 
# --------------------------------------
# Source index: split each source into chunks once, embed once, reuse for every sentence
# --------------------------------------
def _units_from_doc(doc, max_words=60, overlap=10):
    """Source sentences; very long ones (e.g. Data2txt JSON) are cut into overlapping word windows."""
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
    ids = uniq['source_id'].tolist()
    texts = [str(t) for t in uniq['reference']]
 
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
        tf = csc_matrix((vals, (rows, cols)), shape=(len(units), max(len(vocab), 1)),
                        dtype=np.float32)
        dl = np.array([len(t) for t in tokens_per_unit], dtype=float)
        unit_tokset = [set(t) for t in tokens_per_unit]
 
        index[sid] = {
            'units': units,
            'emb': emb,
            'unit_tokset': unit_tokset,
            'unit_bigrams': [set(zip(t, t[1:])) for t in tokens_per_unit],
            'tok_all': set().union(*unit_tokset),
            'lower_text': text.lower(),
            'numbers': set(extract_numbers(text)),
            'vocab': vocab, 'df': df_counter, 'tf': tf,
            'dl': dl, 'avgdl': float(dl.mean()) if dl.mean() > 0 else 1.0,
        }
    return index
 
 
def bm25_best_norm(q_terms, idx, k1=1.5, b=0.75):
    """Best BM25 score over source chunks, divided by its theoretical maximum (0..1)."""
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
    if not cols or max_possible == 0:
        return 0.0
    tf = idx['tf'][:, cols].toarray()
    denom = tf + k1 * (1 - b + b * idx['dl'][:, None] / idx['avgdl'])
    score = ((tf * (k1 + 1)) / denom * np.array(idfs)).sum(axis=1)
    return float(score.max() / max_possible)
 
 
def _entity_matched(ent, idx):
    if ent in idx['lower_text']: return True
    toks = content_tokens(ent)
    return bool(toks) and all(t in idx['tok_all'] for t in toks)


# --------------------------------------
# 2. Lexical (best chunk) and entity / number features
# --------------------------------------
def compute_lexical_features(df, index, nlp, embedder):
    sentences = df['sentence_chunks'].astype(str).tolist()
    source_ids = df['source_id'].tolist()
 
    sent_emb = embedder.encode(sentences, batch_size=128, normalize_embeddings=True,
                               show_progress_bar=True)
    ents_per_sentence = [
        [e.text.lower() for e in doc.ents if e.label_ not in SKIP_ENT_LABELS]
        for doc in tqdm(nlp.pipe(sentences, batch_size=256, disable=["parser", "lemmatizer"]),
                        total=len(sentences), desc="NER on sentences")
    ]
 
    feats, evidence = [], []
    for i, (sid, sent) in enumerate(tqdm(zip(source_ids, sentences), total=len(sentences),
                                         desc="Lexical features")):
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
            'bigram_overlap_best': (len(bigrams & idx['unit_bigrams'][best]) / len(bigrams)
                                    if bigrams else ov_best),
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
        evidence.append({
            'evidence_top1': idx['units'][best],
            'evidence_top3': " ".join(idx['units'][j] for j in sorted(top3)),
        })
 
    return (pd.DataFrame(feats, index=df.index),
            pd.DataFrame(evidence, index=df.index))
 
 
# --------------------------------------
# 3. Context features
# --------------------------------------
def compute_context_features(df):
    assert 'id' in df.columns, "need the response 'id' column (one id per response)"
    out = pd.DataFrame(index=df.index)
    for t in ['QA', 'Summary', 'Data2txt']:
        out[f'is_{t}'] = (df['task_type'] == t).astype(int)
 
    grp = df.groupby('id')
    size = grp['id'].transform('size')
    pos = grp.cumcount()                      # rows are in sentence order after explode()
    out['sent_pos_rel'] = np.where(size > 1, pos / (size - 1).clip(lower=1), 0.0)
    out['n_sents_resp'] = size
    words = df['sentence_chunks'].astype(str).str.split().str.len()
    out['resp_len_words'] = words.groupby(df['id']).transform('sum')
    return out
 
 
# --------------------------------------
# 4. NLI features (small cross-encoder, premise = retrieved evidence, hypothesis = sentence)
# --------------------------------------
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
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits
        out[ids] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out
 
 
def compute_nli_features(evidence_df, hypotheses, model_name="cross-encoder/nli-deberta-v3-small",
                         batch_size=32, max_length=320, device=None):
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
        print("WARNING: label names not recognised, assuming order contradiction/entailment/neutral")
        i_con, i_ent, i_neu = 0, 1, 2
 
    # keep the hypothesis short so truncating the premise ('only_first') always works
    hyps = [" ".join(str(h).split()[:120]) for h in hypotheses]
 
    p1 = _nli_probs(evidence_df['evidence_top1'].tolist(), hyps, tok, model, device,
                    batch_size, max_length)
    p3 = _nli_probs(evidence_df['evidence_top3'].tolist(), hyps, tok, model, device,
                    batch_size, max_length)
 
    return pd.DataFrame({
        'nli_entail_top1': p1[:, i_ent], 'nli_contra_top1': p1[:, i_con],
        'nli_entail_ctx': p3[:, i_ent], 'nli_contra_ctx': p3[:, i_con],
        'nli_neutral_ctx': p3[:, i_neu],
    }, index=evidence_df.index)
 
 
# --------------------------------------
# Convenience wrappers
# --------------------------------------
def build_features_v2(df, nlp, embedder, use_nli=True, nli_model="cross-encoder/nli-deberta-v3-small"):
    index = build_source_index(df, nlp, embedder)
    lexical, evidence = compute_lexical_features(df, index, nlp, embedder)
    parts = [lexical, compute_context_features(df)]
    if use_nli:
        parts.append(compute_nli_features(evidence, df['sentence_chunks'].astype(str).tolist(),
                                          model_name=nli_model))
    return pd.concat(parts, axis=1)
 
 
def load_or_build_v2(df, cache_path, nlp, embedder, **kwargs):
    """Compute once, then reload from disk. Returns df with the new columns appended."""
    if os.path.exists(cache_path):
        print(f"Loading cached v2 features from {cache_path}")
        new = pd.read_pickle(cache_path)
    else:
        new = build_features_v2(df, nlp, embedder, **kwargs)
        new.to_pickle(cache_path)
    return pd.concat([df.drop(columns=[c for c in new.columns if c in df.columns]), new], axis=1)
 
 
# --------------------------------------
# Cross-validated evaluation (use this while iterating - NOT the test set)
# --------------------------------------
def cv_evaluate(model, X, y, groups, seeds=(0, 1, 2), n_splits=5):
    """Source-disjoint stratified CV, repeated over seeds. Returns mean/std of the metrics."""
    y = np.asarray(y)
    rows = []
    best_oof = None                       # OOF probabilities of the first seed
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

CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)       # to_pickle() does NOT create folders, so make it first

train_cache = f'{CACHE_DIR}/exploded_train_cache.pkl'   # sentences + original 5 features + label
test_cache = f'{CACHE_DIR}/exploded_test_cache.pkl'
v2_train_cache = f'{CACHE_DIR}/v2_train_features.pkl'   # ONLY the new v2 feature columns
v2_test_cache = f'{CACHE_DIR}/v2_test_features.pkl'
tfidf_cache = f'{CACHE_DIR}/tfidf_vectorizer_cache.pkl'

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
    exploded_train['tfidf_similarity'] = compute_tfidf_overlap(exploded_train, global_tfidf)
    exploded_train['is_hallucination'] = exploded_train.apply(tag_hallucination, axis=1)

    exploded_test['sentence_length'] = exploded_test['sentence_chunks'].str.split().str.len()
    exploded_test['word_overlap_ratio'] = exploded_test.apply(calculate_overlap, axis=1)
    exploded_test['unmatched_numbers_count'] = exploded_test.apply(count_unmatched_numbers, axis=1)
    exploded_test['semantic_similarity'] = compute_batch_cosine_similarity(exploded_test)
    exploded_test['tfidf_similarity'] = compute_tfidf_overlap(exploded_test, global_tfidf)
    exploded_test['is_hallucination'] = exploded_test.apply(tag_hallucination, axis=1)


    #print(exploded_train[['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count']].describe())
    #print()
    #print(exploded_test[['sentence_length', 'word_overlap_ratio', 'unmatched_numbers_count']].describe())
    #print()
    
    # Save the finished DataFrames for next time
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

print('\n --- Feature Correlation Check.')
corr = exploded_train['word_overlap_ratio'].corr(exploded_train['content_overlap_all'])
print(f'Correlation between word_overlap_ratio and content_overlap_all: {corr:.4f}')
if corr>0.95:
    print('Dropping content_overlap_all due to high collinearity.')
    all_cols.remove('content_overlap_all')
    LEXICAL_BEST.remove('content_overlap_all')

# ------------------------------------------------
# 3. Model Training
# ------------------------------------------------

print("\n --- Data loaded for model training...")

X = exploded_train[all_cols]
Y = exploded_train['is_hallucination']
groups = exploded_train['source_id']

models = {
    'Dummy (Baseline)': DummyClassifier(strategy='prior'),
    'Decision Tree (Pruned)': DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight='balanced', random_state=2024),
    'Logistic Regression (Scaled)': make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000)),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=2024, n_jobs=-1),
    'XGBoost': XGBClassifier(scale_pos_weight=8, random_state=2024, n_jobs=-1)
}

global_best_model_name = None
global_best_auc = 0.0
global_oof_probs = None
global_best_threshold = 0.5
cv_results = {}

for name, model in models.items():
    mean_metrics, std_metrics, oof = cv_evaluate(model, X, Y, groups)
    cv_results[name] = (mean_metrics, std_metrics)
    print(f"\n| Model:  {name:<25} ")
    print(f"| AUROC:  {mean_metrics['auroc']:.4f} ± {std_metrics['auroc']:.4f}")
    print(f"| PR-AUC: {mean_metrics['pr_auc']:.4f}")
    print(f"| F1:     {mean_metrics['best_f1']:.4f}")

    if mean_metrics['auroc']>global_best_auc:
        global_best_auc = mean_metrics['auroc']
        global_best_model_name = name
        global_oof_probs = oof
        global_best_threshold = mean_metrics['best_thr']

print(f"\nChampion Model: {global_best_model_name} (AUROC: {global_best_auc:.4f})")


print("\n --- Tuning Response-level Threshold (OOF).")
exploded_train['oof_prob'] = global_oof_probs
resp_train = exploded_train.groupby('id').agg(
    {
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

print(f"Optimal Response-level Threshold: {best_resp_thr:.2f} (OOF F1: {best_resp_f1:.4f})")
print(f"Train Response-level Base Rate: {resp_train['is_hallucination'].mean():.4f}")

# ------------------------------------------------
# 4. Leave-one-out Ablation Test
# ------------------------------------------------
print("\n --- Ablation Study.")

Y = exploded_train['is_hallucination']
groups = exploded_train['source_id']

experiments = {
    'All Features': all_cols,
    'No NLI': [c for c in all_cols if c not in NLI],
    'No Context': [c for c in all_cols if c not in CONTEXT],
    'No Entity/Num': [c for c in all_cols if c not in ENTITY_NUMBER],
    'No Lexical': [c for c in all_cols if c not in LEXICAL_BEST],
    'Evidence Only (No NLI/Context)': BASE + LEXICAL_BEST + ENTITY_NUMBER
}

for name, cols in experiments.items():
    mean, std, _ = cv_evaluate(models[global_best_model_name], exploded_train[cols], Y, groups)
    print(f"{name:<30} AUROC {mean['auroc']:.4f}±{std['auroc']:.4f} | PR-AUC {mean['pr_auc']:.4f} | F1 {mean['best_f1']:.4f}")


# ------------------------------------------------
# 5. Reporting: Task, Response, and Baselines 
# ------------------------------------------------

print("\n --- Final Reporting.")
final_model = clone(models[global_best_model_name])
final_model.fit(exploded_train[all_cols], Y)
test_probs = final_model.predict_proba(exploded_test[all_cols])[:, 1]
test_predictions = (test_probs >= global_best_threshold).astype(int)

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
# 6. Freeze the detector(s). XAI lives in xai_faithfulness.py and only LOADS these files.
# ------------------------------------------------------

def tune_response_threshold(oof_probs, frame):
    tmp = frame[['id', 'is_hallucination']].copy()
    tmp['p'] = oof_probs
    r = tmp.groupby('id').agg({'is_hallucination': 'max', 'p': 'max'})
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.01, 1.0, 0.01):
        f = f1_score(r['is_hallucination'], (r['p'] >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return best_t

def freeze(tag, cols, sent_thr, resp_thr, cv_mean, cv_std):
    cols = list(cols)
    # fit on plain arrays (no feature-name warnings, faster SHAP/LIME)
    model = clone(models[global_best_model_name]).fit(exploded_train[cols].values, Y.values)

    # stratified training sample (task x label): LIME training data + background rows
    parts = [g.sample(min(len(g), 1000), random_state=FREEZE_SEED)
             for _, g in exploded_train.groupby(['task_type', 'is_hallucination'])]
    X_ref = pd.concat(parts)[cols].values

    X_te = exploded_test[cols].values
    p_te = model.predict_proba(X_te)[:, 1]

    meta = ['id', 'source_id', 'task_type', 'is_hallucination', 'sentence_chunks']
    scored = exploded_test[meta + cols].copy()
    scored['p_hallu'] = p_te
    scored.to_pickle(f'{FROZEN_DIR}/test_scored_{tag}.pkl')

    resp = scored.groupby('id').agg({'is_hallucination': 'max', 'p_hallu': 'max'})
    model.predict_proba(X_te[:1000])                       # warm-up
    t0 = time.time(); model.predict_proba(X_te[:1000])
    ms_per_sentence = (time.time() - t0) / 1000 * 1000

    test_metrics = {
        'sentence_auroc': roc_auc_score(scored['is_hallucination'], p_te),
        'sentence_pr_auc': average_precision_score(scored['is_hallucination'], p_te),
        'sentence_f1': f1_score(scored['is_hallucination'], (p_te >= sent_thr).astype(int), zero_division=0),
        'response_auroc': roc_auc_score(resp['is_hallucination'], resp['p_hallu']),
        'response_f1': f1_score(resp['is_hallucination'], (resp['p_hallu'] >= resp_thr).astype(int), zero_division=0),
        'response_base_rate': float(resp['is_hallucination'].mean()),
        'classifier_ms_per_sentence': ms_per_sentence,
    }
    test_metrics = {k: float(v) for k, v in test_metrics.items()}

    bundle = {
        'tag': tag, 'model_name': global_best_model_name, 'model': model, 'feature_cols': cols,
        'feature_groups': {'BASE': BASE, 'LEXICAL_BEST': LEXICAL_BEST, 'ENTITY_NUMBER': ENTITY_NUMBER,
                           'CONTEXT': CONTEXT, 'NLI': NLI},
        'threshold_sentence': float(sent_thr), 'threshold_response': float(resp_thr),
        'medians': exploded_train[cols].median().values, 'means': exploded_train[cols].mean().values,
        'X_ref': X_ref,
        'cv_mean': {k: float(v) for k, v in cv_mean.items()}, 'cv_std': {k: float(v) for k, v in cv_std.items()},
        'test_metrics': test_metrics, 'pipeline': PIPELINE_MODELS,
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'xgboost': xgboost.__version__},
        'created': datetime.date.today().isoformat(),
    }
    path = f'{FROZEN_DIR}/frozen_{tag}.joblib'
    joblib.dump(bundle, path, compress=3)
    size_mb = os.path.getsize(path) / 1e6
    with open(f'{FROZEN_DIR}/frozen_{tag}_metrics.json', 'w') as f:
        json.dump({'tag': tag, 'model': global_best_model_name, 'features': cols,
                   'threshold_sentence': float(sent_thr), 'threshold_response': float(resp_thr),
                   'cv_mean': bundle['cv_mean'], 'cv_std': bundle['cv_std'],
                   'test': test_metrics, 'model_file_mb': size_mb}, f, indent=2)
    print(f"[{tag}] saved {path} ({size_mb:.1f} MB) | {len(cols)} features | "
          f"test sentence AUROC {test_metrics['sentence_auroc']:.4f} | "
          f"classifier {ms_per_sentence:.3f} ms/sentence")

FROZEN_DIR = './frozen'
os.makedirs(FROZEN_DIR, exist_ok=True)
FREEZE_SEED = 2024

# keep in sync with the models used in build_features_v2 / load_or_build_v2
PIPELINE_MODELS = {'spacy': 'en_core_web_sm', 'embedder': 'all-MiniLM-L6-v2',
                   'nli': 'cross-encoder/nli-deberta-v3-small', 'tfidf_cache': tfidf_cache}

print("\n --- Freezing detectors.")
cm, cs = cv_results[global_best_model_name]
freeze('full', all_cols, global_best_threshold, best_resp_thr, cm, cs)

evidence_cols = BASE + LEXICAL_BEST + ENTITY_NUMBER       # no context, no NLI
em, es, e_oof = cv_evaluate(models[global_best_model_name], exploded_train[evidence_cols], Y, groups)
freeze('evidence_only', evidence_cols, em['best_thr'], tune_response_threshold(e_oof, exploded_train), em, es)