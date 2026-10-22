# --------------------------------------
# Libraries
# --------------------------------------

from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import re, os, pickle, math, joblib
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
from scipy.sparse import csc_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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


CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)

hb_cache = f'{CACHE_DIR}/hb_exploded_cache.pkl'   
hb_v2_cache = f'{CACHE_DIR}/hb_v2_features.pkl' 
hb_tfidf_cache = f'{CACHE_DIR}/hb_tfidf_vectorizer.pkl'

if os.path.exists(hb_cache) and os.path.exists(hb_v2_cache) and os.path.exists(hb_tfidf_cache):
    print("\n --- Found cached data. Bypassing spaCy chunking and loading directly...")
    exploded_df = pd.read_pickle(hb_cache)
    with open(hb_tfidf_cache, 'rb') as f:
        global_tfidf = pickle.load(f)

else:
    print("No cache found. Running full chunking and feature extraction pipeline...")

    df = pd.DataFrame(load_dataset("PatronusAI/HaluBench")['test'])
    df = df[df['source_ds'] != 'RAGTruth'].copy()

    df = df.rename(columns = 
            {'passage':'reference', 
            'answer':'response', 
            'id':'source_id', 
            'source_ds':'task_type'
    })

    df['is_hallucination'] = (df['label'] == 'FAIL').astype(int) # FAIL = 1 (Hallucinated), PASS = 0
    df['id'] = df.index.astype(str)

    print('\n --- Chunking Sentences...')
    df['sentence_chunks'] = df['response'].progress_apply(chunk_into_sentence)
    exploded_df = df.explode('sentence_chunks').dropna(subset = ['sentence_chunks']).reset_index(drop=True)

    unpacked = pd.DataFrame(exploded_df['sentence_chunks'].tolist(), index=exploded_df.index)
    exploded_df['sentence_chunks'] = unpacked[0]
    exploded_df['start_char'] = unpacked[1]
    exploded_df['end_char'] = unpacked[2]

    print('\n --- Fitting TF-IDF and extracting base features.')
    unique_sources = df['reference'].astype(str).unique().tolist()
    global_tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    global_tfidf.fit(unique_sources)

    exploded_df['sentence_length'] = exploded_df['sentence_chunks'].str.split().str.len()
    exploded_df['word_overlap_ratio'] = exploded_df.apply(calculate_overlap, axis=1)
    exploded_df['unmatched_numbers_count'] = exploded_df.apply(count_unmatched_numbers, axis=1)
    exploded_df['semantic_similarity'] = compute_batch_cosine_similarity(exploded_df)
    exploded_df['tfidf_similarity'] = compute_tfidf_overlap(exploded_df, global_tfidf)

    exploded_df.to_pickle(hb_cache)
    with open(hb_tfidf_cache, 'wb') as f:
        pickle.dump(global_tfidf, f)

exploded_df = load_or_build_v2(exploded_df, hb_v2_cache, nlp, embedder)
all_cols = BASE + LEXICAL_BEST + ENTITY_NUMBER + CONTEXT + NLI

if 'content_overlap_all' in all_cols and 'content_overlap_all' in exploded_df.columns:
    corr = exploded_df['word_overlap_ratio'].corr(exploded_df['content_overlap_all'])
    if corr>0.95:
        all_cols.remove('content_overlap_all')


Y = exploded_df['is_hallucination']
groups = exploded_df['source_id']

print("\n" + "="*55)
print(" EXPERIMENT 1: ZERO-SHOT TRANSFER (RAGTruth -> HaluBench)")
print("="*55)

# 1. Load the frozen model and its EXACT feature columns
FROZEN_DIR = './frozen'
bundle = joblib.load(f'{FROZEN_DIR}/frozen_full.joblib')
frozen_model = bundle['model']
frozen_cols = bundle['feature_cols'] # Critical: MUST match the exact columns the model was trained on

print(f"Loaded frozen model: {bundle['model_name']} with {len(frozen_cols)} features.")

# 2. Predict on HaluBench without retraining
X_transfer = exploded_df[frozen_cols]
transfer_probs = frozen_model.predict_proba(X_transfer)[:, 1]

# 3. Add to DataFrame and calculate Response-Level metrics
exploded_df['transfer_prob'] = transfer_probs
exploded_df['baseline_prob'] = 1.0 - exploded_df['word_overlap_ratio']

resp_transfer = exploded_df.groupby('id').agg({
    'is_hallucination': 'max',
    'transfer_prob': 'max',
    'baseline_prob': 'max',
    'task_type': 'first' # HaluBench source domain (e.g., FinanceBench, DROP)
}).reset_index()

# 4. Overall Transfer Metrics
transfer_base_rate = resp_transfer['is_hallucination'].mean()
transfer_auc = roc_auc_score(resp_transfer['is_hallucination'], resp_transfer['transfer_prob'])
transfer_pr = average_precision_score(resp_transfer['is_hallucination'], resp_transfer['transfer_prob'])
baseline_auc = roc_auc_score(resp_transfer['is_hallucination'], resp_transfer['baseline_prob'])

print(f"HaluBench Response Base Rate: {transfer_base_rate:.4f}")
print(f"Overlap-Only Baseline AUROC:  {baseline_auc:.4f}")
print(f"Transfer Model AUROC:         {transfer_auc:.4f}")
print(f"Transfer Model PR-AUC:        {transfer_pr:.4f}")

# 5. Breakdown by HaluBench Source Domain
print("\n--- Zero-Shot Transfer AUROC by Source ---")
for src in resp_transfer['task_type'].unique():
    sub = resp_transfer[resp_transfer['task_type'] == src]
    # Check if a source has only 1 class (prevents ROC-AUC crash)
    if sub['is_hallucination'].nunique() > 1:
        s_auc = roc_auc_score(sub['is_hallucination'], sub['transfer_prob'])
        print(f"Source: {src:<15} | AUROC: {s_auc:.4f}")
    else:
        print(f"Source: {src:<15} | AUROC: N/A (Only 1 class present)")


print("\n" + "="*55)
print(" EXPERIMENT 2: INTERNAL HALUBENCH CROSS-VALIDATION")
print("="*55)

# 1. Clone the architecture from the frozen model so we are comparing apples-to-apples
cv_model = clone(frozen_model)

print("Running StratifiedGroupKFold (3 Seeds) directly on HaluBench...")
mean_metrics, std_metrics, oof_probs = cv_evaluate(cv_model, X_transfer, Y, groups)

# 2. Attach Out-Of-Fold probabilities to calculate response-level CV metrics
exploded_df['cv_oof_prob'] = oof_probs

resp_cv = exploded_df.groupby('id').agg({
    'is_hallucination': 'max',
    'cv_oof_prob': 'max'
}).reset_index()

cv_resp_auc = roc_auc_score(resp_cv['is_hallucination'], resp_cv['cv_oof_prob'])
cv_resp_pr = average_precision_score(resp_cv['is_hallucination'], resp_cv['cv_oof_prob'])

print(f"Internal CV Sentence AUROC: {mean_metrics['auroc']:.4f} ± {std_metrics['auroc']:.4f}")
print(f"Internal CV Response AUROC: {cv_resp_auc:.4f}")
print(f"Internal CV Response PR-AUC:{cv_resp_pr:.4f}")