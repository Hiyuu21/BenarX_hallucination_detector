import spacy, os, re, pickle, math
from spacy.lang.en.stop_words import STOP_WORDS
from scipy.sparse import csc_matrix
from config import CACHE_DIR, DATA_DIR, TOKEN_RE, NUM_RE, SKIP_ENT_LABELS
from tqdm import tqdm
from collections import Counter
from data_loader import DataLoader
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()
print(f"HF_TOKEN: {'OK.' if os.getenv('HF_TOKEN') else 'api key not found - continue as anonymous user'}")

tfidf_cache = f'{CACHE_DIR}/tfidf_vectorizer_cache.pkl'
if os.path.exists(tfidf_cache):
    with open(tfidf_cache, 'rb') as f:
        TFIDF = pickle.load(f)
else:
    dl = DataLoader(f"{DATA_DIR}/raw/ragtruth/train.jsonl")
    train, _ = dl.load_ragtruth()
    unique_train_sources = train['reference'].astype(str).unique().tolist()
    TFIDF = TfidfVectorizer(stop_words='english', max_features=10000)
    TFIDF.fit(unique_train_sources)
    with open(tfidf_cache, 'wb') as f:
        pickle.dump(TFIDF,f)

class FeatureExtractor:
    def __init__(self, config):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
        self.nli_model_name = "cross-encoder/nli-deberta-v3-small"
        self.feature_names = config.ALL_FEATURES

    # ----------------------
    # BASIC FEATURES
    # ----------------------

    def calculate_overlap(self, row):
        sentence_set = set(re.findall(r'\b\w+\b', str(row['sentence_chunks']).lower()))
        reference_set = set(re.findall(r'\b\w+\b', str(row['reference']).lower()))
        intersection = sentence_set & reference_set
        return len(intersection)/ len(sentence_set) if len(sentence_set) > 0 else 0.0

    def count_unmatched_numbers(self, row):
        sentence_nums = re.findall(r'\d+', str(row['sentence_chunks']).lower())
        reference_nums = set(re.findall(r'\d+', str(row['reference']).lower()))
        counter = sum(1 for num in sentence_nums if num not in reference_nums)
        return counter

    def precompute_source_embeddings(self, df):
        unique_refs = df[['source_id', 'reference']].drop_duplicates()
        ref_emb_dict = {}
        for _, row in unique_refs.iterrows():
            sid = row['source_id']
            ref_text = str(row['reference'])
            ref_doc = self.nlp(ref_text)
            ref_sents = [s.text.strip() for s in ref_doc.sents if s.text.strip()]
            if not ref_sents:
                ref_sents = [ref_text]
            ref_emb_dict[sid] = self.embedder.encode(ref_sents, show_progress_bar=False, batch_size=64)
        return ref_emb_dict

    def compute_batch_cosine_similarity(self, df):
        ref_emb_dict = self.precompute_source_embeddings(df)
        sentence_list = df['sentence_chunks'].astype(str).tolist()
        sent_embs = self.embedder.encode(sentence_list, show_progress_bar=True, batch_size=128)

        # Pre-allocate array to GUARANTEE shape matching
        similarities = np.zeros(len(df))
        for idx, row in enumerate(df.itertuples()):
            s_emb = sent_embs[idx].reshape(1, -1)
            r_embs = ref_emb_dict.get(row.source_id)
            if r_embs is not None and len(r_embs) > 0:
                sims = cosine_similarity(s_emb, r_embs)
                similarities[idx] = float(np.max(sims))
                
        return similarities

    def compute_tfidf_overlap(self, df, vectorizer):
        sent_vecs = vectorizer.transform(df['sentence_chunks'].astype(str))
        ref_vecs = vectorizer.transform(df['reference'].astype(str))
        
        # Efficient row-wise sparse dot product (cosine similarity since L2-normalized)
        sims = sent_vecs.multiply(ref_vecs).sum(axis=1)
        return np.asarray(sims).ravel()

    def tag_hallucination(self, row):
        s_start, s_end = row['start_char'], row['end_char']
        for l in row['labels']:
            if max(s_start, l['start']) < min(s_end, l['end']):
                return 1
        return 0


    # ----------------------
    # TEXT HELPERS
    # ----------------------

    def content_tokens(self, text):
        return [t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOP_WORDS]

    def _norm_num(self, s):
        v = float(s)
        return str(int(v) if v.is_integer else repr(v)) # 4.50 -> 4.5

    def extract_numbers(self, text):
        t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))','',str(text))
        return [self._norm_num(x) for x in NUM_RE.findall(t)]


    # ----------------------
    # GET SOURCE INDEX
    # ----------------------

    def _unit_from_docs(self, doc, max_words=60, overlap=10):
        "Long source sentences will be cut into overlapping word windows"
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

    def build_source_index(self, df, max_words = 60):
        uniq = df[['source_id', 'reference']].drop_duplicates('source_id')
        ids = uniq['source_id'].tolist()
        texts = [str(t) for t in uniq['reference']]

        units_per_source, all_units = {}, []
        docs = self.nlp.pipe(texts, batch_size=32, disable=["ner", "lemmatizer"])
        for sid, doc in tqdm(zip(ids, docs), total=len(ids), desc="Splitting sources"):
            units = self._units_from_doc(doc, max_words)
            units_per_source[sid] = units
            all_units.extend(units)
        emb_all = self.embedder.encode(all_units, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
        
        index, offset = {}, 0
        for sid, text in zip(ids, texts):
            units = units_per_source[sid]
            emb = emb_all[offset:offset + len(units)]
            offset += len(units)
            tokens_per_unit = [self.content_tokens(u) for u in units]
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
                'numbers': set(self.extract_numbers(text)),
                'vocab': vocab, 'df': df_counter, 'tf': tf,
                'dl': dl, 'avgdl': float(dl.mean()) if dl.mean() > 0 else 1.0,
            }
        return index


    # ------------------------------------------------------------------------------------------------------------
    # BM25 
    # -> probabilistic ranking function to find out the relevance between the documents to the search query
    # -> Cross-reference the response and the source context
    # ------------------------------------------------------------------------------------------------------------

    def bm25_best_norm(self, q_terms, idx, k1=1.5, b=0.75):
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
    
    
    def _entity_matched(self, ent, idx):
        if ent in idx['lower_text']: return True
        toks = self.content_tokens(ent)
        return bool(toks) and all(t in idx['tok_all'] for t in toks)

    
    # --------------------------------------
    # Lexical and entity / number features
    # -> Lexical means the exact string/word
    # --------------------------------------
    def compute_lexical_features(self, df, index):
        sentences = df['sentence_chunks'].astype(str).tolist()
        source_ids = df['source_id'].tolist()
    
        sent_emb = self.embedder.encode(sentences, batch_size=128, normalize_embeddings=True,
                                show_progress_bar=True)
        ents_per_sentence = [
            [e.text.lower() for e in doc.ents if e.label_ not in SKIP_ENT_LABELS]
            for doc in tqdm(self.nlp.pipe(sentences, batch_size=256, disable=["parser", "lemmatizer"]),
                            total=len(sentences), desc="NER on sentences")
        ]
    
        feats, evidence = [], []
        for i, (sid, sent) in enumerate(tqdm(zip(source_ids, sentences), total=len(sentences),
                                            desc="Lexical features")):
            idx = index[sid]
            sims = idx['emb'] @ sent_emb[i]
            order = np.argsort(-sims)
            best, top3 = order[0], order[:3]
    
            toks = self.content_tokens(sent)
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
                'bm25_best_norm': self.bm25_best_norm(tset, idx),
            }
    
            ents = ents_per_sentence[i]
            n_unmatched = sum(0 if self._entity_matched(e, idx) else 1 for e in ents)
            f['n_entities'] = len(ents)
            f['n_entities_unmatched'] = n_unmatched
            f['entity_match_ratio'] = 1.0 - n_unmatched / len(ents) if ents else 1.0
    
            nums = self.extract_numbers(sent)
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
    # Context features
    # --------------------------------------
    def compute_context_features(self, df):
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
    # NLI features (small cross-encoder, premise = retrieved evidence, hypothesis = sentence)
    # --------------------------------------
    def _nli_probs(self, premises, hypotheses, tok, model, device, batch_size, max_length):
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
    
    
    def compute_nli_features(self, evidence_df, hypotheses, batch_size=32, max_length=320, device=None):
    
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        tok = AutoTokenizer.from_pretrained(self.nli_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name).eval().to(device)
    
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
    
        p1 = self._nli_probs(evidence_df['evidence_top1'].tolist(), hyps, tok, model, device,
                        batch_size, max_length)
        p3 = self._nli_probs(evidence_df['evidence_top3'].tolist(), hyps, tok, model, device,
                        batch_size, max_length)
    
        return pd.DataFrame({
            'nli_entail_top1': p1[:, i_ent], 'nli_contra_top1': p1[:, i_con],
            'nli_entail_ctx': p3[:, i_ent], 'nli_contra_ctx': p3[:, i_con],
            'nli_neutral_ctx': p3[:, i_neu],
        }, index=evidence_df.index)
    
    
    # One-call feature
    
    def extract(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Extracts all features for a DataFrame of sentences."""
        print("Extracting Base Features...")
        features_df = pd.DataFrame(index=dataset.index)

        # Base features
        features_df["sentence_length"] = dataset['sentence_chunks'].str.split().str.len()
        features_df["word_overlap_ratio"] = dataset.apply(self.calculate_overlap, axis=1)
        features_df["unmatched_numbers_count"] = dataset.apply(self.count_unmatched_numbers, axis=1)
        features_df["semantic_similarity"] = self.compute_batch_cosine_similarity(dataset)
        features_df["tfidf_similarity"] = self.compute_tfidf_overlap(dataset, TFIDF) # Use global TFIDF

        # Get source index
        source_index = self.build_source_index(dataset)

        # lexical & entity features
        lexical_df, evidence_df = self.compute_lexical_features(dataset, source_index)

        # context features
        context_df = self.compute_context_features(dataset)

        # NLI features
        nli_df = self.compute_nli_features(
            evidence_df=evidence_df,
            hypotheses=dataset['sentence_chunks'].astype(str).tolist()
        )

        features_df = pd.concat([features_df, lexical_df, context_df, nli_df], axis=1)
        
        # Add Target Label (for training)
        if 'labels' in dataset.columns:
            features_df["is_hallucination"] = dataset.apply(self.tag_hallucination, axis=1)
            
        return features_df