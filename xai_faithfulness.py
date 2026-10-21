"""
xai_faithfulness.py - SHAP vs LIME vs Combined, tested for faithfulness to the FROZEN detector.

It only LOADS what detector_v2_2.py froze (./frozen/frozen_<TAG>.joblib and
./frozen/test_scored_<TAG>.pkl). It never retrains or recomputes features.

What is measured, for the sentences the detector FLAGGED (or a stratified sample, see EXPLAIN_SET)
  * Faithfulness : reset the explanation's top-k features and see how much P(hallucination) drops.
                   AOPC (probability) and AOPC / drop in log-odds. Bigger drop = more faithful.
                   Controls: random ordering, global feature importance, and an occlusion reference.
  * Agreement    : SHAP vs LIME (Spearman, top-1 match, top-3 Jaccard)
  * Stability    : LIME re-run with different seeds (SHAP is deterministic)
  * Cost         : seconds per explanation, approximate memory
  * Per task     : the same faithfulness numbers for QA / Summary / Data2txt

Usage:   python xai_faithfulness.py            (TAG = 'full')
         python xai_faithfulness.py evidence_only
Needs:   shap, lime, scipy, matplotlib, tqdm (psutil optional)
"""
import json, os, sys, time, warnings

import joblib, matplotlib, shap
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr, wilcoxon
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import psutil
except ImportError:
    psutil = None

# ======================================================================
# CONFIG
# ======================================================================
TAG = sys.argv[1] if len(sys.argv) > 1 else 'full'      # 'full' or 'evidence_only'
FROZEN_DIR = os.environ.get('FROZEN_DIR', './frozen')
OUT_DIR = os.environ.get('XAI_OUT_DIR', f'./xai_results/{TAG}')

EXPLAIN_SET = 'flagged'     # 'flagged' = sentences with p >= threshold (what the detector flags)
                            # 'all'     = stratified over task x true label
N_EXPLAIN = 600             # total sentences explained (split evenly over the strata)
MAX_K = 10                  # reset at most this many features / groups
GROUPINGS = ['task_only', 'family']   # 'none' | 'task_only' | 'family'  (see FAMILIES below)
RESET_MODES = ['median', 'marginal']  # how a "removed" feature is replaced
N_RANDOM = 10               # random orderings averaged for the random baseline
N_BG = 10                   # background rows averaged in 'marginal' mode
LIME_SAMPLES = 2000         # perturbed samples per LIME explanation
LIME_ATTRIBUTION = 'contribution'   # 'contribution' = weight*(x-mean)/scale  (comparable with SHAP)
                                    # 'weight'       = raw LIME weight (a slope, not a contribution)
SHAP_MODE = 'path_dependent'        # or 'interventional' (slower, uses X_ref as background)
STAB_N, STAB_REPEATS = 100, 3       # LIME stability: instances x repeats
SEED = 42

TASK_COLS = ['is_QA', 'is_Summary', 'is_Data2txt']
FAMILIES = {   # used when grouping == 'family' (correlated features are explained as one unit)
    'overlap': ['word_overlap_ratio', 'tfidf_similarity', 'content_overlap_best', 'content_overlap_top3',
                'content_overlap_all', 'bigram_overlap_best', 'bm25_best_norm'],
    'semantic': ['semantic_similarity', 'sim_top3_mean'],
    'entity': ['n_entities', 'n_entities_unmatched', 'entity_match_ratio'],
    'number': ['unmatched_numbers_count', 'n_numbers', 'n_numbers_unmatched', 'number_match_ratio'],
    'nli': ['nli_entail_top1', 'nli_contra_top1', 'nli_entail_ctx', 'nli_contra_ctx', 'nli_neutral_ctx'],
    'task': TASK_COLS,
    'length_position': ['sentence_length', 'sent_pos_rel', 'n_sents_resp', 'resp_len_words'],
}

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6 if psutil else float('nan')


# ======================================================================
# Load the frozen detector and its scored test set
# ======================================================================
bundle = joblib.load(f'{FROZEN_DIR}/frozen_{TAG}.joblib')
model, cols = bundle['model'], bundle['feature_cols']
thr = bundle['threshold_sentence']
medians, X_ref = np.asarray(bundle['medians']), np.asarray(bundle['X_ref'])
test = pd.read_pickle(f'{FROZEN_DIR}/test_scored_{TAG}.pkl')
print(f"Loaded frozen '{TAG}' ({bundle['model_name']}, {len(cols)} features, threshold {thr:.2f}, "
      f"created {bundle['created']}). Test sentences: {len(test)}")


def p1(A):
    return model.predict_proba(A)[:, 1]


# ======================================================================
# Stratified sample of sentences to explain
# ======================================================================
def stratified_sample(df, n, by, seed):
    parts = [g for _, g in df.groupby(by)]
    per = int(np.ceil(n / len(parts)))
    out = pd.concat([g.sample(min(per, len(g)), random_state=seed) for g in parts])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


if EXPLAIN_SET == 'flagged':
    pool = test[test['p_hallu'] >= thr]
    sample = stratified_sample(pool, N_EXPLAIN, ['task_type'], SEED)
else:
    sample = stratified_sample(test, N_EXPLAIN, ['task_type', 'is_hallucination'], SEED)

X = sample[cols].values.astype(float)
p0 = p1(X)
N, d = X.shape
print(f"Explaining {N} sentences ({EXPLAIN_SET}); per task: {sample['task_type'].value_counts().to_dict()}")

# ======================================================================
# SHAP
# ======================================================================
if SHAP_MODE == 'interventional':
    explainer = shap.TreeExplainer(model, data=X_ref[:200], feature_perturbation='interventional')
else:
    explainer = shap.TreeExplainer(model)


def shap_attr(A):
    sv = explainer.shap_values(A, check_additivity=False)
    if isinstance(sv, list):                 # older shap: one array per class
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                         # newer shap: (n, features, classes)
        sv = sv[:, :, -1]
    assert sv.shape == A.shape, f"unexpected SHAP shape {sv.shape} for input {A.shape}"
    return sv


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


mem0 = rss_mb()
t0 = time.time()
S = shap_attr(X)
shap_total = time.time() - t0
shap_mem = rss_mb() - mem0

# sanity check: does SHAP add up to the model output (and in which space)?
base = float(np.atleast_1d(explainer.expected_value)[-1])
err_prob = np.median(np.abs(S.sum(1) + base - p0))
err_logit = np.median(np.abs(S.sum(1) + base - logit(p0)))
space = 'probability' if err_prob <= err_logit else 'log-odds'
print(f"SHAP additivity: values live in {space} space (median abs error {min(err_prob, err_logit):.2e})")
if min(err_prob, err_logit) > 1e-2:
    print("  WARNING: SHAP does not add up to the model output - check shapes/model type before trusting it.")

t0 = time.time()
for i in range(min(30, N)):
    shap_attr(X[i:i + 1])
shap_single_ms = (time.time() - t0) / min(30, N) * 1000

# ======================================================================
# LIME
# ======================================================================
def make_lime(seed):
    return LimeTabularExplainer(X_ref, feature_names=cols, class_names=['faithful', 'hallucinated'],
                                mode='classification', discretize_continuous=False,
                                feature_selection='none', random_state=seed)


def lime_attr(A, seed, desc):
    ex = make_lime(seed)
    W = np.zeros(A.shape, dtype=float)
    for i in tqdm(range(len(A)), desc=desc):
        e = ex.explain_instance(A[i], model.predict_proba, num_features=A.shape[1],
                                num_samples=LIME_SAMPLES, labels=(1,))
        for j, w in e.local_exp[1]:
            W[i, j] = w
    if LIME_ATTRIBUTION == 'contribution':
        return W * (A - ex.scaler.mean_) / ex.scaler.scale_
    return W


mem0 = rss_mb()
t0 = time.time()
L = lime_attr(X, SEED, "LIME")
lime_total = time.time() - t0
lime_mem = rss_mb() - mem0

np.savez_compressed(f'{OUT_DIR}/attributions.npz', X=X, p0=p0, shap=S, lime=L, cols=np.array(cols),
                    task=sample['task_type'].values, label=sample['is_hallucination'].values)


# ======================================================================
# Units (feature groups), orderings, perturbation
# ======================================================================
def make_units(mode):
    idx = {c: i for i, c in enumerate(cols)}
    if mode == 'none':
        groups = {c: [c] for c in cols}
    elif mode == 'task_only':          # the 3 one-hot task columns are ONE feature
        groups = {c: [c] for c in cols if c not in TASK_COLS}
        groups['task_type'] = [c for c in TASK_COLS if c in idx]
    else:                              # 'family'
        groups = {fam: [c for c in mem if c in idx] for fam, mem in FAMILIES.items()}
        assigned = {c for mem in groups.values() for c in mem}
        for c in cols:
            if c not in assigned:
                groups[c] = [c]
    names = [n for n, mem in groups.items() if mem]
    return names, [[idx[c] for c in groups[n]] for n in names]


def to_units(A, unit_idx):
    return np.stack([A[:, ix].sum(axis=1) for ix in unit_idx], axis=1)


def l1(A):
    return A / (np.abs(A).sum(axis=1, keepdims=True) + 1e-12)


def order_of(A):                        # most positive (pushes toward 'hallucinated') first
    return np.argsort(-A, axis=1, kind='stable')


def rank_avg_order(A1, A2):
    r1 = np.argsort(order_of(A1), axis=1)
    r2 = np.argsort(order_of(A2), axis=1)
    score = (r1 + r2) / 2.0 - 1e-6 * (l1(A1) + l1(A2))      # tie-break by combined size
    return np.argsort(score, axis=1, kind='stable')


def prob_with_mask(mask, ref, mode):
    """P(hallucinated) after resetting the masked cells."""
    if mode == 'median':
        return p1(np.where(mask, medians[None, :], X))
    acc = np.zeros(N)
    for b in ref:
        acc += p1(np.where(mask, b[None, :], X))
    return acc / len(ref)


def reset_curve(order, unit_idx, K, ref, mode):
    U = len(unit_idx)
    pos = np.empty((N, U), dtype=int)
    pos[np.arange(N)[:, None], order] = np.arange(U)[None, :]
    unit_of_col = np.full(d, -1)
    for u, ix in enumerate(unit_idx):
        unit_of_col[ix] = u
    col_pos = pos[:, unit_of_col]                      # rank of each column's unit, per sentence
    P = np.empty((N, K + 1))
    P[:, 0] = p0
    for k in range(1, K + 1):
        P[:, k] = prob_with_mask(col_pos < k, ref, mode)
    return P


def drops(P):
    dprob = P[:, [0]] - P[:, 1:]
    dlogit = logit(P[:, [0]]) - logit(P[:, 1:])
    return dprob, dlogit                               # (N, K) each


def boot_ci(v, B=1000):
    idx = rng.integers(0, len(v), (B, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5])


def paired_p(a, b):
    if np.allclose(a, b):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def agreement(A1, A2, topn=3):
    rhos = [spearmanr(a, b).correlation for a, b in zip(A1, A2) if np.std(a) > 0 and np.std(b) > 0]
    n = min(topn, A1.shape[1])
    jac = []
    for a, b in zip(A1, A2):
        sa, sb = set(np.argsort(-a)[:n]), set(np.argsort(-b)[:n])
        jac.append(len(sa & sb) / len(sa | sb))
    return float(np.nanmean(rhos)), float((A1.argmax(1) == A2.argmax(1)).mean()), float(np.mean(jac))


# ======================================================================
# Main comparison
# ======================================================================
all_summaries = []
for grouping in GROUPINGS:
    names, unit_idx = make_units(grouping)
    U = len(names)
    K = min(MAX_K, U)
    S_u, L_u = to_units(S, unit_idx), to_units(L, unit_idx)
    C_u = 0.5 * (l1(S_u) + l1(L_u))
    imp = getattr(model, 'feature_importances_', None)
    imp_u = np.array([imp[ix].sum() for ix in unit_idx]) if imp is not None else np.zeros(U)

    orders = {
        'SHAP': order_of(S_u),
        'LIME': order_of(L_u),
        'Combined (mean)': order_of(C_u),
        'Combined (rank)': rank_avg_order(S_u, L_u),
        'Global importance': np.tile(np.argsort(-imp_u), (N, 1)),
    }
    top_unit = {m: [names[j] for j in np.bincount(o[:, 0], minlength=U).argsort()[::-1][:3]]
                for m, o in orders.items()}

    for mode in RESET_MODES:
        ref = X_ref[rng.choice(len(X_ref), N_BG, replace=False)]
        res = {m: drops(reset_curve(o, unit_idx, K, ref, mode)) for m, o in orders.items()}

        rd, rl = [], []                                              # random baseline
        for _ in range(N_RANDOM):
            a, b = drops(reset_curve(order_of(rng.random((N, U))), unit_idx, K, ref, mode))
            rd.append(a); rl.append(b)
        res['Random'] = (np.mean(rd, axis=0), np.mean(rl, axis=0))

        occ = np.zeros((N, U))                                       # metric-aligned reference
        for u, ix in enumerate(unit_idx):
            mask = np.zeros((N, d), dtype=bool)
            mask[:, ix] = True
            occ[:, u] = p0 - prob_with_mask(mask, ref, mode)
        res['Occlusion (reference)'] = drops(reset_curve(order_of(occ), unit_idx, K, ref, mode))

        rows = []
        for m, (dp, dl) in res.items():
            aopc = dp.mean(axis=1)
            lo, hi = boot_ci(aopc)
            rows.append({'grouping': grouping, 'reset': mode, 'method': m, 'K': K,
                         'AOPC_prob': aopc.mean(), 'CI_low': lo, 'CI_high': hi,
                         'AOPC_logodds': dl.mean(axis=1).mean(),
                         'drop@1_prob': dp[:, 0].mean(), 'drop@3_prob': dp[:, :min(3, K)].mean(axis=0)[-1],
                         'drop@1_logodds': dl[:, 0].mean()})
        summ = pd.DataFrame(rows)
        all_summaries.append(summ)

        print(f"\n=== grouping={grouping} ({U} units) | reset={mode} | N={N} | K={K} | "
              f"explaining P(hallucinated) in {space} space for SHAP ===")
        print(summ[['method', 'AOPC_prob', 'CI_low', 'CI_high', 'AOPC_logodds',
                    'drop@1_prob', 'drop@3_prob']].round(4).to_string(index=False))

        a = {m: res[m][0].mean(axis=1) for m in res}
        tests = [('SHAP vs LIME', 'SHAP', 'LIME'), ('SHAP vs Random', 'SHAP', 'Random'),
                 ('LIME vs Random', 'LIME', 'Random'),
                 ('Combined(mean) vs SHAP', 'Combined (mean)', 'SHAP'),
                 ('Combined(mean) vs LIME', 'Combined (mean)', 'LIME'),
                 ('SHAP vs Global importance', 'SHAP', 'Global importance')]
        print("Paired Wilcoxon p-values on per-sentence AOPC (many tests - read effect sizes first):")
        for label, x, y in tests:
            print(f"  {label:<28} p = {paired_p(a[x], a[y]):.4g}   (mean diff {a[x].mean() - a[y].mean():+.4f})")

        print("Per task (AOPC_prob):")
        per_task = pd.DataFrame({m: pd.Series(a[m]).groupby(sample['task_type'].values).mean() for m in a})
        print(per_task.round(4).to_string())
        per_task.to_csv(f'{OUT_DIR}/per_task_{grouping}_{mode}.csv')

        plt.figure(figsize=(6.5, 4.2))
        for m, (dp, _) in res.items():
            plt.plot(range(1, K + 1), dp.mean(axis=0), marker='o', ms=3,
                     ls='--' if m in ('Random', 'Occlusion (reference)') else '-', label=m)
        plt.xlabel(f'top-k {grouping if grouping != "none" else "features"} reset')
        plt.ylabel('mean drop in P(hallucinated)')
        plt.title(f'{TAG}: faithfulness curve ({grouping}, reset={mode})')
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/curve_{grouping}_{mode}.png', dpi=150)
        plt.close()

    print(f"\n[{grouping}] most common #1 unit per method: {top_unit}")
    sp, t1, j3 = agreement(S_u, L_u)
    print(f"[{grouping}] SHAP vs LIME agreement: Spearman {sp:.3f} | top-1 match {t1:.3f} | top-3 Jaccard {j3:.3f}")
    mean_abs = pd.Series(np.abs(S_u).mean(axis=0), index=names).sort_values(ascending=False)
    print(f"[{grouping}] mean |SHAP| per unit (top 8):\n{mean_abs.head(8).round(4).to_string()}")

pd.concat(all_summaries).to_csv(f'{OUT_DIR}/faithfulness_summary.csv', index=False)

# ======================================================================
# LIME stability (SHAP is deterministic, so it is perfectly stable by construction)
# ======================================================================
names, unit_idx = make_units('task_only')
M = min(STAB_N, N)
runs = [to_units(lime_attr(X[:M], SEED + 1 + r, f"LIME stability {r + 1}/{STAB_REPEATS}"), unit_idx)
        for r in range(STAB_REPEATS)]
st_sp, st_j3 = [], []
for i in range(STAB_REPEATS):
    for j in range(i + 1, STAB_REPEATS):
        sp, _, j3 = agreement(runs[i], runs[j])
        st_sp.append(sp); st_j3.append(j3)
print(f"\nLIME stability over {STAB_REPEATS} seeds ({M} sentences, {LIME_SAMPLES} samples): "
      f"Spearman {np.mean(st_sp):.3f} | top-3 Jaccard {np.mean(st_j3):.3f}   (SHAP: 1.000 by construction)")

# ======================================================================
# Cost
# ======================================================================
cost = {'shap_batch_ms_per_sentence': shap_total / N * 1000, 'shap_single_ms': shap_single_ms,
        'lime_ms_per_sentence': lime_total / N * 1000, 'lime_samples': LIME_SAMPLES,
        'shap_rss_delta_mb': shap_mem, 'lime_rss_delta_mb': lime_mem,
        'lime_stability_spearman': float(np.mean(st_sp)), 'lime_stability_top3_jaccard': float(np.mean(st_j3)),
        'classifier_ms_per_sentence': bundle['test_metrics']['classifier_ms_per_sentence']}
print("\nCost:", json.dumps({k: round(v, 3) for k, v in cost.items()}, indent=2))
with open(f'{OUT_DIR}/cost_and_stability.json', 'w') as f:
    json.dump(cost, f, indent=2)
print(f"\nSaved everything to {OUT_DIR}/")