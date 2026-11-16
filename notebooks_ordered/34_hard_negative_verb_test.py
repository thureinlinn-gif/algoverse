"""
Phase 7, Experiment A: noun-matched hard-negative verb test on the full 3,268-segment
validation set, using the existing representative heads (no training).

Why: Top-1 over 282 verbs is dominated by verbs the objects already give away
('add' is 35.65% of val; a single DINOv2 frame nearly matches the full clip). The
benchmarks where V-JEPA2 beats image models are ones where appearance cannot
separate the classes. The in-domain equivalent: for each segment, choose between
the true verb and the verb the *rest of the caption* most suggests -- the
masked-caption text model's highest-scoring wrong verb -- so the objects and
context in view cannot settle it. Within-pair accuracy has a 50% chance level.

Steps
 1. Masked-caption text baseline: TF-IDF 1-2-grams + logistic regression. C is chosen
    by 5-fold video-grouped CV on train. Val segments are scored by the model fit on
    all of train; train segments get *out-of-fold* probabilities from that same CV,
    so every segment's hard negative comes from a model that never saw its caption
    (script 35 uses the train-segment negatives -- an in-sample text model would
    produce systematically easier negatives there).
 2. Negatives per segment: hard = text model's top verb != true; two controls --
    uniform-random over the 282 verbs, and frequency-matched random (drawn from the
    train verb distribution). Asserted never equal to the true verb.
 3. Score true vs. negative with every representative head (script 33's registry):
    DINOv2 mean-pool, DINOv2 order-aware halves, DINOv3 (if script 32 cached its text
    tower), V-JEPA2 label-free mean-pool, V-JEPA2 SSv2-pooler headline. Blind control
    per head: the mean train embedding scored the same way (prior only).
 4. Subsets: all; excluding 'add' (true or negative); and by pair type --
      synonym       : both verbs in one SYNONYM_GROUPS group (cut/chop/slice, ...).
                      These differ mostly by caption wording, not motion, so every
                      model should sit near chance; reported separately so they
                      don't dilute the rest.
      same_object   : NOT synonyms, and the negative verb occurs with this segment's
                      direct object somewhere in train captions (pour/add the milk,
                      add/remove the lid). Objects fixed, action differs -- the
                      motion-relevant subset.
      other         : everything else.
 5. V-JEPA2 head minus each DINO head, per subset, with the video-cluster bootstrap.

Pre-registered decision quantity (REPRODUCE.md Phase 7): a V-JEPA2 head beats the
best DINO head by >= 3pp on hard-pair accuracy, all segments, cluster CI excluding 0.

Reads : annotations/segments_verbs_timed_clean.csv, processed/nb8_text_features_*.pt,
        embeddings + checkpoints via 33_linear_head_lib
Writes: processed/nb8_text_baseline_masked_caption.csv  (all 13,003 segments)
        processed/nb8_hard_negative_pairs.csv           (val, per segment x model)
        processed/nb8_hard_negative_summary.csv
        processed/nb8_hard_negative_diffs.csv
        processed/nb8_hard_negative_synonym_groups.csv  (the judgment call, for review)
~5-10 min CPU (mostly the text-model CV), seconds of GPU.
"""
import os, sys, importlib
from collections import Counter
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
lib = importlib.import_module('33_linear_head_lib')
OUT_DIR = lib.OUT_DIR

TEXT_C_GRID = [1.0, 10.0, 100.0]
N_FOLDS = 5
SEED = 0
DECISION_PP = 0.03

# Judgment call, exported for review: verbs captioners use interchangeably for the
# same visible action. Kept narrow on purpose -- pour/add, add/remove, flip/stir are
# NOT synonyms; they are exactly the pairs the test exists to probe.
SYNONYM_GROUPS = [
    {'cut', 'chop', 'slice', 'dice', 'mince'},
    {'mix', 'stir', 'combine', 'whisk', 'blend', 'beat', 'toss'},
    {'add', 'put', 'place', 'set', 'drop', 'transfer', 'insert'},
    {'cook', 'fry', 'saute', 'sear', 'brown'},
    {'boil', 'simmer'},
    {'top', 'garnish'},
    {'heat', 'preheat', 'warm'},
    {'remove', 'take'},
    {'pour', 'drizzle'},
    {'sprinkle', 'season', 'dust'},
    {'flip', 'turn'},
    {'press', 'flatten'},
]
SYN_OF = {v: i for i, g in enumerate(SYNONYM_GROUPS) for v in g}


def is_synonym(a, b):
    return a in SYN_OF and SYN_OF.get(a) == SYN_OF.get(b)


df, verb_vocab, label2id = lib.load_segments()
df_train = df[df['subset'] == 'training'].reset_index(drop=True)
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
n_verbs = len(verb_vocab)
print(f"Train / Val: {len(df_train):,} / {len(df_val):,}  |  verbs: {n_verbs}")

pd.DataFrame([{'group': i, 'verb': v, 'in_vocab': v in label2id}
              for i, g in enumerate(SYNONYM_GROUPS) for v in sorted(g)]) \
    .to_csv(os.path.join(OUT_DIR, 'nb8_hard_negative_synonym_groups.csv'), index=False)


# ── 1. Masked-caption text baseline ──────────────────────────────────────────
def fit_predict(C, tr_text, tr_y, te_text):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    Xtr = vec.fit_transform(tr_text)
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(Xtr, tr_y)
    proba = np.zeros((len(te_text), n_verbs), dtype=np.float32)
    proba[:, clf.classes_] = clf.predict_proba(vec.transform(te_text))
    return proba


tr_text = df_train['masked_caption'].str.lower().to_numpy()
tr_y = df_train['verb_label'].to_numpy()
folds = list(GroupKFold(n_splits=N_FOLDS).split(tr_text, tr_y, groups=df_train['video_id']))

print("\n=== Text baseline: choosing C by video-grouped CV on train ===")
jobs = [(C, k) for C in TEXT_C_GRID for k in range(N_FOLDS)]
outs = Parallel(n_jobs=min(len(jobs), os.cpu_count()))(
    delayed(fit_predict)(C, tr_text[folds[k][0]], tr_y[folds[k][0]], tr_text[folds[k][1]]) for C, k in jobs)
oof = {C: np.zeros((len(df_train), n_verbs), dtype=np.float32) for C in TEXT_C_GRID}
for (C, k), proba in zip(jobs, outs):
    oof[C][folds[k][1]] = proba
cv_acc = {C: (oof[C].argmax(1) == tr_y).mean() for C in TEXT_C_GRID}
for C, a in cv_acc.items():
    print(f"  C={C:<6g} out-of-fold Top-1 {a*100:.2f}%")
best_C = max(cv_acc, key=cv_acc.get)
print(f"  -> C={best_C:g}")

val_proba = fit_predict(best_C, tr_text, tr_y, df_val['masked_caption'].str.lower().to_numpy())
val_y = df_val['verb_label'].to_numpy()
print(f"Text baseline on val (fit on all train): Top-1 {(val_proba.argmax(1) == val_y).mean()*100:.2f}%  "
      f"(N={len(df_val):,})")


# ── 2. Negatives ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
train_freq = np.bincount(tr_y, minlength=n_verbs).astype(float)


def negatives(proba, y):
    masked = proba.copy()
    masked[np.arange(len(y)), y] = -1.0
    hard = masked.argmax(1)
    uni = np.array([rng.choice([j for j in range(n_verbs) if j != t]) for t in y])
    freq = []
    for t in y:
        p = train_freq.copy(); p[t] = 0; p /= p.sum()
        freq.append(rng.choice(n_verbs, p=p))
    freq = np.array(freq)
    assert not (hard == y).any() and not (uni == y).any() and not (freq == y).any()
    order = (-proba).argsort(1)
    true_rank = (order == y[:, None]).argmax(1) + 1
    return hard, uni, freq, true_rank


def text_table(d, proba, fit):
    y = d['verb_label'].to_numpy()
    hard, uni, freq, true_rank = negatives(proba, y)
    return pd.DataFrame({
        'segment_id': d['segment_id'], 'video_id': d['video_id'], 'subset': d['subset'],
        'ground_truth': d['extracted_verb'], 'text_top1_pred': [verb_vocab[i] for i in proba.argmax(1)],
        'text_top1_correct': proba.argmax(1) == y, 'text_true_rank': true_rank,
        'text_true_prob': proba[np.arange(len(y)), y],
        'hard_negative': [verb_vocab[i] for i in hard], 'uniform_negative': [verb_vocab[i] for i in uni],
        'freq_negative': [verb_vocab[i] for i in freq], 'text_fit': fit})


text_df = pd.concat([text_table(df_train, oof[best_C], f'out-of-fold (C={best_C:g})'),
                     text_table(df_val, val_proba, f'fit on train (C={best_C:g})')], ignore_index=True)


# ── Direct objects (spaCy), for the same-object subset and script 36 ─────────
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner'])


def direct_object(doc, verb):
    """Lemma of the verb's direct object; falls back to the first noun after the
    verb when the small parser mis-tags the imperative (common for 'add ...')."""
    vtok = next((t for t in doc if t.lower_ == verb or t.lemma_ == verb), None)
    if vtok is not None:
        objs = [c for c in vtok.children if c.dep_ in ('dobj', 'obj')]
        if objs:
            return objs[0].lemma_.lower()
    start = vtok.i + 1 if vtok is not None else 0
    noun = next((t for t in doc[start:] if t.pos_ in ('NOUN', 'PROPN')), None)
    return noun.lemma_.lower() if noun is not None else ''


print("\nParsing captions for direct objects...")
docs = nlp.pipe(df['caption'].str.lower().tolist(), batch_size=512, n_process=8)
df['object'] = [direct_object(doc, v) for doc, v in zip(docs, df['extracted_verb'])]
text_df = text_df.merge(df[['segment_id', 'object']], on='segment_id')
print(f"  segments with an object: {(df['object'] != '').mean()*100:.1f}%")
text_df.to_csv(os.path.join(OUT_DIR, 'nb8_text_baseline_masked_caption.csv'), index=False)

train_pairs = set(zip(df.loc[df['subset'] == 'training', 'extracted_verb'],
                      df.loc[df['subset'] == 'training', 'object']))
pairs = text_df[text_df['subset'] == 'validation'].set_index('segment_id').loc[df_val['segment_id']].reset_index()
pairs['synonym_pair'] = [is_synonym(a, b) for a, b in zip(pairs['ground_truth'], pairs['hard_negative'])]
pairs['same_object'] = [(o != '') and ((neg, o) in train_pairs)
                        for neg, o in zip(pairs['hard_negative'], pairs['object'])]
pairs['pair_type'] = np.where(pairs['synonym_pair'], 'synonym',
                              np.where(pairs['same_object'], 'same_object', 'other'))
print("Pair types:", pairs['pair_type'].value_counts().to_dict())
print("Most common hard-negative pairs:")
print((pairs['ground_truth'] + ' -> ' + pairs['hard_negative']).value_counts().head(12).to_string())


# ── 3. Score every head ──────────────────────────────────────────────────────
NEG_COLS = {'hard': 'hard_negative', 'uniform': 'uniform_negative', 'freq': 'freq_negative'}
ids_val = pairs['segment_id'].tolist()
true_idx = torch.tensor(pairs['ground_truth'].map(label2id).to_numpy())
models = []
for tag in lib.HEADS:
    r = lib.load_representative_head(tag, df_val, verb_vocab)
    if r is None:
        continue
    W, X, ids, info = r
    assert ids == ids_val, f"{tag}: val coverage differs from the canonical 3,268"
    S = lib.score_matrix(X, W, info['text_feats'], info['logit_scale'])
    s_true = S.gather(1, true_idx[:, None]).squeeze(1)
    for neg, col in NEG_COLS.items():
        neg_idx = torch.tensor(pairs[col].map(label2id).to_numpy())
        pairs[f'{tag}__{neg}'] = (s_true > S.gather(1, neg_idx[:, None]).squeeze(1)).numpy()
    # Blind control: the mean train embedding, scored identically -> the head's prior only.
    Xtr, _ = lib.load_embeddings(lib.HEADS[tag]['emb_dir'], df_train['segment_id'].tolist())
    s_blind = lib.score_matrix(Xtr.mean(0, keepdim=True), W, info['text_feats'], info['logit_scale'])[0]
    neg_idx = torch.tensor(pairs['hard_negative'].map(label2id).to_numpy())
    pairs[f'{tag}_blind__hard'] = (s_blind[true_idx] > s_blind[neg_idx]).numpy()
    models.append(tag)
pairs['text_model__hard'] = pairs['text_top1_correct']   # by construction: picks true iff its top-1 is true
pairs.to_csv(os.path.join(OUT_DIR, 'nb8_hard_negative_pairs.csv'), index=False)


# ── 4. Summary ───────────────────────────────────────────────────────────────
SUBSETS = {
    'all': np.ones(len(pairs), bool),
    'excl_add': ((pairs['ground_truth'] != 'add') & (pairs['hard_negative'] != 'add')).to_numpy(),
    'synonym': (pairs['pair_type'] == 'synonym').to_numpy(),
    'same_object': (pairs['pair_type'] == 'same_object').to_numpy(),
    'other': (pairs['pair_type'] == 'other').to_numpy(),
}
rows = []
for col in [c for c in pairs.columns if '__' in c]:
    model, neg = col.split('__')
    for sub, m in SUBSETS.items():
        if neg != 'hard' and sub != 'all':
            continue
        cb = lib.cluster_bootstrap(pairs.loc[m, col], pairs.loc[m, 'video_id'])
        rows.append({'model': model, 'label': lib.HEAD_LABELS.get(model, model), 'negative': neg,
                     'subset': sub, 'n': int(m.sum()), 'n_videos': cb['n_videos'],
                     'pair_acc': cb['mean'], 'ci_lo': cb['ci_lo'], 'ci_hi': cb['ci_hi']})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR, 'nb8_hard_negative_summary.csv'), index=False)

print("\n=== Within-pair accuracy (chance = 50%), N per subset in brackets ===")
piv = summary[summary['negative'] == 'hard'].pivot_table(index='model', columns='subset', values='pair_acc')
piv = piv[list(SUBSETS)] * 100
print(f"subset N: { {s: int(m.sum()) for s, m in SUBSETS.items()} }")
print(piv.round(1).to_string())
print("\nControls (all segments): uniform-random / frequency-matched-random negatives")
print((summary[summary['negative'] != 'hard'].pivot_table(index='model', columns='negative', values='pair_acc')
       * 100).round(1).to_string())


# ── 5. V-JEPA2 minus each DINO head ──────────────────────────────────────────
VJ = [m for m in models if m.startswith('vjepa2')]
DINO = [m for m in models if m.startswith('dino')]
diff_rows = []
for vj in VJ:
    for dn in DINO:
        for sub, m in SUBSETS.items():
            d = pairs.loc[m, f'{vj}__hard'].astype(float) - pairs.loc[m, f'{dn}__hard'].astype(float)
            cb = lib.cluster_bootstrap(d, pairs.loc[m, 'video_id'])
            diff_rows.append({'vjepa2': vj, 'dino': dn, 'subset': sub, 'n': int(m.sum()),
                              'n_videos': cb['n_videos'], 'discordance': (d != 0).mean(),
                              'diff': cb['mean'], 'ci_lo': cb['ci_lo'], 'ci_hi': cb['ci_hi'],
                              'p_value': cb['p_value']})
diffs = pd.DataFrame(diff_rows)
diffs.to_csv(os.path.join(OUT_DIR, 'nb8_hard_negative_diffs.csv'), index=False)

print("\n=== V-JEPA2 minus DINO, hard-pair accuracy (pp, video-cluster 95% CI) ===")
for r in diffs.itertuples():
    print(f"  {r.vjepa2:<22} - {r.dino:<13} [{r.subset:<11} n={r.n:>4}] "
          f"{r.diff*100:+6.2f}  [{r.ci_lo*100:+.2f}, {r.ci_hi*100:+.2f}]  p={r.p_value:.3f}")

best_dino = max(DINO, key=lambda m: pairs[f'{m}__hard'].mean())
print(f"\n=== Pre-registered check (all segments, vs best DINO head = {best_dino}) ===")
for vj in VJ:
    r = diffs[(diffs['vjepa2'] == vj) & (diffs['dino'] == best_dino) & (diffs['subset'] == 'all')].iloc[0]
    hit = r['diff'] >= DECISION_PP and r['ci_lo'] > 0
    half_width = (r['ci_hi'] - r['ci_lo']) / 2
    print(f"  {vj}: {r['diff']*100:+.2f}pp [{r['ci_lo']*100:+.2f}, {r['ci_hi']*100:+.2f}] -> "
          f"{'POSITIVE' if hit else 'null'}  (CI half-width {half_width*100:.1f}pp; "
          f"80%-power MDE ~{half_width/1.96*2.8*100:.1f}pp)")
if 'dinov3' not in models:
    print("  NOTE: DINOv3 missing (weights absent) -- 'best DINO' is over DINOv2 heads only.")
print("DONE")
