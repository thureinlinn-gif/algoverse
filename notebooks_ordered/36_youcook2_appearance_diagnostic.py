"""
Phase 7, Experiment C: how much of YouCook2 verb classification is solvable from
appearance and context alone. This is the section that explains the backbone null
(V-JEPA 2's own Table 4 has COIN, the closest instructional-video benchmark, as an
exact V-JEPA2 / DINOv2 tie), so it is needed whatever A and B show. No GPU, no
training: existing heads through script 32's cached features, plus saved CSVs.

Tables (all on the canonical 3,268 validation segments):
 1. Overall: blind always-'add'; masked-caption text model (script 34, no video at
    all); single middle frame through the DINOv2 / DINOv3 representative heads
    (nb8_singleframe_motion_test.py); full clip for every representative head.
 2. Split by whether the objects determine the verb, two definitions:
      object rule : the segment's direct object (script 34's parse) has a train-majority
                    verb; 'determines' if that verb is the true verb, 'does_not' if
                    not, 'unseen' if the object never occurs in train captions.
      text model  : the masked-caption text model is right / wrong.
    (The n=1,377 quoted in the plan came from an unsaved ad-hoc computation whose
    definition was not recorded; these two definitions replace it.)
 3. Duration quartiles: Top-1 per head by segment length.

Reads : processed/nb8_text_baseline_masked_caption.csv (34), nb8_singleframe_ranks_{dinov2,dinov3}.csv,
        nb8_ranks_dinov3_representative.csv (fallback if DINOv3 weights absent), heads via 33
Writes: processed/nb8_appearance_diagnostic_overall.csv, nb8_appearance_diagnostic_object_split.csv,
        nb8_appearance_diagnostic_duration.csv
~1 min.
"""
import os, sys, importlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
lib = importlib.import_module('33_linear_head_lib')
OUT_DIR = lib.OUT_DIR

df, verb_vocab, label2id = lib.load_segments()
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
val = df_val.set_index('segment_id')
text = pd.read_csv(os.path.join(OUT_DIR, 'nb8_text_baseline_masked_caption.csv')).set_index('segment_id')
N = len(df_val)

# ── Per-segment correctness for every row of the table ───────────────────────
correct, preds = {}, {}
correct['blind_always_add'] = (val['extracted_verb'] == 'add')
preds['blind_always_add'] = pd.Series('add', index=val.index)
tv = text.loc[val.index]
correct['text_masked_caption'] = tv['text_top1_correct'].astype(bool)
preds['text_masked_caption'] = tv['text_top1_pred']
for bb in ('dinov2', 'dinov3'):
    sf = pd.read_csv(os.path.join(OUT_DIR, f'nb8_singleframe_ranks_{bb}.csv')).set_index('segment_id')
    correct[f'{bb}_single_frame'] = sf['top1_correct'].astype(bool)
for tag in lib.HEADS:
    r = lib.load_representative_head(tag, df_val, verb_vocab, verbose=False)
    if r is None:
        ranks = pd.read_csv(os.path.join(OUT_DIR, lib.HEADS[tag]['published'])).set_index('segment_id')
        print(f"[{tag}] text tower not cached -- using its published ranks CSV")
    else:
        ranks = r[3]['ranks']
    correct[f'{tag}_full_clip'] = ranks['top1_correct'].astype(bool)
    if 'top1_pred' in ranks:
        preds[f'{tag}_full_clip'] = ranks['top1_pred']

DESCRIPTION = {
    'blind_always_add': 'no input: always "add"',
    'text_masked_caption': 'caption with the verb masked, no video (TF-IDF + LR)',
    'dinov2_single_frame': 'one middle frame, DINOv2 head', 'dinov3_single_frame': 'one middle frame, DINOv3 head',
}
rows = []
for name, c in correct.items():
    c = c.reindex(val.index)
    n = int(c.notna().sum())
    p = preds.get(name)
    mf1 = (f1_score(val.loc[p.index, 'extracted_verb'], p, average='macro', zero_division=0)
           if p is not None else np.nan)
    tag = name.replace('_full_clip', '')
    rows.append({'row': name, 'description': DESCRIPTION.get(name, f"full clip, {lib.HEAD_LABELS.get(tag, tag)}"),
                 'n': n, 'top1': c.astype(float).mean(), 'macro_f1': mf1})
overall = pd.DataFrame(rows)
overall.to_csv(os.path.join(OUT_DIR, 'nb8_appearance_diagnostic_overall.csv'), index=False)
print(f"=== 1. What appearance / context alone gets (N={N:,} val) ===")
print(overall.assign(top1=lambda d: (d.top1 * 100).round(2), macro_f1=lambda d: (d.macro_f1 * 100).round(2))
      .to_string(index=False))
for bb in ('dinov2', 'dinov3'):
    both = pd.concat([correct[f'{bb}_single_frame'], correct[f'{bb}_full_clip']], axis=1, join='inner')
    print(f"  {bb}: full clip - single frame = {(both.iloc[:, 1].mean() - both.iloc[:, 0].mean())*100:+.2f}pp "
          f"(N={len(both):,})")


# ── 2. Do the objects determine the verb? ────────────────────────────────────
tr = df[df['subset'] == 'training'].merge(text[['object']], left_on='segment_id', right_index=True)
majority = tr[tr['object'] != ''].groupby('object')['extracted_verb'].agg(lambda s: s.value_counts().idxmax())
obj = tv['object'].fillna('')
maj = obj.map(majority)
buckets = {
    'object rule: determines': (maj == val['extracted_verb']),
    'object rule: does_not': maj.notna() & (maj != val['extracted_verb']),
    'object rule: unseen object': maj.isna(),
    'text model: right': tv['text_top1_correct'].astype(bool),
    'text model: wrong': ~tv['text_top1_correct'].astype(bool),
}
split_rows = []
for bname, mask in buckets.items():
    ids = mask[mask].index
    for name, c in correct.items():
        if name in ('text_masked_caption',) and bname.startswith('text model'):
            continue
        cc = c.reindex(ids).dropna()
        split_rows.append({'bucket': bname, 'row': name, 'n': len(cc), 'top1': cc.astype(float).mean()})
split = pd.DataFrame(split_rows)
split.to_csv(os.path.join(OUT_DIR, 'nb8_appearance_diagnostic_object_split.csv'), index=False)
print(f"\n=== 2. Top-1 by whether the objects determine the verb ===")
piv = split.pivot_table(index='row', columns='bucket', values='top1') * 100
print("bucket N:", {b: int(m.sum()) for b, m in buckets.items()})
print(piv.round(1).to_string())


# ── 3. Duration ──────────────────────────────────────────────────────────────
dur = (val['end_sec'] - val['start_sec'])
qbin = pd.qcut(dur, 4, labels=[f'Q{i}' for i in range(1, 5)])
edges = pd.qcut(dur, 4, retbins=True)[1]
dur_rows = []
for q in qbin.cat.categories:
    ids = qbin[qbin == q].index
    for name, c in correct.items():
        cc = c.reindex(ids).dropna()
        dur_rows.append({'quartile': q, 'sec_lo': edges[int(q[1]) - 1], 'sec_hi': edges[int(q[1])],
                         'row': name, 'n': len(cc), 'top1': cc.astype(float).mean()})
dur_df = pd.DataFrame(dur_rows)
dur_df.to_csv(os.path.join(OUT_DIR, 'nb8_appearance_diagnostic_duration.csv'), index=False)
print(f"\n=== 3. Top-1 by segment duration quartile (edges, s: {np.round(edges, 1).tolist()}) ===")
print((dur_df.pivot_table(index='row', columns='quartile', values='top1') * 100).round(1).to_string())
print("DONE")
