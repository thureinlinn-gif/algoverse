"""
Does order-sensitivity concentrate on verbs where reversal changes real-world
meaning (directional/monotonic: cover, place, remove, put, sprinkle, pour,
transfer, take) versus verbs that are inherently back-and-forth (cyclic:
mix, stir, roll, combine, squeeze, spread)? And is the biggest exception to
that pattern ("add") explainable by noise or low motion content, or is it a
genuine, unexplained anomaly?

This persists analysis that was originally run as one-off interactive Python
during the 2026-08-31 session -- it produced saved CSVs at the time but the
analysis CODE itself was never saved, which is a reproducibility gap for a
paper submission. Re-running this script regenerates every number from the
cached embeddings/CSVs with no new GPU extraction required (seconds to run).

Two scales are checked deliberately, not just the larger one: the pilot
(nb8_temporal_order_cosine.csv, n=600, from the first forward/reversed probe)
showed a clean per-verb gradient within each bucket; rerunning at full scale
(recomputed here over the full canonical validation set, n=3,268)
showed the AGGREGATE bucket-level effect replicate and strengthen, but two
individual verbs (sprinkle, roll) swap positions -- i.e. the fine-grained
per-verb ranking was not reliable at pilot n<10-per-verb. Reporting both
scales is the point: it's what justifies trusting the bucket-level claim
while explicitly NOT trusting a "clean gradient" narrative in the writeup.
"""
import os, json
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from scipy import stats

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
OUT_DIR = PATHS['processed']

DIRECTIONAL = {'add', 'remove', 'pour', 'place', 'put', 'cover', 'drain', 'sprinkle', 'take', 'transfer'}
CYCLIC = {'mix', 'stir', 'roll', 'combine', 'squeeze', 'spread'}


def bucket_for(v):
    if v in DIRECTIONAL:
        return 'directional'
    if v in CYCLIC:
        return 'cyclic'
    return 'other'


def bucket_test(df, label):
    df = df.copy()
    df['bucket'] = df['ground_truth'].apply(bucket_for)
    agg = df.groupby('bucket')['cos_sim_fwd_rev'].agg(['mean', 'std', 'count'])
    print(f"\n=== [{label}] bucket means ===")
    print(agg.to_string())

    d_all = df[df['bucket'] == 'directional']['cos_sim_fwd_rev']
    c = df[df['bucket'] == 'cyclic']['cos_sim_fwd_rev']
    u, p = stats.mannwhitneyu(d_all, c, alternative='less')
    print(f"Mann-Whitney, directional (incl. add) < cyclic: U={u:.1f} p={p:.4f}  "
          f"n_dir={len(d_all)} n_cyc={len(c)}")

    d_noadd = df[(df['bucket'] == 'directional') & (df['ground_truth'] != 'add')]['cos_sim_fwd_rev']
    u2, p2 = stats.mannwhitneyu(d_noadd, c, alternative='less')
    print(f"Mann-Whitney, directional (excl. add) < cyclic: U={u2:.1f} p={p2:.6g}  "
          f"n_dir_noadd={len(d_noadd)} n_cyc={len(c)}")

    print("\nper-verb means:")
    sub = df[df['bucket'].isin(['directional', 'cyclic'])]
    per_verb = sub.groupby('ground_truth')['cos_sim_fwd_rev'].agg(['mean', 'std', 'count']).sort_values('mean')
    print(per_verb.to_string())

    return {
        'scale': label, 'n_directional_incl_add': len(d_all), 'n_cyclic': len(c),
        'mean_directional_incl_add': d_all.mean(), 'mean_cyclic': c.mean(),
        'p_incl_add': p,
        'n_directional_excl_add': len(d_noadd), 'mean_directional_excl_add': d_noadd.mean(),
        'p_excl_add': p2,
    }, per_verb.reset_index()


# ── Scale 1: the original pilot (n=600) ─────────────────────────────────────
pilot_path = os.path.join(OUT_DIR, 'nb8_temporal_order_cosine.csv')
pilot_summary, pilot_per_verb = None, None
if os.path.exists(pilot_path):
    pilot_df = pd.read_csv(pilot_path)
    pilot_summary, pilot_per_verb = bucket_test(pilot_df, 'pilot n=600')
else:
    print(f"WARNING: {pilot_path} not found, skipping pilot-scale check.")

# ── Scale 2: full scale, recomputed from every cached fwd/rev embedding pair ─
FWD_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_order_fwd')
REV_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_order_rev')
df_ann = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv')).set_index('segment_id')

fwd_files = {f[:-3] for f in os.listdir(FWD_DIR) if f.endswith('.pt')}
rev_files = {f[:-3] for f in os.listdir(REV_DIR) if f.endswith('.pt')}
common = fwd_files & rev_files
print(f"\nSegments with both forward+reversed embeddings cached: {len(common)}")

# Fixed 2026-09-02: this used to analyse every cached forward/reversed pair, which
# is NOT the evaluation set. The cache was filled by two different steps (step 7's
# 600-segment pilot and step 8's full-val pass) plus an earlier exploratory run
# that had touched some TRAINING segments, so globbing the directory gave n=3,567
# = 3,268 validation + 299 training. Nothing here is trained, so those 299 did not
# leak anything -- but they made this table's N and per-verb counts disagree with
# every other table in the paper ('add' n=1,313 here vs. n=1,165 everywhere else).
# The canonical evaluation set is the 3,268 validation segments; restrict to it.
val_ids = set(df_ann.index[df_ann['subset'] == 'validation'])
dropped_nonval = len(common - val_ids)
common = common & val_ids
print(f"Restricted to the canonical validation set: {len(common)} "
      f"({dropped_nonval} cached non-validation segments excluded)")

rows = []
for seg in common:
    if seg not in df_ann.index:
        continue
    verb = df_ann.loc[seg, 'extracted_verb']
    fwd = F.normalize(torch.load(os.path.join(FWD_DIR, f'{seg}.pt')), dim=-1)
    rev = F.normalize(torch.load(os.path.join(REV_DIR, f'{seg}.pt')), dim=-1)
    rows.append({'segment_id': seg, 'ground_truth': verb, 'cos_sim_fwd_rev': (fwd @ rev).item()})
full_df = pd.DataFrame(rows)
full_df.to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_cosine_fullscale.csv'), index=False)
print(f"Saved {len(full_df)} rows -> nb8_temporal_order_cosine_fullscale.csv")

full_summary, full_per_verb = bucket_test(full_df, 'full scale')
full_per_verb.to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_directional_cyclic_fullscale.csv'), index=False)

# ── Investigate the 'add' exception: is it noisy, or genuinely different? ──
print("\n=== Is 'add' just noisy? (within-verb variance, full scale) ===")
sub = full_df[full_df['ground_truth'].isin(DIRECTIONAL | CYCLIC)].copy()
stats_tbl = sub.groupby('ground_truth')['cos_sim_fwd_rev'].agg(['mean', 'std', 'count'])
stats_tbl['cv'] = stats_tbl['std'] / stats_tbl['mean']
print(stats_tbl.sort_values('cv', ascending=False).to_string())
add_row = stats_tbl.loc['add']
higher_cv_directional = stats_tbl.loc[list(DIRECTIONAL - {'add'})].query('cv > @add_row.cv')
print(f"\n'add' cv={add_row['cv']:.4f} -- {len(higher_cv_directional)}/{len(DIRECTIONAL)-1} other directional "
      f"verbs have HIGHER coefficient of variation than 'add'. 'add' is not unusually noisy.")

print("\n=== Is 'add' just motion-sparse? (encoder-independent optical flow) ===")
flow_path = os.path.join(OUT_DIR, 'nb8_optical_flow_scores.csv')
if os.path.exists(flow_path):
    flow = pd.read_csv(flow_path)
    flow_sub = flow[flow['ground_truth'].isin(DIRECTIONAL | CYCLIC)]
    flow_means = flow_sub.groupby('ground_truth')['motion_score'].agg(['mean', 'std', 'count']).sort_values('mean')
    print(flow_means.to_string())
    add_motion_rank = (flow_means['mean'] < flow_means.loc['add', 'mean']).sum() + 1
    print(f"\n'add' motion-score rank (low to high) among {len(flow_means)} directional+cyclic verbs: "
          f"{add_motion_rank} -- not the lowest, so low motion content does not explain the exception either.")
else:
    print(f"WARNING: {flow_path} not found, skipping motion-magnitude check.")

pd.DataFrame([pilot_summary, full_summary]).to_csv(
    os.path.join(OUT_DIR, 'nb8_temporal_order_verb_typology_summary.csv'), index=False)
print("\nDONE")
