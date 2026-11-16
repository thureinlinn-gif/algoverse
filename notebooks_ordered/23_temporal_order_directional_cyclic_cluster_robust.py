"""
Robustness check for script 17's directional-vs-cyclic claim, per mentor
feedback (2026-09-08): the published p=1.05e-07 treats 3,268 segments as
independent draws, but the claim is actually about 16 VERBS (10 directional,
6 cyclic), and segments cluster both within verbs and within validation
videos. With group sizes 10 and 6, the smallest one-sided p-value an exact
Mann-Whitney can ever return -- even under perfect separation -- is
1/C(16,6) = 1.2485e-04. The published number sits ~840x below that floor,
which is only possible because the independence assumption is violated.

This script reports three views of the same effect side by side, entirely
from data already on disk -- no GPU, no new extraction, seconds to run:

  1. segment level    -- script 17's original test, reproduced here for
                          reference and explicitly flagged as sitting below
                          its own achievable floor (not a valid inference).
  2. verb level        -- the correct unit for the claim: collapse to one
                          mean cos_sim_fwd_rev per verb (10 directional + 6
                          cyclic numbers), exact Mann-Whitney over those.
                          THIS is the number to report going forward.
  3. video-cluster
     bootstrap         -- resample validation VIDEOS (not segments) with
                          replacement, recompute the directional-minus-cyclic
                          mean gap per replicate, percentile CI. Addresses
                          the same non-independence from the resampling side
                          rather than the rank-test side.

Reads : processed/nb8_temporal_order_cosine_fullscale.csv (script 17's output)
        annotations/segments_verbs_timed_clean.csv (for video_id)
Writes: processed/nb8_temporal_order_directional_cyclic_verblevel.csv
        processed/nb8_temporal_order_directional_cyclic_videocluster_bootstrap.csv
"""
import os, json
from math import comb
import numpy as np
import pandas as pd
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


# ── Load + join video_id ─────────────────────────────────────────────────────
cos_path = os.path.join(OUT_DIR, 'nb8_temporal_order_cosine_fullscale.csv')
df = pd.read_csv(cos_path)
df['bucket'] = df['ground_truth'].apply(bucket_for)

ann = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
df = df.merge(ann[['segment_id', 'video_id']], on='segment_id', how='left')
missing_vid = int(df['video_id'].isna().sum())
if missing_vid:
    print(f"WARNING: {missing_vid} segments could not be matched to a video_id -- dropping them")
    df = df.dropna(subset=['video_id'])

sub = df[df['bucket'].isin(['directional', 'cyclic'])].copy()
print(f"n_directional_segments={len(sub[sub.bucket == 'directional'])}  "
      f"n_cyclic_segments={len(sub[sub.bucket == 'cyclic'])}  "
      f"n_videos={sub['video_id'].nunique()}")

CONDITIONS = [(False, 'incl. add'), (True, 'excl. add')]

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("VIEW 1 -- segment level (script 17's original unit; below its own floor)")
print("=" * 78)
for excl, tag in CONDITIONS:
    s = sub[sub['ground_truth'] != 'add'] if excl else sub
    d = s[s.bucket == 'directional']['cos_sim_fwd_rev']
    c = s[s.bucket == 'cyclic']['cos_sim_fwd_rev']
    u, pval = stats.mannwhitneyu(d, c, alternative='less')
    print(f"\n{tag}: n_dir_segments={len(d)} n_cyc_segments={len(c)}")
    print(f"Mann-Whitney (asymptotic, segment-level draws): U={u:.1f} p={pval:.6g}")
    print("-- NOT a valid inference: segments are not independent draws, they cluster "
          "inside 16 verbs and inside videos. Reproduced only to show what changed.")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("VIEW 2 -- verb level (the correct unit -- REPORT THIS ONE)")
print("=" * 78)
verb_level_rows = []
for excl, tag in CONDITIONS:
    s = sub[sub['ground_truth'] != 'add'] if excl else sub
    per_verb = s.groupby(['ground_truth', 'bucket'])['cos_sim_fwd_rev'].mean().reset_index()
    d = per_verb[per_verb.bucket == 'directional']['cos_sim_fwd_rev']
    c = per_verb[per_verb.bucket == 'cyclic']['cos_sim_fwd_rev']
    n_d, n_c = len(d), len(c)
    floor = 1.0 / comb(n_d + n_c, n_c)
    u, pval = stats.mannwhitneyu(d, c, alternative='less', method='exact')
    print(f"\n{tag}: n_directional_verbs={n_d}  n_cyclic_verbs={n_c}  "
          f"exact floor = 1/C({n_d + n_c},{n_c}) = {floor:.6g}")
    print(f"Mann-Whitney (exact, verb-level draws): U={u:.1f} p={pval:.6g}")
    print(per_verb.sort_values('cos_sim_fwd_rev').to_string(index=False))
    verb_level_rows.append({
        'condition': tag, 'n_directional_verbs': n_d, 'n_cyclic_verbs': n_c,
        'exact_floor_p': floor, 'U': u, 'p_exact': pval,
        'mean_directional': d.mean(), 'mean_cyclic': c.mean(),
    })
pd.DataFrame(verb_level_rows).to_csv(
    os.path.join(OUT_DIR, 'nb8_temporal_order_directional_cyclic_verblevel.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("VIEW 3 -- video-cluster bootstrap (resample videos, not segments)")
print("=" * 78)
rng = np.random.default_rng(0)
N_BOOT = 10000
cluster_rows = []
for excl, tag in CONDITIONS:
    s = sub[sub['ground_truth'] != 'add'] if excl else sub

    # Per-video sufficient stats -- vectorized resampling, no per-replicate pandas work.
    vstats = s.groupby('video_id').apply(lambda g: pd.Series({
        'n_dir': int((g.bucket == 'directional').sum()),
        'sum_dir': g.loc[g.bucket == 'directional', 'cos_sim_fwd_rev'].sum(),
        'n_cyc': int((g.bucket == 'cyclic').sum()),
        'sum_cyc': g.loc[g.bucket == 'cyclic', 'cos_sim_fwd_rev'].sum(),
    }), include_groups=False).reset_index()
    n_videos = len(vstats)
    n_dir_a, sum_dir_a = vstats['n_dir'].to_numpy(), vstats['sum_dir'].to_numpy()
    n_cyc_a, sum_cyc_a = vstats['n_cyc'].to_numpy(), vstats['sum_cyc'].to_numpy()

    obs_diff = (sum_dir_a.sum() / n_dir_a.sum()) - (sum_cyc_a.sum() / n_cyc_a.sum())

    idx = rng.integers(0, n_videos, size=(N_BOOT, n_videos))  # resample video clusters w/ replacement
    boot_n_dir = n_dir_a[idx].sum(axis=1)
    boot_sum_dir = sum_dir_a[idx].sum(axis=1)
    boot_n_cyc = n_cyc_a[idx].sum(axis=1)
    boot_sum_cyc = sum_cyc_a[idx].sum(axis=1)
    valid = (boot_n_dir > 0) & (boot_n_cyc > 0)
    n_degenerate = int((~valid).sum())
    boot_diffs = (boot_sum_dir[valid] / boot_n_dir[valid]) - (boot_sum_cyc[valid] / boot_n_cyc[valid])

    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_boot = 2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean())
    excludes_zero = not (ci_lo <= 0 <= ci_hi)
    print(f"\n{tag}: n_videos={n_videos}  degenerate_reps_dropped={n_degenerate}/{N_BOOT}")
    print(f"obs(mean_directional - mean_cyclic) = {obs_diff:+.4f}")
    print(f"95% cluster-bootstrap CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
          f"bootstrap p={p_boot:.4g}  CI excludes zero: {excludes_zero}")
    cluster_rows.append({
        'condition': tag, 'n_videos': n_videos, 'obs_diff': obs_diff,
        'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p_boot': p_boot, 'ci_excludes_zero': excludes_zero,
    })
pd.DataFrame(cluster_rows).to_csv(
    os.path.join(OUT_DIR, 'nb8_temporal_order_directional_cyclic_videocluster_bootstrap.csv'), index=False)

print("\nSaved -> nb8_temporal_order_directional_cyclic_verblevel.csv")
print("Saved -> nb8_temporal_order_directional_cyclic_videocluster_bootstrap.csv")
print("\nDONE")
