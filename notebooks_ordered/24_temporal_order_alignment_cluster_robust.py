"""
Honest CI for Claim B, per mentor feedback (2026-09-08): script 15's bootstrap
resamples the 3,268 SEGMENTS directly (rng.integers(0, n, n) over
diff_fwd_minus_rev), but segments cluster inside ~450 validation videos
(2-16 segments/video, mean 7.3), so that CI is a little tight -- not wrong in
sign or magnitude, just narrower than it should be.

Claim B: using the real trained V-JEPA2+DinoTXT alignment head, does a
segment's true-verb text score higher against the forward clip than the
reversed one? (Forward wins 64.5% of the time, mean gap +0.0181.)

This script re-bootstraps the same cached per-segment diffs two ways, side
by side, entirely from data already on disk -- no GPU, no retraining,
no new alignment-head forward passes, seconds to run:

  1. segment-level bootstrap -- script 15's original CI, reproduced here for
                                 reference.
  2. video-cluster bootstrap -- resample validation VIDEOS (not segments)
                                 with replacement, pool their segments' diffs
                                 each replicate, percentile CI. This is the
                                 honest one.

Reads : processed/nb8_temporal_order_text_alignment.csv (script 15's output)
        annotations/segments_verbs_timed_clean.csv (for video_id)
Writes: processed/nb8_temporal_order_alignment_cluster_robust.csv
"""
import os, json
import numpy as np
import pandas as pd

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
OUT_DIR = PATHS['processed']

# ── Load + join video_id ─────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_text_alignment.csv'))
ann = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
df = df.merge(ann[['segment_id', 'video_id']], on='segment_id', how='left')
missing_vid = int(df['video_id'].isna().sum())
if missing_vid:
    print(f"WARNING: {missing_vid} segments could not be matched to a video_id -- dropping them")
    df = df.dropna(subset=['video_id'])

n = len(df)
diffs = df['diff_fwd_minus_rev'].to_numpy()
obs_mean = diffs.mean()
n_videos = df['video_id'].nunique()
print(f"N segments={n}  N videos={n_videos}  mean diff(fwd-rev)={obs_mean:+.4f}")

N_BOOT = 10000
rng = np.random.default_rng(0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("VIEW 1 -- segment-level bootstrap (script 15's original -- reproduced for reference)")
print("=" * 78)
boot_seg = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
seg_ci_lo, seg_ci_hi = np.percentile(boot_seg, [2.5, 97.5])
seg_p = 2 * min((boot_seg <= 0).mean(), (boot_seg >= 0).mean())
seg_halfwidth = (seg_ci_hi - seg_ci_lo) / 2
print(f"mean diff = {obs_mean:+.4f}  95% CI = [{seg_ci_lo:+.4f}, {seg_ci_hi:+.4f}]  "
      f"bootstrap p={seg_p:.4g}")
print(f"CI half-width = {seg_halfwidth:.5f}  "
      f"(observed mean is {obs_mean / seg_halfwidth:.1f} half-widths from zero)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("VIEW 2 -- video-cluster bootstrap (resample videos, not segments -- the honest one)")
print("=" * 78)
rng2 = np.random.default_rng(0)
vstats = df.groupby('video_id')['diff_fwd_minus_rev'].agg(['count', 'sum']).reset_index()
n_a = vstats['count'].to_numpy()
sum_a = vstats['sum'].to_numpy()

idx = rng2.integers(0, n_videos, size=(N_BOOT, n_videos))
boot_n = n_a[idx].sum(axis=1)
boot_sum = sum_a[idx].sum(axis=1)
boot_cluster = boot_sum / boot_n

clu_ci_lo, clu_ci_hi = np.percentile(boot_cluster, [2.5, 97.5])
clu_p = 2 * min((boot_cluster <= 0).mean(), (boot_cluster >= 0).mean())
clu_halfwidth = (clu_ci_hi - clu_ci_lo) / 2
n_widths = obs_mean / clu_halfwidth
print(f"mean diff = {obs_mean:+.4f}  95% CI = [{clu_ci_lo:+.4f}, {clu_ci_hi:+.4f}]  "
      f"bootstrap p={clu_p:.4g}")
print(f"CI half-width = {clu_halfwidth:.5f}  "
      f"(observed mean is {n_widths:.1f} half-widths / cluster-CI-widths from zero)")
print(f"\nCI widened by {(clu_ci_hi - clu_ci_lo) / (seg_ci_hi - seg_ci_lo):.2f}x "
      f"going from segment-level to video-cluster bootstrap.")
print(f"CI excludes zero: {not (clu_ci_lo <= 0 <= clu_ci_hi)}")

out = pd.DataFrame([
    {'view': 'segment_level', 'n_units': n, 'mean_diff': obs_mean,
     'ci_lo': seg_ci_lo, 'ci_hi': seg_ci_hi, 'p_boot': seg_p,
     'halfwidth': seg_halfwidth, 'widths_from_zero': obs_mean / seg_halfwidth},
    {'view': 'video_cluster', 'n_units': n_videos, 'mean_diff': obs_mean,
     'ci_lo': clu_ci_lo, 'ci_hi': clu_ci_hi, 'p_boot': clu_p,
     'halfwidth': clu_halfwidth, 'widths_from_zero': n_widths},
])
out.to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_alignment_cluster_robust.csv'), index=False)
print("\nSaved -> nb8_temporal_order_alignment_cluster_robust.csv")
print("\nDONE")
