"""
Literal, encoder-independent motion measurement per segment via dense optical
flow (Farneback), computed directly from raw video pixels -- no DINOv2/DINOv3/
V-JEPA2 involved at all, so zero circularity risk for testing any of them.

For each val segment: sample ~10 evenly-spaced frames, downsample, compute
Farneback flow between consecutive pairs, average per-pixel magnitude across
all pairs -> one motion score per segment. Then bucket the already-computed
representative-checkpoint predictions (all three backbones) by motion-score
tercile and compare.
"""
import os, json
import numpy as np
import pandas as pd
import cv2
from decord import VideoReader, cpu
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)

OUT_DIR = PATHS['processed']
VIDEO_DIR = PATHS['videos']

N_FRAMES = 10          # sampled frames per segment for flow computation
FLOW_SIZE = 128        # downsample to this (square) before computing flow

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
print(f"Val segments: {len(df_val):,}  Unique videos: {df_val['video_id'].nunique():,}", flush=True)


def motion_score_for_frames(frames_gray):
    """frames_gray: list of grayscale np arrays, same size. Returns mean
    per-pixel flow magnitude averaged across all consecutive pairs."""
    if len(frames_gray) < 2:
        return 0.0
    mags = []
    for i in range(len(frames_gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i], frames_gray[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(mag.mean())
    return float(np.mean(mags))


records = []
by_video = df_val.groupby('video_id')
errors = 0

for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="Optical flow"):
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        errors += len(grp)
        continue
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        errors += len(grp)
        continue
    fps = vr.get_avg_fps()

    for _, row in grp.iterrows():
        start_idx = max(0, int(row['start_sec'] * fps))
        end_idx = min(len(vr) - 1, int(row['end_sec'] * fps))
        if start_idx >= end_idx:
            records.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                             'motion_score': 0.0})
            continue
        n = min(N_FRAMES, end_idx - start_idx + 1)
        indices = np.linspace(start_idx, end_idx, n, dtype=int)
        try:
            frames = vr.get_batch(indices).asnumpy()  # (n, H, W, 3)
        except Exception:
            errors += 1
            continue
        frames_gray = [
            cv2.cvtColor(cv2.resize(f, (FLOW_SIZE, FLOW_SIZE)), cv2.COLOR_RGB2GRAY)
            for f in frames
        ]
        score = motion_score_for_frames(frames_gray)
        records.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                         'motion_score': score})

flow_df = pd.DataFrame(records)
print(f"\nComputed motion scores for {len(flow_df):,} segments ({errors} errors/skipped)", flush=True)
flow_df.to_csv(os.path.join(OUT_DIR, 'nb8_optical_flow_scores.csv'), index=False)

# ── Sanity check: per-verb average motion score (verbs with >=15 val segments) ──
verb_counts = flow_df['ground_truth'].value_counts()
reliable_verbs = verb_counts[verb_counts >= 15].index
verb_motion = flow_df[flow_df['ground_truth'].isin(reliable_verbs)].groupby('ground_truth')['motion_score'].agg(['mean', 'count'])
verb_motion = verb_motion.sort_values('mean', ascending=False)
print("\n=== Highest average motion score (n>=15) ===")
print(verb_motion.head(15).to_string())
print("\n=== Lowest average motion score (n>=15) ===")
print(verb_motion.tail(15).to_string())
verb_motion.to_csv(os.path.join(OUT_DIR, 'nb8_optical_flow_by_verb.csv'))

# ── Bucket by tercile, compare all three backbones' already-computed ranks ──
flow_df['tercile'] = pd.qcut(flow_df['motion_score'], 3, labels=['low_motion', 'mid_motion', 'high_motion'])
flow_df = flow_df.set_index('segment_id')

rows = []
for tag in ['dinov2', 'dinov3', 'vjepa2']:
    ranks = pd.read_csv(os.path.join(OUT_DIR, f'nb8_ranks_{tag}_representative.csv')).set_index('segment_id')
    merged = ranks.join(flow_df[['tercile']], how='inner')
    for tercile, grp in merged.groupby('tercile', observed=True):
        rows.append({
            'model': tag, 'tercile': tercile, 'n': len(grp),
            'micro_top1': grp['top1_correct'].mean(), 'micro_top5': grp['top5_correct'].mean(),
            'macro_top1': grp.groupby('ground_truth')['top1_correct'].mean().mean(),
            'mrr': grp['reciprocal_rank'].mean() if 'reciprocal_rank' in grp.columns else float('nan'),
        })
final_df = pd.DataFrame(rows)
print("\n=== Performance by optical-flow tercile ===")
print(final_df.to_string(index=False))
final_df.to_csv(os.path.join(OUT_DIR, 'nb8_vjepa2_advantage_by_optical_flow.csv'), index=False)
print(f"\nSaved -> {os.path.join(OUT_DIR, 'nb8_vjepa2_advantage_by_optical_flow.csv')}", flush=True)
print("DONE", flush=True)
