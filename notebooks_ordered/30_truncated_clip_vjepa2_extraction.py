"""
Truncated-clip embedding extraction, V-JEPA2 side: same idea as script 29,
but V-JEPA2's joint spatiotemporal pooling can't be truncated after the fact
the way a per-frame mean can (it's not separable), so this needs a fresh
forward pass per truncation level, over the validation set only (3,268
segments, not all 13,003).

Fairness constraint (mentor, 2026-09-08): vision encoder + pretrained
attentive pooler ONLY -- identical to script 12's already-validated
methodology. `encoder = full_model.vjepa2`, `pooler = full_model.pooler`,
`skip_predictor=True` (skips V-JEPA2's own JEPA predictor network -- not used
here), and the classification head's weights are never called in the forward
pass. No new model, no different loading path from what's already in the
pipeline -- this script changes ONLY which frames are sampled per segment.

Truncation: FRAMES_PER_CLIP is fixed by the checkpoint (16, fpc16) regardless
of input span, so "seeing only the first frac of the clip" means sampling
those 16 frames evenly across [start_idx, start_idx + frac*(end_idx-start_idx)]
instead of the full [start_idx, end_idx] -- same frame COUNT, narrower time
WINDOW, which is the correct way to simulate partial context for a model that
always consumes a fixed number of frames.

Cost: ~35-40 min per truncation level at script 12's measured rate (2.44h /
13,003 segments), x3 levels (25/50/75%) over 3,268 segments ~ 1.5-2h total.
100% is not re-extracted here -- already on disk from script 12.

Reads : raw video (independent of script 22/26 -- V-JEPA2 needs its own
        frames, sampled to FRAMES_PER_CLIP, not DINOv2's up-to-50-frame set)
Writes: embeddings/<...>/vjepa2_pretrainedpool_trunc{25,50,75}/<segment_id>.pt
"""
import os, sys, json, time
import numpy as np
import torch
import pandas as pd
from decord import VideoReader, cpu
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda')

MODEL_NAME = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
VIDEO_DIR = PATHS['videos']
TRUNCATIONS = [0.25, 0.50, 0.75]

processor = VJEPA2VideoProcessor.from_pretrained(MODEL_NAME)
full_model = VJEPA2ForVideoClassification.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, ignore_mismatched_sizes=True,
).to(DEVICE)
encoder = full_model.vjepa2
pooler = full_model.pooler
for p in encoder.parameters():
    p.requires_grad = False
for p in pooler.parameters():
    p.requires_grad = False
encoder.eval(); pooler.eval()
FRAMES_PER_CLIP = encoder.config.frames_per_clip
print(f"Loaded {MODEL_NAME}. frames_per_clip={FRAMES_PER_CLIP} (vision encoder + pretrained "
      f"pooler ONLY -- no classifier head, no predictor -- matching script 12)", flush=True)

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
print(f"Validation segments: {len(df_val):,}", flush=True)

for frac in TRUNCATIONS:
    pct = int(frac * 100)
    EMB_DIR = os.path.join(PATHS['embeddings'], f'vjepa2_pretrainedpool_trunc{pct}')
    os.makedirs(EMB_DIR, exist_ok=True)
    print(f"\n=== Truncation frac={frac} -> {EMB_DIR} ===", flush=True)

    saved = existing = skipped = 0
    start_time = time.time()
    by_video = df_val.groupby('video_id')
    for vi, (video_id, grp) in enumerate(tqdm(by_video, total=by_video.ngroups, desc=f"frac={frac}")):
        todo = [row for _, row in grp.iterrows()
                if not os.path.exists(os.path.join(EMB_DIR, f"{row['segment_id']}.pt"))]
        if not todo:
            existing += len(grp)
            continue
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            skipped += len(todo)
            continue
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
        except Exception as e:
            print(f"  ERROR opening {video_id}: {e}", flush=True)
            skipped += len(todo)
            continue
        fps = vr.get_avg_fps()

        for row in todo:
            out_path = os.path.join(EMB_DIR, f"{row['segment_id']}.pt")
            try:
                start_idx = max(0, int(row['start_sec'] * fps))
                end_idx = min(len(vr) - 1, int(row['end_sec'] * fps))
                truncated_end = start_idx + max(1, int(frac * (end_idx - start_idx)))
                truncated_end = min(truncated_end, len(vr) - 1)
                if truncated_end <= start_idx:
                    truncated_end = start_idx + 1
                indices = np.linspace(start_idx, truncated_end, FRAMES_PER_CLIP, dtype=int)
                frames = vr.get_batch(indices).asnumpy()
                clip = torch.from_numpy(frames).permute(0, 3, 1, 2)
                inputs = processor(clip, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = encoder(**inputs, skip_predictor=True)
                    pooled = pooler(out.last_hidden_state).squeeze(0).cpu()
                torch.save(pooled, out_path)
                saved += 1
            except Exception as e:
                print(f"  ERROR {row['segment_id']}: {e}", flush=True)
                skipped += 1

        if (vi + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / (saved + 1e-9)
            remaining = len(df_val) - saved - existing - skipped
            eta_min = rate * remaining / 60
            print(f"  [{vi+1}/{by_video.ngroups} videos] saved={saved:,} existing={existing:,} "
                  f"skipped={skipped:,} elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min", flush=True)

    total_files = len([f for f in os.listdir(EMB_DIR) if f.endswith('.pt')])
    print(f"frac={frac} done. Saved: {saved:,}  Existing: {existing:,}  Skipped: {skipped:,}  "
          f"Total: {total_files:,}", flush=True)

print("\nALL TRUNCATION LEVELS DONE")
