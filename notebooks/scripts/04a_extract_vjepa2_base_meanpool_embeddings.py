"""
Re-extract V-JEPA2 embeddings using the actual base encoder
(facebook/vjepa2-vitl-fpc64-256) instead of the SSv2-action-classification-
finetuned checkpoint (facebook/vjepa2-vitl-fpc16-256-ssv2) NB6 originally used.

Why: the ssv2 checkpoint is fine-tuned on Something-Something-V2 (174 generic
object-manipulation classes, unrelated to cooking) -- if that fine-tuning
touched the encoder weights (typical unless explicitly frozen, which NB6 never
did), every V-JEPA2 embedding in this project has been specialized toward a
different task/domain rather than reflecting general video understanding.
The base checkpoint is the actual pretrained encoder, never task-finetuned,
and processes 64 frames/clip instead of 16 (4x the temporal context).

Same mean-pool-over-tubelet-tokens logic as NB6, output dim still 1024 (same
as before), just a different, more appropriate checkpoint and more frames.
Grouped by video (1,771 unique videos for 13,003 segments) to avoid reopening
the same file repeatedly.
"""
import os, sys, json, time
import numpy as np
import torch
import pandas as pd
from decord import VideoReader, cpu
from transformers import AutoModel, VJEPA2VideoProcessor
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda')

MODEL_NAME = 'facebook/vjepa2-vitl-fpc64-256'
EMB_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_base_dinotxt')
os.makedirs(EMB_DIR, exist_ok=True)
print(f"Saving embeddings to: {EMB_DIR}", flush=True)

processor = VJEPA2VideoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
FRAMES_PER_CLIP = model.config.frames_per_clip
print(f"Model loaded. frames_per_clip={FRAMES_PER_CLIP}  hidden_size={model.config.hidden_size}", flush=True)

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
print(f"Total segments: {len(df):,}  Unique videos: {df['video_id'].nunique():,}", flush=True)

VIDEO_DIR = PATHS['videos']
saved = existing = skipped = 0
start_time = time.time()

by_video = df.groupby('video_id')
for vi, (video_id, grp) in enumerate(tqdm(by_video, total=by_video.ngroups, desc="Extracting V-JEPA2 (base)")):
    # skip whole video group only if every segment already has an embedding
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
        vr = VideoReader(video_path, ctx=cpu(0))
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
            indices = np.linspace(start_idx, end_idx, FRAMES_PER_CLIP, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            clip = torch.from_numpy(frames).permute(0, 3, 1, 2)
            inputs = processor(clip, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, skip_predictor=True)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            torch.save(emb, out_path)
            saved += 1
        except Exception as e:
            print(f"  ERROR {row['segment_id']}: {e}", flush=True)
            skipped += 1

    if (vi + 1) % 100 == 0:
        elapsed = time.time() - start_time
        rate = elapsed / (saved + 1e-9)
        remaining = len(df) - saved - existing - skipped
        eta_min = rate * remaining / 60
        print(f"  [{vi+1}/{by_video.ngroups} videos] saved={saved:,} existing={existing:,} "
              f"skipped={skipped:,} elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min", flush=True)

total_files = len([f for f in os.listdir(EMB_DIR) if f.endswith('.pt')])
print(f"\nDone. Saved this run: {saved:,}  Already existing: {existing:,}  "
      f"Skipped/errors: {skipped:,}  Total files in dir: {total_files:,}", flush=True)

# Spot check
sample = torch.load(os.path.join(EMB_DIR, os.listdir(EMB_DIR)[0]))
print(f"Spot check — shape: {sample.shape}  mean: {sample.mean():.4f}  std: {sample.std():.4f}", flush=True)
assert sample.shape == (1024,), f"Expected (1024,), got {sample.shape}"
print("DONE", flush=True)
