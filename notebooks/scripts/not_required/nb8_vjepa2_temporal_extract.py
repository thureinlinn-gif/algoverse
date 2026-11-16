"""
Re-extract V-JEPA2 (base checkpoint) with space-only pooling, keeping the 32
per-timestep vectors instead of collapsing everything to one mean vector.
Confirmed via source inspection: last_hidden_state is (B, 8192, 1024) from
Conv3d(...).flatten(2) -> row-major flatten of (T'=32, H'=16, W'=16), so
consecutive blocks of 256 tokens = all spatial patches at one timestep.
.view(32, 256, 1024).mean(dim=1) -> (32, 1024) is a correct space-only pool.

This is the diagnostic-only run: only V-JEPA2 gets this, to test whether
naive mean-pooling was disproportionately hurting it before committing to
the full symmetric (all-three) version.
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
EMB_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_base_temporal')
os.makedirs(EMB_DIR, exist_ok=True)
print(f"Saving per-timestep embeddings to: {EMB_DIR}", flush=True)

processor = VJEPA2VideoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
FRAMES_PER_CLIP = model.config.frames_per_clip
patch = model.config.patch_size
tubelet = model.config.tubelet_size
crop = model.config.crop_size
T_STEPS = FRAMES_PER_CLIP // tubelet
SPATIAL = (crop // patch) ** 2
print(f"Model loaded. frames_per_clip={FRAMES_PER_CLIP} T_STEPS={T_STEPS} SPATIAL={SPATIAL}", flush=True)

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
print(f"Total segments: {len(df):,}  Unique videos: {df['video_id'].nunique():,}", flush=True)

VIDEO_DIR = PATHS['videos']
saved = existing = skipped = 0
start_time = time.time()

by_video = df.groupby('video_id')
for vi, (video_id, grp) in enumerate(tqdm(by_video, total=by_video.ngroups, desc="Extracting V-JEPA2 (per-timestep)")):
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
            tokens = out.last_hidden_state.squeeze(0)  # (8192, 1024)
            per_timestep = tokens.view(T_STEPS, SPATIAL, -1).mean(dim=1).cpu()  # (32, 1024)
            torch.save(per_timestep, out_path)
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

sample = torch.load(os.path.join(EMB_DIR, os.listdir(EMB_DIR)[0]))
print(f"Spot check — shape: {sample.shape}  mean: {sample.mean():.4f}  std: {sample.std():.4f}", flush=True)
assert sample.shape == (T_STEPS, 1024), f"Expected ({T_STEPS}, 1024), got {sample.shape}"

# Sanity check: mean over the 32 timesteps should reproduce the already-cached
# fully-pooled embedding for the same segment (same underlying forward pass).
existing_pooled_dir = os.path.join(PATHS['embeddings'], 'vjepa2_base_dinotxt')
seg_id = os.listdir(EMB_DIR)[0].replace('.pt', '')
old = torch.load(os.path.join(existing_pooled_dir, f"{seg_id}.pt"))
recon = sample.mean(dim=0)
diff = (old - recon).abs().max().item()
print(f"Consistency check vs already-cached fully-pooled embedding — max abs diff: {diff:.6f} "
      f"(should be ~0)", flush=True)
print("DONE", flush=True)
