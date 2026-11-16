"""
Re-extract DINOv2+DinoTXT features WITHOUT collapsing time.

Why this exists: notebook 04 runs the frozen DinoTXT vision tower per frame and
saves only `frame_embeddings.mean(dim=0)`. That mean is order-invariant by
construction, so the DINOv2/DINOv3 arm of the temporal-order comparison cannot
represent clip direction -- not because the encoder can't, but because the
pipeline threw the time axis away before anything downstream could use it. A
reviewer is entitled to call that a baseline built to lose, and they would be
right: the honest comparison is V-JEPA2's temporal pooling against DINOv2 given
an order-aware pooling of its own, over the same per-frame features.

The per-frame features were never written to disk, so they have to be recomputed.
This script does exactly what notebook 04 does -- same frame sampling (up to
MAX_FRAMES=50 evenly spaced within the segment), same official eval transform,
same frozen `encode_image(..., normalize=True)` -- and saves the full (T, 2048)
stack instead of its mean.

The existing pooled baseline is recoverable from these files exactly:

    torch.load(f'{PERFRAME_DIR}/{seg}.pt').mean(dim=0)   ==   torch.load(f'{EMB_DIR}/{seg}.pt')

which is the point: any order-aware head trained on top of these is compared
against the published mean-pool number over identical features, so the only
thing that differs between the two arms is the pooling. The script verifies that
identity on the first VERIFY_N segments it writes and aborts if it fails.

Cost: the vision-tower forward is the same work notebook 04 already did, so
expect roughly what that notebook took end to end (~2-3h on one A10 for 13,003
segments; decoding dominates). Storage: ~5.3 GB in fp32 (13,003 segments x up to
50 frames x 2048 dims). Resumable -- already-written segments are skipped.

Verified on this box 2026-09-02 over 5 random segments: T=50, dim=2048, and
max |mean(per-frame) - published pooled| = 5.7e-05 (fp32 attention-kernel noise,
not a pipeline difference).

Reads : annotations/segments_verbs_timed_clean.csv, videos/*.mp4
Writes: embeddings/<...>/dinov2_perframe/<segment_id>.pt, each a (T, 2048) tensor
"""
import os, sys, json

# The pinned xFormers wheel was built against a different torch/python than this
# venv now has, so its memory_efficient_attention kernel refuses fp32 inputs and
# the DINOv2 VISION tower dies on import-time dispatch. (The text tower never hits
# it, which is why training and the ranking passes were unaffected.) Setting this
# before dinov2 is imported takes the plain-PyTorch attention path instead --
# verified 2026-09-02 to reproduce the published pooled vectors to ~5e-5.
os.environ.setdefault('XFORMERS_DISABLED', '1')

import numpy as np
import pandas as pd
import torch
from PIL import Image
from decord import VideoReader, cpu
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.join(PATHS['repos'], 'dinov2'))
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l
from dinov2.data.transforms import make_classification_eval_transform

MAX_FRAMES = 50               # identical to notebook 04
VERIFY_N = 25                 # segments to check mean(per-frame) == published pooled vector
POOLED_DIR = PATHS['embeddings']
PERFRAME_DIR = os.path.join(PATHS['embeddings'], 'dinov2_perframe')
os.makedirs(PERFRAME_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
print(f"Segments to extract: {len(df):,}  ->  {PERFRAME_DIR}")

dinotxt = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
for p in dinotxt.parameters():
    p.requires_grad = False
SHARED_DIM = dinotxt.model_config.embed_dim
preprocess = make_classification_eval_transform()
print(f"SHARED_DIM={SHARED_DIM}  MAX_FRAMES={MAX_FRAMES}")


def extract_frames(vr, fps, start_sec, end_sec):
    """Byte-for-byte the frame selection notebook 04 uses."""
    start_idx = max(0, int(start_sec * fps))
    end_idx = min(len(vr) - 1, int(end_sec * fps))
    if start_idx >= end_idx:
        return [Image.fromarray(vr[start_idx].asnumpy())]
    n_frames = min(MAX_FRAMES, end_idx - start_idx + 1)
    indices = np.linspace(start_idx, end_idx, n_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(len(frames))]


@torch.no_grad()
def per_frame_embedding(vr, fps, start_sec, end_sec):
    frames = extract_frames(vr, fps, start_sec, end_sec)
    batch = torch.stack([preprocess(f) for f in frames]).to(DEVICE)
    return dinotxt.encode_image(batch, normalize=True).cpu()   # (T, SHARED_DIM)


written = skipped = errors = 0
verified = 0
max_pool_gap = 0.0
for video_id, grp in tqdm(df.groupby('video_id'), desc="Per-frame DINOv2"):
    todo = [r for _, r in grp.iterrows()
            if not os.path.exists(os.path.join(PERFRAME_DIR, f"{r['segment_id']}.pt"))]
    skipped += len(grp) - len(todo)
    if not todo:
        continue
    video_path = os.path.join(PATHS['videos'], f"{video_id}.mp4")
    if not os.path.exists(video_path):
        errors += len(todo)
        continue
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        errors += len(todo)
        continue
    fps = vr.get_avg_fps()
    for row in todo:
        try:
            emb = per_frame_embedding(vr, fps, row['start_sec'], row['end_sec'])
        except Exception as e:
            print(f"  ERROR {row['segment_id']}: {type(e).__name__}: {e}", flush=True)
            errors += 1
            continue
        assert emb.ndim == 2 and emb.shape[1] == SHARED_DIM, f"bad shape {tuple(emb.shape)}"

        # The whole argument rests on these being the SAME features notebook 04
        # pooled, so check that on the first VERIFY_N and stop early if not.
        if verified < VERIFY_N:
            pooled_path = os.path.join(POOLED_DIR, f"{row['segment_id']}.pt")
            if os.path.exists(pooled_path):
                gap = (emb.mean(dim=0) - torch.load(pooled_path, map_location='cpu')).abs().max().item()
                max_pool_gap = max(max_pool_gap, gap)
                assert gap < 1e-3, (   # observed ~5e-5; 1e-3 catches a real mismatch, not kernel noise
                    
                    f"{row['segment_id']}: mean of the per-frame features differs from the "
                    f"published pooled vector by {gap:.2e}. These are not the same features, "
                    f"so an order-aware head trained on them would not be comparable to the "
                    f"published mean-pool baseline. Stopping.")
                verified += 1

        torch.save(emb, os.path.join(PERFRAME_DIR, f"{row['segment_id']}.pt"))
        written += 1

print(f"\nWritten: {written:,} | already present: {skipped:,} | errors/skipped: {errors:,}")
print(f"Pooling identity verified on {verified} segments "
      f"(max |mean(per-frame) - published pooled| = {max_pool_gap:.2e})")
print(f"\nNext: train an order-aware head on {PERFRAME_DIR} and compare it against the")
print("published mean-pool DINOv2 number. Same features, same projection budget, only")
print("the pooling differs -- that is the comparison a reviewer will ask for.")
print("DONE")
