"""
Precompute the order-aware DINOv2 pooling (concat of first-half-mean,
second-half-mean -- REPRODUCE.md's "cheapest honest version") for every
segment, forward orientation, so it can be trained against with the exact
same ProjDataset/DataLoader machinery script 07 uses for every other
backbone. This is the last piece needed before training an order-aware
DINOv2+DinoTXT head: does giving DINOv2 fair pooling change (a) verb
classification accuracy and (b) whether the trained alignment head still
prefers forward over reversed for the true verb (script 15's "Claim B"),
the way V-JEPA2's does?

No GPU -- this is pure pooling arithmetic over script 22's already-extracted
per-frame stacks. Reversed-orientation embeddings are NOT precomputed here;
per the mentor's flip trick, torch.flip(x, [0]) on the cached per-frame stack
gives the reversed stack exactly, computed on the fly wherever it's needed
(script 28), no extra storage.

Reads : embeddings/<...>/dinov2_perframe/<segment_id>.pt (script 22's output,
        ALL 13,003 segments -- train + val, unlike script 25 which only
        needed validation)
Writes: embeddings/<...>/dinov2_perframe_halves/<segment_id>.pt, each a
        (4096,) tensor -- concat(mean(first half), mean(second half)),
        forward orientation
"""
import os, json
import pandas as pd
import torch
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
PERFRAME_DIR = os.path.join(PATHS['embeddings'], 'dinov2_perframe')
HALVES_DIR = os.path.join(PATHS['embeddings'], 'dinov2_perframe_halves')
os.makedirs(HALVES_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
available = {f[:-3] for f in os.listdir(PERFRAME_DIR) if f.endswith('.pt')}
df = df[df['segment_id'].isin(available)].reset_index(drop=True)
print(f"Segments with per-frame features: {len(df):,} / 13,003")

written = skipped = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="Halves-pool"):
    out_path = os.path.join(HALVES_DIR, f"{row['segment_id']}.pt")
    if os.path.exists(out_path):
        skipped += 1
        continue
    stack = torch.load(os.path.join(PERFRAME_DIR, f"{row['segment_id']}.pt"), map_location='cpu')
    T = stack.shape[0]
    half = max(T // 2, 1)
    halves = torch.cat([stack[:half].mean(dim=0), stack[half:].mean(dim=0)]) if T > half \
        else torch.cat([stack.mean(dim=0), stack.mean(dim=0)])
    torch.save(halves, out_path)
    written += 1

print(f"\nWritten: {written:,} | already present: {skipped:,}")
n_final = len([f for f in os.listdir(HALVES_DIR) if f.endswith('.pt')])
print(f"Total in {HALVES_DIR}: {n_final:,}")
print("DONE")
