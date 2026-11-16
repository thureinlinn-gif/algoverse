"""
Is the forward-vs-reversed probe accuracy inflated by how the train/test split
was drawn?

Step 7's probe originally split by segment_id. Both versions of a segment always
went to the same side, so it could never cheat by matching a clip to its own
reversed pair -- but two clips from the SAME VIDEO could land on opposite sides,
so in principle it could learn a per-video (per-kitchen, per-camera) shortcut.

This script settles that empirically instead of arguing about it. It reuses the
forward/reversed embeddings step 8 already cached, so it needs no GPU extraction
and no video decoding -- seconds to run. Two things change versus step 7:

  * full canonical evaluation set (all 3,268 validation segments) rather than
    step 7's 600-segment pilot, and
  * 5 random splits per split level, so the comparison is against seed noise
    rather than a single draw.

Both split levels are run side by side. If video-level accuracy comes in well
below segment-level, the original number was inflated and step 7's headline
needs restating. If they agree, the pilot number stands as reported.

Reads : embeddings/<...>/vjepa2_pretrainedpool_order_{fwd,rev}/*.pt (step 8)
        annotations/segments_verbs_timed_clean.csv
Writes: processed/nb8_temporal_order_probe_split_level.csv
"""
import os, json, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
OUT_DIR = PATHS['processed']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FWD_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_order_fwd')
REV_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_order_rev')
N_SPLIT_SEEDS = 5
TRAIN_FRAC = 0.8
PROBE_EPOCHS = 200          # same probe hyperparameters as step 7
PROBE_LR = 1e-3
PROBE_WD = 1e-2

ann = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv')).set_index('segment_id')

# Canonical evaluation set only. The cache holds a few hundred extra segments
# from earlier exploratory runs, some of them TRAINING segments -- see the note
# in step 9 for why globbing the cache directory is not the evaluation set.
val_ids = set(ann.index[ann['subset'] == 'validation'])
fwd = {f[:-3] for f in os.listdir(FWD_DIR) if f.endswith('.pt')}
rev = {f[:-3] for f in os.listdir(REV_DIR) if f.endswith('.pt')}
segments = sorted((fwd & rev) & val_ids)
video_of = ann.loc[segments, 'video_id'].to_dict()
print(f"Segments: {len(segments):,} over {len(set(video_of.values())):,} videos "
      f"({len((fwd & rev) - val_ids)} cached non-validation segments excluded)")

emb = {}
for s in segments:
    emb[s] = (F.normalize(torch.load(os.path.join(FWD_DIR, f'{s}.pt')), dim=-1).numpy(),
              F.normalize(torch.load(os.path.join(REV_DIR, f'{s}.pt')), dim=-1).numpy())


def train_ids_for(split_level, seed):
    """Segment ids on the train side, holding out TRAIN_FRAC of whichever unit
    `split_level` names."""
    rng = random.Random(seed)
    if split_level == 'video':
        videos = sorted(set(video_of.values()))
        rng.shuffle(videos)
        held = set(videos[:int(TRAIN_FRAC * len(videos))])
        return {s for s in segments if video_of[s] in held}
    order = list(segments)
    rng.shuffle(order)
    return set(order[:int(TRAIN_FRAC * len(order))])


def build_xy(id_set):
    X, y = [], []
    for s in id_set:
        X.append(emb[s][0]); y.append(0.0)   # forward
        X.append(emb[s][1]); y.append(1.0)   # reversed
    return np.stack(X), np.array(y, dtype=np.float32)


def run_probe(split_level, seed):
    tr = train_ids_for(split_level, seed)
    te = set(segments) - tr
    X_tr, y_tr = build_xy(tr)
    X_te, y_te = build_xy(te)
    X_tr = torch.tensor(X_tr).to(DEVICE); y_tr = torch.tensor(y_tr).to(DEVICE)
    X_te = torch.tensor(X_te).to(DEVICE); y_te = torch.tensor(y_te).to(DEVICE)

    torch.manual_seed(seed); np.random.seed(seed)
    probe = nn.Linear(X_tr.shape[1], 1).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    for _ in range(PROBE_EPOCHS):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(probe(X_tr).squeeze(-1), y_tr).backward()
        opt.step()

    with torch.no_grad():
        acc_tr = (((torch.sigmoid(probe(X_tr).squeeze(-1)) > 0.5).float()) == y_tr).float().mean().item()
        acc_te = (((torch.sigmoid(probe(X_te).squeeze(-1)) > 0.5).float()) == y_te).float().mean().item()
    return {'split_level': split_level, 'split_seed': seed,
            'n_train_segments': len(tr), 'n_test_segments': len(te),
            'train_acc': acc_tr, 'test_acc': acc_te}


rows = [run_probe(level, seed)
        for level in ('segment', 'video')
        for seed in range(N_SPLIT_SEEDS)]
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_probe_split_level.csv'), index=False)

print("\n=== Forward-vs-reversed probe, segment-level vs video-level split ===")
print(f"(N={len(segments):,} validation segments, {N_SPLIT_SEEDS} random splits each, chance = 50.00%)\n")
for level, g in res.groupby('split_level'):
    print(f"  split by {level:<8} test acc {g['test_acc'].mean()*100:6.2f}% "
          f"+/- {g['test_acc'].std()*100:.2f}   per-split: "
          f"{[round(a*100, 2) for a in g['test_acc']]}")

gap = (res.query("split_level == 'segment'")['test_acc'].mean()
       - res.query("split_level == 'video'")['test_acc'].mean()) * 100
seed_sd = res.query("split_level == 'video'")['test_acc'].std() * 100
print(f"\nsegment-level minus video-level: {gap:+.2f}pp "
      f"(seed-to-seed sd at video level: {seed_sd:.2f}pp)")
print("Inflation from the segment-level split is within seed noise."
      if abs(gap) <= 2 * seed_sd else
      "*** Segment-level split materially inflates the probe -- report the video-level number. ***")
print(f"\nSaved -> {os.path.join(OUT_DIR, 'nb8_temporal_order_probe_split_level.csv')}")
print("DONE")
