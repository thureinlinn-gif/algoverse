"""
Forward-vs-reversed clip discrimination: a structural test of whether
V-JEPA2's encoder actually uses temporal order, as opposed to a
classification-accuracy proxy for it.

DINOv2/DINOv3 embeddings in this project are a MEAN over per-frame vectors
sampled across the clip. A mean is provably invariant to the order its terms
are summed in -- reversing frame order cannot change it, by construction, for
any preprocessing details. No experiment can find a counterexample; this is
included only as a documented mathematical fact, not measured.

V-JEPA2's encoder uses temporal positional embeddings and joint spatiotemporal
attention, so reversing frame order changes what content lands at which
position -- its embedding CAN differ. Whether it DOES, in a way a probe can
detect, is the actual empirical question this script answers.

Method: sample N val segments. For each, extract V-JEPA2's frozen pretrained
SSv2 pooler embedding twice -- once with frames in normal chronological
order, once with the same frames reversed. Two signals:
  1. Cosine similarity between forward/reversed embeddings, per segment
     (no training needed -- if this is ~1.0, order isn't encoded; the more
     it drops below 1.0, the more order-sensitive the representation is).
  2. A linear probe trained to predict "is this embedding from the reversed
     clip?" from the embedding alone, split by VIDEO id (every segment of a
     video, and both versions of every one of those segments, go to the same
     side of the split -- so the probe can neither match a segment to its own
     pair nor memorise a per-kitchen cue from sibling clips). Chance = 50%.

     Changed 2026-09-02: this split used to be by segment_id, which kept the
     pair-matching cheat blocked but let sibling clips from one video sit on
     both sides. See 07b for the full-validation-scale, 5-seed measurement of
     how much that mattered -- the answer is ~0.15pp, i.e. nothing, but the
     split is by video here regardless so the protocol needs no caveat.
"""
import os, json, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
VIDEO_DIR = PATHS['videos']
N_SAMPLE = 600
SEED = 0

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
df_sample = df_val.sample(n=min(N_SAMPLE, len(df_val)), random_state=SEED).reset_index(drop=True)
print(f"Sampled {len(df_sample)} val segments for order-discrimination test", flush=True)

MODEL_NAME = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
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
print(f"FRAMES_PER_CLIP={FRAMES_PER_CLIP}", flush=True)


@torch.no_grad()
def embed_clip(vr, fps, start_sec, end_sec, reversed_order):
    start_idx = max(0, int(start_sec * fps))
    end_idx = min(len(vr) - 1, int(end_sec * fps))
    if start_idx >= end_idx:
        return None
    indices = np.linspace(start_idx, end_idx, FRAMES_PER_CLIP, dtype=int)
    if reversed_order:
        indices = indices[::-1].copy()
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) -- already in the order given
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2)
    inputs = processor(clip, return_tensors='pt').to(DEVICE)
    enc_out = encoder(**inputs, skip_predictor=True)
    pooled = pooler(enc_out.last_hidden_state)
    return F.normalize(pooled.squeeze(0), dim=-1).cpu()


records = []
by_video = df_sample.groupby('video_id')
errors = 0
for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="Forward+reversed extraction"):
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
        try:
            fwd = embed_clip(vr, fps, row['start_sec'], row['end_sec'], reversed_order=False)
            rev = embed_clip(vr, fps, row['start_sec'], row['end_sec'], reversed_order=True)
            if fwd is None or rev is None:
                errors += 1
                continue
            cos_sim = F.cosine_similarity(fwd.unsqueeze(0), rev.unsqueeze(0)).item()
            records.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                             'cos_sim_fwd_rev': cos_sim,
                             'fwd_emb': fwd.numpy(), 'rev_emb': rev.numpy()})
        except Exception as e:
            print(f"  ERROR {row['segment_id']}: {e}", flush=True)
            errors += 1

print(f"\nExtracted {len(records)} segments ({errors} errors/skipped)", flush=True)
del full_model, encoder, pooler
torch.cuda.empty_cache()

# ── Signal 1: cosine similarity between forward and reversed embeddings ────
cos_sims = np.array([r['cos_sim_fwd_rev'] for r in records])
print(f"\n=== Cosine similarity, forward vs reversed embedding ===", flush=True)
print(f"mean={cos_sims.mean():.4f}  std={cos_sims.std():.4f}  "
      f"min={cos_sims.min():.4f}  max={cos_sims.max():.4f}", flush=True)
print(f"(1.0 = order has zero effect, i.e. behaves like DINOv2/DINOv3's mean-pool; "
      f"lower = order-sensitive)", flush=True)

cos_df = pd.DataFrame([{'segment_id': r['segment_id'], 'ground_truth': r['ground_truth'],
                         'cos_sim_fwd_rev': r['cos_sim_fwd_rev']} for r in records])
cos_df.to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_cosine.csv'), index=False)

# ── Signal 2: linear probe -- can we tell forward from reversed from the embedding alone? ──
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
seg_to_video = df_sample.set_index('segment_id')['video_id'].to_dict()
video_ids = sorted({seg_to_video[r['segment_id']] for r in records})
random.shuffle(video_ids)
vsplit = int(0.8 * len(video_ids))
train_videos = set(video_ids[:vsplit])
train_ids = {r['segment_id'] for r in records if seg_to_video[r['segment_id']] in train_videos}
test_ids = {r['segment_id'] for r in records} - train_ids
print(f"Video-level split: {len(train_videos)} train videos / "
      f"{len(video_ids) - len(train_videos)} test videos", flush=True)

def build_xy(id_set):
    X, y = [], []
    for r in records:
        if r['segment_id'] not in id_set:
            continue
        X.append(r['fwd_emb']); y.append(0)
        X.append(r['rev_emb']); y.append(1)
    return np.stack(X), np.array(y)

X_train, y_train = build_xy(train_ids)
X_test, y_test = build_xy(test_ids)
print(f"\nProbe train: {len(X_train)} examples ({len(train_ids)} segments)  "
      f"test: {len(X_test)} examples ({len(test_ids)} segments)", flush=True)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

probe = nn.Linear(X_train.shape[1], 1).to(DEVICE)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
for epoch in range(200):
    probe.train()
    opt.zero_grad()
    logits = probe(X_train_t).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, y_train_t)
    loss.backward(); opt.step()

probe.eval()
with torch.no_grad():
    test_pred = (torch.sigmoid(probe(X_test_t).squeeze(-1)) > 0.5).float()
    test_acc = (test_pred == y_test_t).float().mean().item()
    train_pred = (torch.sigmoid(probe(X_train_t).squeeze(-1)) > 0.5).float()
    train_acc = (train_pred == y_train_t).float().mean().item()

print(f"\n=== Linear probe: forward vs. reversed clip ===", flush=True)
print(f"Train accuracy: {train_acc*100:.2f}%", flush=True)
print(f"Test accuracy:  {test_acc*100:.2f}%  (chance = 50.00%)", flush=True)
print(f"\nFor comparison: DINOv2/DINOv3 test accuracy on this task is EXACTLY 50.00% "
      f"by construction (mean-pooling is order-invariant) -- not measured, proven.", flush=True)

pd.DataFrame([{
    'model': 'vjepa2_pretrainedpool', 'n_segments': len(records),
    'mean_cos_sim_fwd_rev': float(cos_sims.mean()),
    'probe_train_acc': train_acc, 'probe_test_acc': test_acc,
    'probe_n_train': len(X_train), 'probe_n_test': len(X_test),
    'split_level': 'video',
}]).to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_probe_result.csv'), index=False)

print("\nDONE", flush=True)
