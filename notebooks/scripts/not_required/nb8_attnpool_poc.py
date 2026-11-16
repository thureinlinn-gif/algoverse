"""
Fast proof-of-concept: does a learned attention-pooling head beat naive
mean-pooling for V-JEPA2, on a small sample, before committing to the full
~8h re-extraction + full training run?

Samples ~1,500 segments (stratified train/val), extracts BOTH per-timestep
(32x1024) and mean-pooled (1024,) V-JEPA2 embeddings for just this subset
(~1h instead of ~8h for the full dataset), then trains both pooling
approaches fresh on the identical small sample -- same train/val split, same
recipe, same epochs -- so it's a fair, apples-to-apples comparison uncontaminated
by different dataset sizes.

This is directional only (small sample, few epochs) -- a real positive signal
here is a good reason to commit to the full run; a flat/negative result here
is weaker evidence against (small samples may just be too little data to train
the ~4M-param pooling head well), but still a useful, cheap sanity check
before spending 8 hours.
"""
import os, sys, json, math, random, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import AutoModel, VJEPA2VideoProcessor
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
VIDEO_DIR = PATHS['videos']

N_TRAIN_SAMPLE = 1200
N_VAL_SAMPLE = 300
SEED = 0

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
n_verbs = len(verb_vocab)
df['verb_label'] = df['extracted_verb'].map(label2id)

rng = np.random.RandomState(SEED)
df_train_full = df[df['subset'] == 'training'].reset_index(drop=True)
df_val_full = df[df['subset'] == 'validation'].reset_index(drop=True)
df_train = df_train_full.sample(n=min(N_TRAIN_SAMPLE, len(df_train_full)), random_state=SEED).reset_index(drop=True)
df_val = df_val_full.sample(n=min(N_VAL_SAMPLE, len(df_val_full)), random_state=SEED).reset_index(drop=True)
df_sample = pd.concat([df_train, df_val], ignore_index=True)
print(f"Sampled: {len(df_train)} train + {len(df_val)} val = {len(df_sample)} segments "
      f"({df_sample['video_id'].nunique()} unique videos)", flush=True)

# ── Step 1: extract per-timestep embeddings for just this subset ───────────
MODEL_NAME = 'facebook/vjepa2-vitl-fpc64-256'
EMB_DIR_POC = os.path.join(PATHS['embeddings'], 'vjepa2_poc_temporal')
os.makedirs(EMB_DIR_POC, exist_ok=True)

processor = VJEPA2VideoProcessor.from_pretrained(MODEL_NAME)
vjepa_model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE).eval()
for p in vjepa_model.parameters():
    p.requires_grad = False
FRAMES_PER_CLIP = vjepa_model.config.frames_per_clip
T_STEPS = FRAMES_PER_CLIP // vjepa_model.config.tubelet_size
SPATIAL = (vjepa_model.config.crop_size // vjepa_model.config.patch_size) ** 2
print(f"V-JEPA2 loaded. T_STEPS={T_STEPS} SPATIAL={SPATIAL}", flush=True)

t0 = time.time()
by_video = df_sample.groupby('video_id')
saved = 0
for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="POC extraction"):
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        continue
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        continue
    fps = vr.get_avg_fps()
    for _, row in grp.iterrows():
        out_path = os.path.join(EMB_DIR_POC, f"{row['segment_id']}.pt")
        if os.path.exists(out_path):
            saved += 1
            continue
        try:
            start_idx = max(0, int(row['start_sec'] * fps))
            end_idx = min(len(vr) - 1, int(row['end_sec'] * fps))
            indices = np.linspace(start_idx, end_idx, FRAMES_PER_CLIP, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            clip = torch.from_numpy(frames).permute(0, 3, 1, 2)
            inputs = processor(clip, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = vjepa_model(**inputs, skip_predictor=True)
            tokens = out.last_hidden_state.squeeze(0)
            per_timestep = tokens.view(T_STEPS, SPATIAL, -1).mean(dim=1).cpu()
            torch.save(per_timestep, out_path)
            saved += 1
        except Exception as e:
            print(f"  ERROR {row['segment_id']}: {e}", flush=True)

print(f"\nExtraction done: {saved}/{len(df_sample)} in {(time.time()-t0)/60:.1f} min", flush=True)
del vjepa_model
torch.cuda.empty_cache()

# ── Step 2: train both pooling approaches on this identical small sample ───
DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

tokenizer_v2 = get_tokenizer()
IN_DIM, OUT_DIM = 1280, 1024
NUM_EPOCHS = 8
LR = 3e-4


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fresh_dinov2_dinotxt():
    m = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


class AttentionPool(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class TemporalDataset(torch.utils.data.Dataset):
    """Loads per-timestep (32, 1024) embeddings."""
    def __init__(self, df, emb_dir):
        self.records = [(os.path.join(emb_dir, f"{r['segment_id']}.pt"), int(r['verb_label']))
                         for _, r in df.iterrows() if os.path.exists(os.path.join(emb_dir, f"{r['segment_id']}.pt"))]
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        p, label = self.records[idx]
        return torch.load(p, map_location='cpu'), label


class MeanPooledDataset(torch.utils.data.Dataset):
    """Derives the mean-pooled (1024,) vector on the fly from the same
    per-timestep embeddings, for a perfectly matched baseline (identical
    underlying forward pass, only the pooling differs)."""
    def __init__(self, df, emb_dir):
        self.records = [(os.path.join(emb_dir, f"{r['segment_id']}.pt"), int(r['verb_label']))
                         for _, r in df.iterrows() if os.path.exists(os.path.join(emb_dir, f"{r['segment_id']}.pt"))]
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        p, label = self.records[idx]
        x = torch.load(p, map_location='cpu')  # (32, 1024)
        return x.mean(dim=0), label  # (1024,)


def collate_temporal(batch):
    embs, labels = zip(*batch)
    return torch.stack(embs), torch.tensor(labels, dtype=torch.long)


def run_meanpool_baseline(seed=0):
    set_seed(seed)
    model = load_fresh_dinov2_dinotxt()
    model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = 'text_model.head.linear_projection' in name

    train_ds = MeanPooledDataset(df_train, EMB_DIR_POC)
    val_ds = MeanPooledDataset(df_val, EMB_DIR_POC)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_temporal)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_temporal)
    verb_tokens_bare = tokenizer_v2.tokenize(verb_vocab).to(DEVICE)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=LR, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)

    def evaluate():
        model.eval()
        with torch.no_grad():
            verb_embs = model.encode_text(verb_tokens_bare, normalize=True)
        c1 = c5 = tot = 0
        with torch.no_grad():
            for embs, labels in val_loader:
                embs = F.normalize(embs.to(DEVICE), dim=-1)
                labels = labels.to(DEVICE)
                sim = model.logit_scale.exp() * embs @ verb_embs.T
                c1 += (sim.argmax(-1) == labels).sum().item()
                top5 = sim.topk(5, dim=-1).indices
                c5 += (top5 == labels.unsqueeze(-1)).any(-1).sum().item()
                tot += labels.size(0)
        return c1/tot, c5/tot

    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for embs, labels in train_loader:
            embs = F.normalize(embs.to(DEVICE), dim=-1)
            labels = labels.to(DEVICE)
            verb_embs = model.encode_text(verb_tokens_bare, normalize=True)
            sim = model.logit_scale.exp() * embs @ verb_embs.T
            loss = F.cross_entropy(sim, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
        top1, top5 = evaluate()
        history.append({'epoch': epoch, 'val_top1': top1, 'val_top5': top5})
        print(f"  [mean-pool baseline] epoch {epoch}/{NUM_EPOCHS} | Val Top-1 {top1:.4f} | Val Top-5 {top5:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(history)


def run_attnpool(seed=0):
    set_seed(seed)
    model = load_fresh_dinov2_dinotxt()
    model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = 'text_model.head.linear_projection' in name
    pool = AttentionPool(dim=OUT_DIM).to(DEVICE)

    train_ds = TemporalDataset(df_train, EMB_DIR_POC)
    val_ds = TemporalDataset(df_val, EMB_DIR_POC)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_temporal)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_temporal)
    verb_tokens_bare = tokenizer_v2.tokenize(verb_vocab).to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(pool.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)

    def evaluate():
        model.eval(); pool.eval()
        with torch.no_grad():
            verb_embs = model.encode_text(verb_tokens_bare, normalize=True)
        c1 = c5 = tot = 0
        with torch.no_grad():
            for embs, labels in val_loader:
                pooled = F.normalize(pool(embs.to(DEVICE)), dim=-1)
                labels = labels.to(DEVICE)
                sim = model.logit_scale.exp() * pooled @ verb_embs.T
                c1 += (sim.argmax(-1) == labels).sum().item()
                top5 = sim.topk(5, dim=-1).indices
                c5 += (top5 == labels.unsqueeze(-1)).any(-1).sum().item()
                tot += labels.size(0)
        return c1/tot, c5/tot

    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); pool.train()
        for embs, labels in train_loader:
            pooled = F.normalize(pool(embs.to(DEVICE)), dim=-1)
            labels = labels.to(DEVICE)
            verb_embs = model.encode_text(verb_tokens_bare, normalize=True)
            sim = model.logit_scale.exp() * pooled @ verb_embs.T
            loss = F.cross_entropy(sim, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
        top1, top5 = evaluate()
        history.append({'epoch': epoch, 'val_top1': top1, 'val_top5': top5})
        print(f"  [attn-pool] epoch {epoch}/{NUM_EPOCHS} | Val Top-1 {top1:.4f} | Val Top-5 {top5:.4f}", flush=True)
    del model, pool
    torch.cuda.empty_cache()
    return pd.DataFrame(history)


print(f"\n=== Training mean-pool baseline (N_train={len(df_train)}, N_val={len(df_val)}) ===", flush=True)
mean_hist = run_meanpool_baseline()
best_mean = mean_hist.loc[mean_hist['val_top1'].idxmax()]
print(f"Best mean-pool: epoch {int(best_mean['epoch'])} Top-1={best_mean['val_top1']:.4f} Top-5={best_mean['val_top5']:.4f}", flush=True)

print(f"\n=== Training attention-pool (same sample) ===", flush=True)
attn_hist = run_attnpool()
best_attn = attn_hist.loc[attn_hist['val_top1'].idxmax()]
print(f"Best attn-pool: epoch {int(best_attn['epoch'])} Top-1={best_attn['val_top1']:.4f} Top-5={best_attn['val_top5']:.4f}", flush=True)

print(f"\n=== POC RESULT ===", flush=True)
print(f"Mean-pool baseline best Top-1: {best_mean['val_top1']*100:.2f}%  Top-5: {best_mean['val_top5']*100:.2f}%", flush=True)
print(f"Attn-pool         best Top-1: {best_attn['val_top1']*100:.2f}%  Top-5: {best_attn['val_top5']*100:.2f}%", flush=True)
print(f"Delta Top-1: {(best_attn['val_top1']-best_mean['val_top1'])*100:+.2f}pp  "
      f"Delta Top-5: {(best_attn['val_top5']-best_mean['val_top5'])*100:+.2f}pp", flush=True)
print("(N_val is small (~300) so treat this as directional, not definitive)", flush=True)
print("DONE", flush=True)
