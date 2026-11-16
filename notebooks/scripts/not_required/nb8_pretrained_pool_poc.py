"""
Controlled POC: does V-JEPA2's own PRETRAINED attentive pooler (frozen, not
fine-tuned) beat naive mean-pooling, when both are followed by the exact same
1.3M-param trainable text-side linear projection (matching the "matched
projection" recipe used everywhere else in this project)?

Uses the SSv2 checkpoint (facebook/vjepa2-vitl-fpc16-256-ssv2) for BOTH
variants -- same encoder, same frames, only the pooling strategy differs --
so this isolates "does pretrained-but-frozen pooler init help" from the three
confounds in NB5 Part 1 (different task/classifier, ~40x more trainable
params, pooler that was also being fine-tuned). Both embedding types are
computed from the SAME forward pass per segment, on the SAME small sample and
train/val split, so it's a clean, matched, apples-to-apples comparison.
"""
import os, sys, json, math, random, time
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
N_TRAIN_SAMPLE = 1200
N_VAL_SAMPLE = 300
SEED = 0

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
n_verbs = len(verb_vocab)
df['verb_label'] = df['extracted_verb'].map(label2id)

df_train_full = df[df['subset'] == 'training'].reset_index(drop=True)
df_val_full = df[df['subset'] == 'validation'].reset_index(drop=True)
df_train = df_train_full.sample(n=min(N_TRAIN_SAMPLE, len(df_train_full)), random_state=SEED).reset_index(drop=True)
df_val = df_val_full.sample(n=min(N_VAL_SAMPLE, len(df_val_full)), random_state=SEED).reset_index(drop=True)
df_sample = pd.concat([df_train, df_val], ignore_index=True)
print(f"Sampled: {len(df_train)} train + {len(df_val)} val = {len(df_sample)} segments "
      f"({df_sample['video_id'].nunique()} unique videos)", flush=True)

# ── Step 1: extract BOTH pooling variants from the SAME forward pass ───────
MODEL_NAME = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
EMB_DIR_MEAN = os.path.join(PATHS['embeddings'], 'vjepa2_poc_ssv2_mean')
EMB_DIR_POOL = os.path.join(PATHS['embeddings'], 'vjepa2_poc_ssv2_pretrainedpool')
os.makedirs(EMB_DIR_MEAN, exist_ok=True)
os.makedirs(EMB_DIR_POOL, exist_ok=True)

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
print(f"Loaded {MODEL_NAME}. frames_per_clip={FRAMES_PER_CLIP}", flush=True)

t0 = time.time()
by_video = df_sample.groupby('video_id')
saved = 0
for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="POC extraction (both poolings)"):
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        continue
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        continue
    fps = vr.get_avg_fps()
    for _, row in grp.iterrows():
        mean_path = os.path.join(EMB_DIR_MEAN, f"{row['segment_id']}.pt")
        pool_path = os.path.join(EMB_DIR_POOL, f"{row['segment_id']}.pt")
        if os.path.exists(mean_path) and os.path.exists(pool_path):
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
                out = encoder(**inputs, skip_predictor=True)
                tokens = out.last_hidden_state  # (1, N, 1024)
                mean_emb = tokens.mean(dim=1).squeeze(0).cpu()
                pool_emb = pooler(tokens).squeeze(0).cpu()
            torch.save(mean_emb, mean_path)
            torch.save(pool_emb, pool_path)
            saved += 1
        except Exception as e:
            print(f"  ERROR {row['segment_id']}: {e}", flush=True)

print(f"\nExtraction done: {saved}/{len(df_sample)} in {(time.time()-t0)/60:.1f} min", flush=True)
del full_model, encoder, pooler
torch.cuda.empty_cache()

# ── Step 2: train the SAME 1.3M-param linear probe on each pooling variant ──
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


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, df, emb_dir):
        self.records = [(os.path.join(emb_dir, f"{r['segment_id']}.pt"), int(r['verb_label']))
                         for _, r in df.iterrows() if os.path.exists(os.path.join(emb_dir, f"{r['segment_id']}.pt"))]
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        p, label = self.records[idx]
        return torch.load(p, map_location='cpu'), label


def collate(batch):
    embs, labels = zip(*batch)
    return torch.stack(embs), torch.tensor(labels, dtype=torch.long)


def run_linear_probe(emb_dir, label, seed=0):
    set_seed(seed)
    model = load_fresh_dinov2_dinotxt()
    model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = 'text_model.head.linear_projection' in name

    train_ds = SimpleDataset(df_train, emb_dir)
    val_ds = SimpleDataset(df_val, emb_dir)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate)
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
        print(f"  [{label}] epoch {epoch}/{NUM_EPOCHS} | Val Top-1 {top1:.4f} | Val Top-5 {top5:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(history)


print(f"\n=== Training on naive mean-pool (N_train={len(df_train)}, N_val={len(df_val)}) ===", flush=True)
mean_hist = run_linear_probe(EMB_DIR_MEAN, 'mean-pool')
best_mean = mean_hist.loc[mean_hist['val_top1'].idxmax()]

print(f"\n=== Training on frozen PRETRAINED pooler ===", flush=True)
pool_hist = run_linear_probe(EMB_DIR_POOL, 'pretrained-pool')
best_pool = pool_hist.loc[pool_hist['val_top1'].idxmax()]

print(f"\n=== POC RESULT (SSv2 checkpoint, both from same forward pass) ===", flush=True)
print(f"Naive mean-pool      best Top-1: {best_mean['val_top1']*100:.2f}%  Top-5: {best_mean['val_top5']*100:.2f}%  (epoch {int(best_mean['epoch'])})", flush=True)
print(f"Frozen pretrained-pool best Top-1: {best_pool['val_top1']*100:.2f}%  Top-5: {best_pool['val_top5']*100:.2f}%  (epoch {int(best_pool['epoch'])})", flush=True)
print(f"Delta Top-1: {(best_pool['val_top1']-best_mean['val_top1'])*100:+.2f}pp  "
      f"Delta Top-5: {(best_pool['val_top5']-best_mean['val_top5'])*100:+.2f}pp", flush=True)
print("(Small sample -- directional, not definitive. Positive delta = worth the full re-extraction.)", flush=True)
print("DONE", flush=True)
