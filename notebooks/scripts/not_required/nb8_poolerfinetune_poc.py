"""
POC: does letting V-JEPA2's pretrained pooler FINE-TUNE (starting from its
SSv2-pretrained weights, not random init) on our cooking-verb data beat using
it FROZEN? Tests confound #2: the pooler currently never adapts to our
domain. Reuses the already-extracted per-timestep raw tokens approach; the
frozen-pooler baseline for this exact sample was already computed in
nb8_pretrained_pool_poc.py (43.67% Top-1) -- no need to recompute it.
"""
import os, sys, json, math, random, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler
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
print(f"Sampled (same split as prior POCs): {len(df_train)} train + {len(df_val)} val", flush=True)

# ── Extract per-timestep (space-pooled) raw tokens: 8 x 1024 for 16-frame clip ──
MODEL_NAME = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
EMB_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_poc_ssv2_pertimestep')
os.makedirs(EMB_DIR, exist_ok=True)
POOLER_INIT_PATH = os.path.join(PATHS['checkpoints'], 'vjepa2_ssv2_pretrained_pooler_init.pt')

processor = VJEPA2VideoProcessor.from_pretrained(MODEL_NAME)
full_model = VJEPA2ForVideoClassification.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, ignore_mismatched_sizes=True,
).to(DEVICE)
encoder = full_model.vjepa2
for p in encoder.parameters():
    p.requires_grad = False
encoder.eval()
FRAMES_PER_CLIP = encoder.config.frames_per_clip
T_STEPS = FRAMES_PER_CLIP // encoder.config.tubelet_size
SPATIAL = (encoder.config.crop_size // encoder.config.patch_size) ** 2
print(f"T_STEPS={T_STEPS} SPATIAL={SPATIAL}", flush=True)

# Save the pretrained pooler's initial weights so training can start from them
torch.save(full_model.pooler.state_dict(), POOLER_INIT_PATH)
print(f"Saved pretrained pooler init -> {POOLER_INIT_PATH}", flush=True)
vjepa_config = encoder.config

t0 = time.time()
by_video = df_sample.groupby('video_id')
saved = 0
for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="Per-timestep extraction"):
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        continue
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        continue
    fps = vr.get_avg_fps()
    for _, row in grp.iterrows():
        out_path = os.path.join(EMB_DIR, f"{row['segment_id']}.pt")
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
                out = encoder(**inputs, skip_predictor=True)
            tokens = out.last_hidden_state.squeeze(0)  # (2048, 1024)
            per_timestep = tokens.view(T_STEPS, SPATIAL, -1).mean(dim=1).cpu()  # (8, 1024)
            torch.save(per_timestep, out_path)
            saved += 1
        except Exception as e:
            print(f"  ERROR {row['segment_id']}: {e}", flush=True)

print(f"\nExtraction done: {saved}/{len(df_sample)} in {(time.time()-t0)/60:.1f} min", flush=True)
del full_model, encoder
torch.cuda.empty_cache()

# ── Train pooler (init from pretrained weights) + linear layer jointly ─────
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


class TemporalDataset(torch.utils.data.Dataset):
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


def run_pooler_finetune(seed=0):
    set_seed(seed)
    model = load_fresh_dinov2_dinotxt()
    model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = 'text_model.head.linear_projection' in name

    pool = VJEPA2AttentivePooler(vjepa_config).to(DEVICE)
    pool.load_state_dict(torch.load(POOLER_INIT_PATH, map_location=DEVICE))
    for p in pool.parameters():
        p.requires_grad = True

    train_ds = TemporalDataset(df_train, EMB_DIR)
    val_ds = TemporalDataset(df_val, EMB_DIR)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate)
    verb_tokens_bare = tokenizer_v2.tokenize(verb_vocab).to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(pool.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable params (proj + pooler): {n_trainable:,}", flush=True)
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
        print(f"  [pooler-finetune] epoch {epoch}/{NUM_EPOCHS} | Val Top-1 {top1:.4f} | Val Top-5 {top5:.4f}", flush=True)
    del model, pool
    torch.cuda.empty_cache()
    return pd.DataFrame(history)


print(f"\n=== Training with pooler fine-tuning (init from pretrained SSv2 weights) ===", flush=True)
hist = run_pooler_finetune()
best = hist.loc[hist['val_top1'].idxmax()]

print(f"\n=== POOLER-FINETUNE POC RESULT ===", flush=True)
print(f"Frozen pretrained-pool (already known): Top-1=43.67%  Top-5=62.00%", flush=True)
print(f"Fine-tuned pretrained-pool:              Top-1={best['val_top1']*100:.2f}%  Top-5={best['val_top5']*100:.2f}%  (epoch {int(best['epoch'])})", flush=True)
print(f"Delta Top-1: {(best['val_top1']-0.4367)*100:+.2f}pp", flush=True)
print("(Small sample -- directional. Positive delta = worth full re-extraction + fine-tuning.)", flush=True)
print("DONE", flush=True)
