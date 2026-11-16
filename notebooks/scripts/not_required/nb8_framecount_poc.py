"""
POC: does feeding the SSv2 16-frame checkpoint's encoder+pooler MORE frames
than it was trained on (32, 64) help, despite the distribution-shift risk?
Frozen encoder + frozen pretrained pooler in all three cases -- only the
frame count fed in changes. Same small sample, same matched 1.3M-param linear
probe recipe as prior POCs, so results are directly comparable to the
already-completed 16-frame frozen-pool result (43.62% Top-1 at full scale).
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

MODEL_NAME = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
FRAME_COUNTS = [16, 32, 64]
EMB_DIRS = {n: os.path.join(PATHS['embeddings'], f'vjepa2_poc_frames{n}') for n in FRAME_COUNTS}
for d in EMB_DIRS.values():
    os.makedirs(d, exist_ok=True)

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
print(f"Loaded {MODEL_NAME}", flush=True)

t0 = time.time()
by_video = df_sample.groupby('video_id')
saved = 0
for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="Multi-framecount extraction"):
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        continue
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        continue
    fps = vr.get_avg_fps()
    for _, row in grp.iterrows():
        start_idx = max(0, int(row['start_sec'] * fps))
        end_idx = min(len(vr) - 1, int(row['end_sec'] * fps))
        all_exist = all(os.path.exists(os.path.join(EMB_DIRS[n], f"{row['segment_id']}.pt")) for n in FRAME_COUNTS)
        if all_exist:
            saved += 1
            continue
        try:
            for n_frames in FRAME_COUNTS:
                out_path = os.path.join(EMB_DIRS[n_frames], f"{row['segment_id']}.pt")
                if os.path.exists(out_path):
                    continue
                indices = np.linspace(start_idx, end_idx, n_frames, dtype=int)
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

print(f"\nExtraction done: {saved}/{len(df_sample)} in {(time.time()-t0)/60:.1f} min", flush=True)
del full_model, encoder, pooler
torch.cuda.empty_cache()

# ── Train the same matched linear probe on each frame count ────────────────
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


results = {}
for n_frames in FRAME_COUNTS:
    print(f"\n=== Training on {n_frames}-frame pretrained-pool ===", flush=True)
    hist = run_linear_probe(EMB_DIRS[n_frames], f'{n_frames}f')
    best = hist.loc[hist['val_top1'].idxmax()]
    results[n_frames] = best
    print(f"Best {n_frames}-frame: epoch {int(best['epoch'])} Top-1={best['val_top1']:.4f} Top-5={best['val_top5']:.4f}", flush=True)

print(f"\n=== FRAME-COUNT POC RESULT ===", flush=True)
for n_frames in FRAME_COUNTS:
    r = results[n_frames]
    print(f"{n_frames} frames: Top-1={r['val_top1']*100:.2f}%  Top-5={r['val_top5']*100:.2f}%", flush=True)
print("(Small sample -- directional. Positive trend with more frames = worth full re-extraction.)", flush=True)
print("DONE", flush=True)
