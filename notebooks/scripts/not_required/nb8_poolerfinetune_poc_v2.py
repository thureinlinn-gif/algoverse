"""
v2 of nb8_poolerfinetune_poc.py: same core test (does letting V-JEPA2's
SSv2-pretrained attentive pooler FINE-TUNE, instead of staying frozen, unlock
more accuracy on cooking verbs?), but fixes two gaps in the original POC:

1. Original only printed results -- nothing was saved to disk, so the result
   was lost when the GPU disconnected. This version saves history + ranks to
   CSV like every other nb8_* script.
2. Original only reported an aggregate Top-1 number. That can't answer the
   actual research question (does V-JEPA2 have an edge on motion verbs
   specifically), so this version adds a per-segment ranking pass on the POC
   val set and breaks it down by optical-flow tercile (using the
   already-computed, encoder-independent nb8_optical_flow_scores.csv) and
   compares directly against the frozen-pretrained-pool baseline (from the
   full-scale run, subset to these same val segment_ids) on the same buckets.

Reuses the already-extracted per-timestep tokens (embeddings/vjepa2_poc_ssv2_pertimestep,
1200 train + 300 val, SSv2 fpc16 checkpoint) and the saved pretrained-pooler
init weights -- no re-extraction, so this should run in a couple of minutes.
"""
import os, sys, json, math, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import VJEPA2ForVideoClassification
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
N_TRAIN_SAMPLE = 1200
N_VAL_SAMPLE = 300
SEED = 0
TAG = 'vjepa2_poolerfinetune_poc'

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
n_verbs = len(verb_vocab)
df['verb_label'] = df['extracted_verb'].map(label2id)

df_train_full = df[df['subset'] == 'training'].reset_index(drop=True)
df_val_full = df[df['subset'] == 'validation'].reset_index(drop=True)
df_train = df_train_full.sample(n=min(N_TRAIN_SAMPLE, len(df_train_full)), random_state=SEED).reset_index(drop=True)
df_val = df_val_full.sample(n=min(N_VAL_SAMPLE, len(df_val_full)), random_state=SEED).reset_index(drop=True)
print(f"Sampled (same split as prior POCs): {len(df_train)} train + {len(df_val)} val", flush=True)

EMB_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_poc_ssv2_pertimestep')
POOLER_INIT_PATH = os.path.join(PATHS['checkpoints'], 'vjepa2_ssv2_pretrained_pooler_init.pt')
n_avail = len([f for f in os.listdir(EMB_DIR) if f.endswith('.pt')])
print(f"Per-timestep embeddings available: {n_avail}", flush=True)
assert n_avail > 0, "No per-timestep embeddings found -- run nb8_poolerfinetune_poc.py first to extract."

MODEL_NAME = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
_cfg_model = VJEPA2ForVideoClassification.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, ignore_mismatched_sizes=True,
)
vjepa_config = _cfg_model.vjepa2.config
del _cfg_model

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
        self.records = [(r['segment_id'], os.path.join(emb_dir, f"{r['segment_id']}.pt"), int(r['verb_label']))
                         for _, r in df.iterrows() if os.path.exists(os.path.join(emb_dir, f"{r['segment_id']}.pt"))]
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        seg_id, p, label = self.records[idx]
        return torch.load(p, map_location='cpu'), label, seg_id


def collate(batch):
    embs, labels, seg_ids = zip(*batch)
    return torch.stack(embs), torch.tensor(labels, dtype=torch.long), list(seg_ids)


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
            for embs, labels, _ in val_loader:
                pooled = F.normalize(pool(embs.to(DEVICE)), dim=-1)
                labels = labels.to(DEVICE)
                sim = model.logit_scale.exp() * pooled @ verb_embs.T
                c1 += (sim.argmax(-1) == labels).sum().item()
                top5 = sim.topk(5, dim=-1).indices
                c5 += (top5 == labels.unsqueeze(-1)).any(-1).sum().item()
                tot += labels.size(0)
        return c1/tot, c5/tot

    history = []
    state_dicts = {}
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); pool.train()
        for embs, labels, _ in train_loader:
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
        state_dicts[epoch] = {
            'proj': {k: v.clone().cpu() for k, v in model.text_model.head.linear_projection.state_dict().items()},
            'pool': {k: v.clone().cpu() for k, v in pool.state_dict().items()},
        }
    del model, pool
    torch.cuda.empty_cache()
    return pd.DataFrame(history), state_dicts


print(f"\n=== Training with pooler fine-tuning (init from pretrained SSv2 weights) ===", flush=True)
hist, state_dicts = run_pooler_finetune()
hist.to_csv(os.path.join(OUT_DIR, f'nb8_{TAG}_history.csv'), index=False)
best = hist.loc[hist['val_top1'].idxmax()]
best_epoch = int(best['epoch'])

FROZEN_BASELINE_TOP1 = 0.4367
print(f"\n=== POOLER-FINETUNE POC RESULT ===", flush=True)
print(f"Frozen pretrained-pool (already known): Top-1=43.67%  Top-5=62.00%", flush=True)
print(f"Fine-tuned pretrained-pool:              Top-1={best['val_top1']*100:.2f}%  Top-5={best['val_top5']*100:.2f}%  (epoch {best_epoch})", flush=True)
print(f"Delta Top-1: {(best['val_top1']-FROZEN_BASELINE_TOP1)*100:+.2f}pp", flush=True)

# ── Reload best-epoch state, do a full per-segment ranking pass on df_val ──
model = load_fresh_dinov2_dinotxt()
model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
model.text_model.head.linear_projection.load_state_dict(
    {k: v.to(DEVICE) for k, v in state_dicts[best_epoch]['proj'].items()})
model.eval()
pool = VJEPA2AttentivePooler(vjepa_config).to(DEVICE)
pool.load_state_dict({k: v.to(DEVICE) for k, v in state_dicts[best_epoch]['pool'].items()})
pool.eval()

verb_tokens = tokenizer_v2.tokenize(verb_vocab).to(DEVICE)
with torch.no_grad():
    verb_embs = model.encode_text(verb_tokens, normalize=True)

records = []
val_ds = TemporalDataset(df_val, EMB_DIR)
with torch.no_grad():
    for x, label, seg_id in tqdm(val_ds, desc="Ranking (poolerfinetune)"):
        pooled = F.normalize(pool(x.unsqueeze(0).to(DEVICE)), dim=-1)
        sim = (model.logit_scale.exp() * pooled @ verb_embs.T).squeeze(0).cpu()
        ranked = sim.argsort(descending=True).tolist()
        rank = ranked.index(label) + 1
        records.append({'segment_id': seg_id, 'rank': rank, 'reciprocal_rank': 1.0/rank,
                         'top1_correct': rank == 1, 'top5_correct': rank <= 5})
ranks_ft = pd.DataFrame(records)
# fill ground_truth from df_val (cleaner than threading id2label through the loop)
gt_map = df_val.set_index('segment_id')['extracted_verb'].to_dict()
ranks_ft['ground_truth'] = ranks_ft['segment_id'].map(gt_map)
ranks_ft = ranks_ft.set_index('segment_id')
ranks_ft.to_csv(os.path.join(OUT_DIR, f'nb8_ranks_{TAG}.csv'))
print(f"\n{TAG} POC val — N={len(ranks_ft)} Top-1={ranks_ft['top1_correct'].mean()*100:.2f}% "
      f"Top-5={ranks_ft['top5_correct'].mean()*100:.2f}%", flush=True)

# ── Optical-flow tercile breakdown, compared against the frozen-pretrainedpool
#    full-scale representative model subset to these SAME 300 val segments ──
flow_df = pd.read_csv(os.path.join(OUT_DIR, 'nb8_optical_flow_scores.csv'))
flow_df['tercile'] = pd.qcut(flow_df['motion_score'], 3, labels=['low_motion', 'mid_motion', 'high_motion'])
flow_df = flow_df.set_index('segment_id')

frozen_path = os.path.join(OUT_DIR, 'nb8_ranks_vjepa2_pretrainedpool_representative.csv')
rows = []
if os.path.exists(frozen_path):
    ranks_frozen_full = pd.read_csv(frozen_path).set_index('segment_id')
    common_ids = ranks_ft.index.intersection(ranks_frozen_full.index)
    print(f"\nSegments in both POC finetuned-pooler set and full-scale frozen-pool ranks: {len(common_ids)}/{len(ranks_ft)}", flush=True)
    ranks_frozen = ranks_frozen_full.loc[common_ids]
    ranks_ft_matched = ranks_ft.loc[common_ids]
    for tag, ranks in [('vjepa2_pretrainedpool_finetuned_POC', ranks_ft_matched),
                        ('vjepa2_pretrainedpool_frozen_fullscale', ranks_frozen)]:
        merged = ranks.join(flow_df[['tercile']], how='inner')
        for tercile, grp in merged.groupby('tercile', observed=True):
            rows.append({'model': tag, 'tercile': tercile, 'n': len(grp),
                         'micro_top1': grp['top1_correct'].mean(), 'micro_top5': grp['top5_correct'].mean(),
                         'mrr': grp['reciprocal_rank'].mean()})
else:
    print(f"\nWARNING: {frozen_path} not found -- skipping frozen-baseline tercile comparison.", flush=True)
    merged = ranks_ft.join(flow_df[['tercile']], how='inner')
    for tercile, grp in merged.groupby('tercile', observed=True):
        rows.append({'model': 'vjepa2_pretrainedpool_finetuned_POC', 'tercile': tercile, 'n': len(grp),
                     'micro_top1': grp['top1_correct'].mean(), 'micro_top5': grp['top5_correct'].mean(),
                     'mrr': grp['reciprocal_rank'].mean()})

tercile_df = pd.DataFrame(rows)
print("\n=== Optical-flow tercile: finetuned pooler (POC) vs frozen pooler (full-scale), matched segments ===")
print(tercile_df.to_string(index=False), flush=True)
tercile_df.to_csv(os.path.join(OUT_DIR, f'nb8_optical_flow_tercile_{TAG}.csv'), index=False)
print(f"\nSaved -> nb8_optical_flow_tercile_{TAG}.csv", flush=True)
print("DONE", flush=True)
