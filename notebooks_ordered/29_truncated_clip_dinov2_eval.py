"""
Truncated-clip classification, DINOv2 side: does DINOv2's ALREADY-TRAINED
mean-pool head degrade gracefully when only the first K% of a clip is
visible, or does it fall apart faster than V-JEPA2's does (script 31)?

Motivation (2026-09-08, mentor-aligned): V-JEPA2 is pretrained as a joint-
embedding PREDICTIVE architecture -- it learns to predict masked spatiotemporal
regions (often future frames) in representation space from visible context.
DINOv2 has no such objective. If that pretraining difference matters for
real-world recognition, it should show up specifically as an advantage under
PARTIAL context, not full-clip context -- something nothing in this project
has tested yet (every prior test used the complete clip).

Per the mentor's fairness constraint: V-JEPA2's side (script 30/31) uses ONLY
its frozen vision encoder + pretrained attentive pooler (`skip_predictor=True`,
no classification head weights touched) -- matching script 12's existing,
already-validated methodology exactly. This script is DINOv2's side of the
same comparison, using the SAME representative checkpoint already trained on
FULL clips (nb8_dinov2_seed0_bestepoch9.pt) -- no retraining for truncation,
so this measures the pretrained representation's inherent behavior under
truncation, not a head that learned to expect it.

Truncation is applied to script 22's cached per-frame stacks (validation set
only): keep the first ceil(frac * T) frames (already time-ordered), then
mean-pool -- approximates "seeing only the first frac of the clip's duration"
since script 22 samples frames evenly across the full span. No GPU needed for
the truncation itself; only the frozen text tower + ranking matmul use GPU,
both cheap.

Reads : embeddings/<...>/dinov2_perframe/<segment_id>.pt (script 22)
        checkpoints/nb8_dinov2_seed0_bestepoch9.pt (published representative)
Writes: processed/nb8_truncated_clip_dinov2.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
CKPT_DIR = PATHS['checkpoints']
PERFRAME_DIR = os.path.join(PATHS['embeddings'], 'dinov2_perframe')
IN_DIM, OUT_DIM = 1280, 2048
CKPT_PATH = os.path.join(CKPT_DIR, 'nb8_dinov2_seed0_bestepoch9.pt')

DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
n_verbs = len(verb_vocab)
assert n_verbs == 282
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)

available = {f[:-3] for f in os.listdir(PERFRAME_DIR) if f.endswith('.pt')}
df_val = df_val[df_val['segment_id'].isin(available)].reset_index(drop=True)
print(f"Validation segments with per-frame features: {len(df_val):,}")

model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
state = torch.load(CKPT_PATH, map_location='cpu')
model.text_model.head.linear_projection.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
model.eval()
tok = get_tokenizer()
print(f"Loaded representative checkpoint: {os.path.basename(CKPT_PATH)}")

TRUNCATIONS = [0.25, 0.50, 0.75, 1.00]

with torch.no_grad():
    verb_embs = model.encode_text(tok.tokenize(verb_vocab).to(DEVICE), normalize=True)
    logit_scale = model.logit_scale.exp()

rows = []
for frac in TRUNCATIONS:
    c1 = c5 = 0
    all_pred, all_true = [], []
    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc=f"frac={frac}"):
        stack = torch.load(os.path.join(PERFRAME_DIR, f"{row['segment_id']}.pt"), map_location='cpu')
        T = stack.shape[0]
        k = max(1, int(np.ceil(frac * T)))
        truncated_mean = stack[:k].mean(dim=0)
        emb = F.normalize(truncated_mean.to(DEVICE).unsqueeze(0), dim=-1)
        with torch.no_grad():
            sim = (logit_scale * emb @ verb_embs.T).squeeze(0)
        pred = sim.argmax().item()
        gt = label2id[row['extracted_verb']]
        top5 = sim.topk(5).indices.cpu().tolist()
        c1 += int(pred == gt)
        c5 += int(gt in top5)
        all_pred.append(pred); all_true.append(gt)
    n = len(df_val)
    macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"frac={frac:.2f}  Top-1={c1/n*100:.2f}%  Top-5={c5/n*100:.2f}%  macro-F1={macro_f1*100:.2f}%")
    rows.append({'model': 'dinov2_meanpool', 'frac_visible': frac, 'top1': c1/n, 'top5': c5/n,
                 'macro_f1': macro_f1, 'n': n})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT_DIR, 'nb8_truncated_clip_dinov2.csv'), index=False)
print(f"\nSaved -> nb8_truncated_clip_dinov2.csv")
print("DONE")
