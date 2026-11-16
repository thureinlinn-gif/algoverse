"""
Evaluate script 30's truncated V-JEPA2 embeddings with the SAME representative
checkpoint already trained on full clips (nb8_vjepa2_pretrainedpool_seed1_bestepoch9.pt,
no retraining), and put the result side by side with script 29's DINOv2 numbers.

This is the actual test: does V-JEPA2's predictive pretraining give it a real
edge recognizing actions from partial context, or does it degrade with
truncation the same way DINOv2 does?

Reads : embeddings/<...>/vjepa2_pretrainedpool_trunc{25,50,75}/ (script 30)
        embeddings/<...>/vjepa2_pretrainedpool/ (script 12, for the frac=1.00 reference)
        checkpoints/nb8_vjepa2_pretrainedpool_seed1_bestepoch9.pt
        processed/nb8_truncated_clip_dinov2.csv (script 29)
Writes: processed/nb8_truncated_clip_vjepa2.csv
        processed/nb8_truncated_clip_comparison.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
CKPT_DIR = PATHS['checkpoints']
IN_DIM, OUT_DIM = 1280, 1024
CKPT_PATH = os.path.join(CKPT_DIR, 'nb8_vjepa2_pretrainedpool_seed1_bestepoch9.pt')

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

model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
state = torch.load(CKPT_PATH, map_location='cpu')
model.text_model.head.linear_projection.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
model.eval()
tok = get_tokenizer()
print(f"Loaded representative checkpoint: {os.path.basename(CKPT_PATH)}")

with torch.no_grad():
    verb_embs = model.encode_text(tok.tokenize(verb_vocab).to(DEVICE), normalize=True)
    logit_scale = model.logit_scale.exp()

TRUNC_DIRS = {
    0.25: os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_trunc25'),
    0.50: os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_trunc50'),
    0.75: os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_trunc75'),
    1.00: os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool'),  # script 12, already full corpus
}

rows = []
for frac, emb_dir in TRUNC_DIRS.items():
    c1 = c5 = n = 0
    all_pred, all_true = [], []
    for _, row in df_val.iterrows():
        p = os.path.join(emb_dir, f"{row['segment_id']}.pt")
        if not os.path.exists(p):
            continue
        emb = F.normalize(torch.load(p, map_location=DEVICE).unsqueeze(0), dim=-1)
        with torch.no_grad():
            sim = (logit_scale * emb @ verb_embs.T).squeeze(0)
        pred = sim.argmax().item()
        gt = label2id[row['extracted_verb']]
        top5 = sim.topk(5).indices.cpu().tolist()
        c1 += int(pred == gt); c5 += int(gt in top5); n += 1
        all_pred.append(pred); all_true.append(gt)
    macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"frac={frac:.2f}  n={n}  Top-1={c1/n*100:.2f}%  Top-5={c5/n*100:.2f}%  macro-F1={macro_f1*100:.2f}%")
    rows.append({'model': 'vjepa2_pretrainedpool', 'frac_visible': frac, 'top1': c1/n, 'top5': c5/n,
                 'macro_f1': macro_f1, 'n': n})

vj_out = pd.DataFrame(rows)
vj_out.to_csv(os.path.join(OUT_DIR, 'nb8_truncated_clip_vjepa2.csv'), index=False)

# ── Consolidated comparison ──────────────────────────────────────────────────
d2_path = os.path.join(OUT_DIR, 'nb8_truncated_clip_dinov2.csv')
if os.path.exists(d2_path):
    d2_out = pd.read_csv(d2_path)
    combined = pd.concat([d2_out, vj_out], ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, 'nb8_truncated_clip_comparison.csv'), index=False)
    print("\n" + "=" * 78)
    print("COMPARISON -- Top-1 by truncation level")
    print("=" * 78)
    pivot = combined.pivot(index='frac_visible', columns='model', values='top1') * 100
    pivot['gap (vjepa2 - dinov2), pp'] = (pivot['vjepa2_pretrainedpool'] - pivot['dinov2_meanpool'])
    print(pivot.to_string(float_format=lambda x: f'{x:.2f}'))
    print("\nIf V-JEPA2's predictive pretraining gives it a real partial-context advantage,")
    print("the gap column should GROW as frac_visible shrinks (V-JEPA2 degrades slower).")
    print("If the gap is flat or shrinks, this hypothesis doesn't hold either.")
else:
    print(f"\nWARNING: {d2_path} not found -- run script 29 first for the full comparison table.")

print("\nDONE")
