"""
Empirical "needs motion vs. single-frame-guessable" verb classification.

Methodology (Schindler & Van Gool 2008; Sevilla-Lara et al. 2021 "Only Time
Can Tell"): don't assign this from linguistic intuition -- measure it. For
each val segment, extract a single (middle) frame and run it through the
already-trained DINOv2 and DINOv3 representative projections (no retraining).
Compare single-frame accuracy to the existing full-clip (50-frame-pooled)
accuracy, per verb. A verb where BOTH DINOv2 and DINOv3 show a real accuracy
drop is classified "needs motion"; where neither drops, "guessable"; anything
else is left unclassified rather than forced into a bucket.

V-JEPA2 is deliberately NOT used to define this criterion -- it's the model
being tested against it, so using it to build the criterion would be circular.
Only after the classification is built (from DINOv2+DINOv3 alone) do we bring
in V-JEPA2's existing full-clip ranks to test whether its performance (edge
or deficit) concentrates in the needs-motion bucket.
"""
import os
# Must be set before dinov2's attention.py is imported. xFormers is pip-installed
# here but its compiled CUDA extension targets a different PyTorch/CUDA build and
# only supports fp16/bf16 -- it silently "looks available" at import time but
# crashes (NotImplementedError) the first time encode_image() is actually called
# with fp32 input. Never triggered before in this session because everything
# prior only called encode_text (a different code path) or used precomputed
# embeddings. Disabling forces the eager PyTorch attention fallback instead.
os.environ['XFORMERS_DISABLED'] = '1'

import sys, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from decord import VideoReader, cpu
from PIL import Image
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
CKPT_DIR = PATHS['checkpoints']
VIDEO_DIR = PATHS['videos']

DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
DINOV3_REPO = os.path.join(PATHS['repos'], 'dinov3')
if DINOV3_REPO not in sys.path:
    sys.path.insert(0, DINOV3_REPO)

from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer
from dinov2.data.transforms import make_classification_eval_transform as make_transform_v2
from dinov3.data.transforms import make_classification_eval_transform as make_transform_v3

DINOV3_BACKBONE_WEIGHTS = '/home/ubuntu/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_DINOTXT_WEIGHTS  = '/home/ubuntu/weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
n_verbs = len(verb_vocab)
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
print(f"Val segments: {len(df_val):,}  Unique videos: {df_val['video_id'].nunique():,}", flush=True)

# ── Load both models ─────────────────────────────────────────────────────────
print("Loading DINOv2...", flush=True)
model_v2 = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
tokenizer_v2 = get_tokenizer()
for p in model_v2.parameters():
    p.requires_grad = False
model_v2.text_model.head.linear_projection = nn.Linear(1280, 2048, bias=False).to(DEVICE)
model_v2.text_model.head.linear_projection.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, 'nb8_dinov2_seed0_bestepoch9.pt'), map_location=DEVICE))
model_v2.eval()
transform_v2 = make_transform_v2()

print("Loading DINOv3...", flush=True)
model_v3, tokenizer_v3 = torch.hub.load(
    DINOV3_REPO, 'dinov3_vitl16_dinotxt_tet1280d20h24l',
    source='local', weights=DINOV3_DINOTXT_WEIGHTS, backbone_weights=DINOV3_BACKBONE_WEIGHTS,
)
model_v3 = model_v3.to(DEVICE).eval()
for p in model_v3.parameters():
    p.requires_grad = False
model_v3.text_model.head.linear_projection = nn.Linear(1280, 2048, bias=False).to(DEVICE)
model_v3.text_model.head.linear_projection.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, 'nb8_dinov3_seed0_bestepoch9.pt'), map_location=DEVICE))
model_v3.eval()
transform_v3 = make_transform_v3()
print("Both models loaded.", flush=True)

with torch.no_grad():
    verb_embs_v2 = model_v2.encode_text(tokenizer_v2.tokenize(verb_vocab).to(DEVICE), normalize=True)
    verb_embs_v3 = model_v3.encode_text(tokenizer_v3.tokenize(verb_vocab).to(DEVICE), normalize=True)

# ── Extract single (middle) frame per segment, video-by-video (avoid reopening
# the same video file per segment), run both backbones immediately ───────────
records_v2, records_v3 = [], []
by_video = df_val.groupby('video_id')

for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="Single-frame eval"):
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        continue
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"  skip {video_id}: {e}", flush=True)
        continue
    fps = vr.get_avg_fps()

    for _, row in grp.iterrows():
        start_idx = max(0, int(row['start_sec'] * fps))
        end_idx = min(len(vr) - 1, int(row['end_sec'] * fps))
        mid_idx = (start_idx + end_idx) // 2
        try:
            frame = Image.fromarray(vr[mid_idx].asnumpy())
        except Exception:
            continue

        gt_idx = label2id.get(row['extracted_verb'], -1)

        with torch.no_grad():
            t2 = transform_v2(frame).unsqueeze(0).to(DEVICE)
            emb2 = model_v2.encode_image(t2, normalize=True)
            sim2 = (model_v2.logit_scale.exp() * emb2 @ verb_embs_v2.T).squeeze(0).cpu()
            ranked2 = sim2.argsort(descending=True).tolist()
            rank2 = ranked2.index(gt_idx) + 1 if gt_idx in ranked2 else n_verbs + 1
            records_v2.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                                'rank': rank2, 'top1_correct': rank2 == 1, 'top5_correct': rank2 <= 5})

            with torch.autocast('cuda', dtype=torch.float):
                t3 = transform_v3(frame).unsqueeze(0).to(DEVICE)
                emb3 = model_v3.encode_image(t3, normalize=True)
            sim3 = (model_v3.logit_scale.exp() * emb3 @ verb_embs_v3.T).squeeze(0).cpu()
            ranked3 = sim3.argsort(descending=True).tolist()
            rank3 = ranked3.index(gt_idx) + 1 if gt_idx in ranked3 else n_verbs + 1
            records_v3.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                                'rank': rank3, 'top1_correct': rank3 == 1, 'top5_correct': rank3 <= 5})

sf_v2 = pd.DataFrame(records_v2).set_index('segment_id')
sf_v3 = pd.DataFrame(records_v3).set_index('segment_id')
sf_v2.to_csv(os.path.join(OUT_DIR, 'nb8_singleframe_ranks_dinov2.csv'))
sf_v3.to_csv(os.path.join(OUT_DIR, 'nb8_singleframe_ranks_dinov3.csv'))
print(f"\nSingle-frame DINOv2 — N={len(sf_v2):,} Top-1={sf_v2['top1_correct'].mean()*100:.2f}% "
      f"Top-5={sf_v2['top5_correct'].mean()*100:.2f}%", flush=True)
print(f"Single-frame DINOv3 — N={len(sf_v3):,} Top-1={sf_v3['top1_correct'].mean()*100:.2f}% "
      f"Top-5={sf_v3['top5_correct'].mean()*100:.2f}%", flush=True)

# ── Compare to existing full-clip ranks, per verb ───────────────────────────
full_v2 = pd.read_csv(os.path.join(OUT_DIR, 'nb8_ranks_dinov2_representative.csv')).set_index('segment_id')
full_v3 = pd.read_csv(os.path.join(OUT_DIR, 'nb8_ranks_dinov3_representative.csv')).set_index('segment_id')

def per_verb_acc(ranks_df, metric='top1_correct'):
    return ranks_df.groupby('ground_truth')[metric].mean()

full_v2_acc = per_verb_acc(full_v2)
full_v3_acc = per_verb_acc(full_v3)
sf_v2_acc = per_verb_acc(sf_v2)
sf_v3_acc = per_verb_acc(sf_v3)

verb_test = pd.DataFrame({
    'full_v2': full_v2_acc, 'sf_v2': sf_v2_acc,
    'full_v3': full_v3_acc, 'sf_v3': sf_v3_acc,
}).fillna(0.0)
verb_test['drop_v2'] = verb_test['full_v2'] - verb_test['sf_v2']
verb_test['drop_v3'] = verb_test['full_v3'] - verb_test['sf_v3']

# Classification: needs motion if BOTH show a real drop (>=0.05 abs Top-1, and
# full-clip accuracy itself isn't just noise-floor near zero); guessable if
# BOTH show near-zero/negative drop. Everything else left unclassified.
DROP_THRESH = 0.05
def classify(row):
    if row['drop_v2'] >= DROP_THRESH and row['drop_v3'] >= DROP_THRESH:
        return 'needs_motion'
    if row['drop_v2'] <= 0.01 and row['drop_v3'] <= 0.01:
        return 'guessable'
    return 'disagree_or_ambiguous'

verb_test['classification'] = verb_test.apply(classify, axis=1)
verb_test = verb_test.reset_index().rename(columns={'index': 'verb'})
verb_test.to_csv(os.path.join(OUT_DIR, 'nb8_motion_classification_empirical.csv'), index=False)
print("\n" + verb_test['classification'].value_counts().to_string(), flush=True)

# ── Now bring in V-JEPA2 (and DINOv2/DINOv3) full-clip performance, bucketed
# by this classification, to test where V-JEPA2's edge/deficit concentrates ──
full_vj = pd.read_csv(os.path.join(OUT_DIR, 'nb8_ranks_vjepa2_representative.csv')).set_index('segment_id')

verb_to_class = dict(zip(verb_test['verb'], verb_test['classification']))
rows = []
for tag, ranks in [('dinov2', full_v2), ('dinov3', full_v3), ('vjepa2', full_vj)]:
    df_r = ranks.copy()
    df_r['classification'] = df_r['ground_truth'].map(verb_to_class)
    for cls, grp in df_r.groupby('classification'):
        rows.append({
            'model': tag, 'classification': cls, 'n_segments': len(grp),
            'n_verbs': grp['ground_truth'].nunique(),
            'micro_top1': grp['top1_correct'].mean(), 'micro_top5': grp['top5_correct'].mean(),
            'macro_top1': grp.groupby('ground_truth')['top1_correct'].mean().mean(),
        })
final_df = pd.DataFrame(rows)
print("\n=== Full-clip performance bucketed by empirical motion classification ===")
print(final_df.to_string(index=False))
final_df.to_csv(os.path.join(OUT_DIR, 'nb8_vjepa2_advantage_by_motion.csv'), index=False)
print(f"\nSaved -> {os.path.join(OUT_DIR, 'nb8_vjepa2_advantage_by_motion.csv')}", flush=True)
print("DONE", flush=True)
