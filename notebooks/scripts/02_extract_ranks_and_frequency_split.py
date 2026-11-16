"""
Frequent-vs-rare verb breakdown across all three backbones' representative
matched-projection checkpoints, using the same threshold convention NB6 used
for its own (V-JEPA2-only) breakdown: frequent = >=50 train occurrences,
rare = <10 train occurrences. No retraining -- reloads the already-saved
representative checkpoints and does one fast ranking pass per backbone
(bare-verb embeddings computed once, cheap).

Also saves per-segment ranks to CSV so future re-analyses (e.g. a corrected
motion/static split) don't need to reload models again.
"""
import os, sys, json
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
CKPT_DIR = PATHS['checkpoints']

DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
DINOV3_REPO = os.path.join(PATHS['repos'], 'dinov3')
if DINOV3_REPO not in sys.path:
    sys.path.insert(0, DINOV3_REPO)

from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

DINOV3_BACKBONE_WEIGHTS = '/home/ubuntu/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_DINOTXT_WEIGHTS  = '/home/ubuntu/weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'

def load_fresh_dinov2_dinotxt():
    m = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

tokenizer_v2 = get_tokenizer()

def load_fresh_dinov3_dinotxt():
    m, tok = torch.hub.load(
        DINOV3_REPO, 'dinov3_vitl16_dinotxt_tet1280d20h24l',
        source='local', weights=DINOV3_DINOTXT_WEIGHTS, backbone_weights=DINOV3_BACKBONE_WEIGHTS,
    )
    m = m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m, tok

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
n_verbs = len(verb_vocab)
df_train = df[df['subset'] == 'training'].reset_index(drop=True)
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)

EMB_DIR_DINOV2 = PATHS['embeddings']
_dinov3_candidates = [os.path.join(PATHS['embeddings'], 'dinov3'), os.path.join(PATHS['embeddings'], 'dinov2', 'dinov3')]
EMB_DIR_DINOV3 = next((p for p in _dinov3_candidates if os.path.isdir(p) and os.listdir(p)), _dinov3_candidates[0])
EMB_DIR_VJEPA2 = os.path.join(PATHS['embeddings'], 'vjepa2_dinotxt')

# Same threshold convention NB6 used for its own (V-JEPA2-only) breakdown
verb_counts = df_train['extracted_verb'].value_counts()
frequent_verbs = set(verb_counts[verb_counts >= 50].index)
rare_verbs = set(verb_counts[verb_counts < 10].index)
def freq_bucket(v):
    if v in frequent_verbs: return 'frequent (>=50 train)'
    if v in rare_verbs: return 'rare (<10 train)'
    return 'mid'
print(f"Frequent verbs: {len(frequent_verbs)}  Rare verbs: {len(rare_verbs)}  "
      f"Mid: {n_verbs - len(frequent_verbs) - len(rare_verbs)}", flush=True)

REPRESENTATIVE = {
    'dinov2': dict(loader=load_fresh_dinov2_dinotxt, emb_dir=EMB_DIR_DINOV2, in_dim=1280, out_dim=2048,
                    ckpt=os.path.join(CKPT_DIR, 'nb8_dinov2_seed0_bestepoch9.pt')),
    'dinov3': dict(loader=load_fresh_dinov3_dinotxt, emb_dir=EMB_DIR_DINOV3, in_dim=1280, out_dim=2048,
                    ckpt=os.path.join(CKPT_DIR, 'nb8_dinov3_seed0_bestepoch9.pt')),
    'vjepa2': dict(loader=load_fresh_dinov2_dinotxt, emb_dir=EMB_DIR_VJEPA2, in_dim=1280, out_dim=1024,
                    ckpt=os.path.join(CKPT_DIR, 'nb8_vjepa2_seed1_bestepoch9.pt')),
}

@torch.no_grad()
def eval_with_ranks(model, tok, emb_dir, df_subset):
    verb_tokens = tok.tokenize(verb_vocab).to(DEVICE)
    verb_embs = model.encode_text(verb_tokens, normalize=True)
    records = []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Ranking"):
        emb_path = os.path.join(emb_dir, f"{row['segment_id']}.pt")
        if not os.path.exists(emb_path):
            continue
        video_emb = torch.load(emb_path, map_location=DEVICE)
        video_emb = F.normalize(video_emb.unsqueeze(0), dim=-1)
        sim = (model.logit_scale.exp() * video_emb @ verb_embs.T).squeeze(0).cpu()
        ranked = sim.argsort(descending=True).tolist()
        gt_idx = label2id.get(row['extracted_verb'], -1)
        rank = ranked.index(gt_idx) + 1 if gt_idx in ranked else n_verbs + 1
        records.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                         'rank': rank, 'reciprocal_rank': 1.0 / rank,
                         'top1_correct': rank == 1, 'top5_correct': rank <= 5})
    return pd.DataFrame(records).set_index('segment_id')

all_ranks = {}
for tag, cfg in REPRESENTATIVE.items():
    print(f"\n=== {tag}: loading representative checkpoint {cfg['ckpt']} ===", flush=True)
    loaded = cfg['loader']()
    model, tok = loaded if isinstance(loaded, tuple) else (loaded, tokenizer_v2)
    model.text_model.head.linear_projection = nn.Linear(cfg['in_dim'], cfg['out_dim'], bias=False).to(DEVICE)
    state = torch.load(cfg['ckpt'], map_location=DEVICE)
    model.text_model.head.linear_projection.load_state_dict(state)
    model.eval()

    ranks = eval_with_ranks(model, tok, cfg['emb_dir'], df_val)
    all_ranks[tag] = ranks
    ranks.to_csv(os.path.join(OUT_DIR, f'nb8_ranks_{tag}_representative.csv'))
    print(f"  N={len(ranks):,}  Top-1={ranks['top1_correct'].mean()*100:.2f}%  "
          f"Top-5={ranks['top5_correct'].mean()*100:.2f}%", flush=True)

    del model
    torch.cuda.empty_cache()

# ── Frequent vs rare breakdown ──────────────────────────────────────────────
rows = []
for tag, ranks in all_ranks.items():
    df_r = ranks.copy()
    df_r['bucket'] = df_r['ground_truth'].apply(freq_bucket)
    for bucket, grp in df_r.groupby('bucket'):
        rows.append({
            'model': tag, 'bucket': bucket, 'n': len(grp),
            'micro_top1': grp['top1_correct'].mean(), 'micro_top5': grp['top5_correct'].mean(),
            'macro_top1': grp.groupby('ground_truth')['top1_correct'].mean().mean(),
            'macro_top5': grp.groupby('ground_truth')['top5_correct'].mean().mean(),
            'mrr': grp['reciprocal_rank'].mean(),
        })
freq_split_df = pd.DataFrame(rows)
print("\n=== Frequent vs. rare verb breakdown ===")
print(freq_split_df.to_string(index=False))
freq_split_df.to_csv(os.path.join(OUT_DIR, 'nb8_frequent_vs_rare_split.csv'), index=False)
print(f"\nSaved -> {os.path.join(OUT_DIR, 'nb8_frequent_vs_rare_split.csv')}", flush=True)
print("DONE", flush=True)
