"""
Macro-F1 for the already-trained representative checkpoints, without retraining.

Top-1 on this dataset is flattered by skew: 'add' alone is 35.65% of the
validation set, which is exactly what the blind baseline scores. Macro-F1 is the
metric that does not reward that, so it belongs next to every Top-1 number in the
paper. Scripts 01 and 06 now compute it during training (2026-09-02), but the
ranks CSVs written before that change stored only rank-derived fields -- no
predicted label -- so macro-F1 could not be recovered from them.

This script closes that gap for the checkpoints already on disk: it reloads each
backbone's representative projection head, re-runs the full-validation ranking
(cheap -- the text tower is encoded once, the rest is a matmul over cached
embeddings), and writes the predicted label alongside the existing fields. It
does NOT retrain anything, so the numbers correspond to the 2-seed run, not the
5-seed run scripts 01/06 are now configured for.

Sanity check: per-segment top1_correct is compared against the existing ranks CSV
and the script aborts on mismatch, since a mismatch would mean the representative
checkpoint being reloaded is not the one those results came from.

Reads : checkpoints/nb8_<tag>_seed<s>_bestepoch<e>.pt
        processed/nb8_matched_projection_full_history.csv, nb8_<tag>_full_history.csv
        processed/nb8_ranks_<tag>_representative.csv
Writes: processed/nb8_ranks_<tag>_representative.csv  (adds a top1_pred column)
        processed/nb8_macro_f1_recomputed.csv
"""
import os, sys, json, glob, re
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
OUT_DIR, CKPT_DIR = PATHS['processed'], PATHS['checkpoints']

sys.path.insert(0, os.path.join(PATHS['repos'], 'dinov2'))
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

DINOV3_REPO = os.path.join(PATHS['repos'], 'dinov3')
DINOV3_BACKBONE_WEIGHTS = '/home/ubuntu/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_DINOTXT_WEIGHTS = '/home/ubuntu/weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
print(f"Validation segments: {len(df_val):,} | verbs: {len(verb_vocab)}")

tokenizer_v2 = get_tokenizer()


def load_dinov2():
    m = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m, tokenizer_v2


def load_dinov3():
    m, tok = torch.hub.load(DINOV3_REPO, 'dinov3_vitl16_dinotxt_tet1280d20h24l', source='local',
                            weights=DINOV3_DINOTXT_WEIGHTS, backbone_weights=DINOV3_BACKBONE_WEIGHTS)
    m = m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m, tok


BACKBONES = {
    'dinov2': dict(loader=load_dinov2, emb_dir=PATHS['embeddings'], in_dim=1280, out_dim=2048,
                   history='nb8_matched_projection_full_history.csv'),
    'dinov3': dict(loader=load_dinov3, emb_dir=os.path.join(PATHS['embeddings'], 'dinov3'),
                   in_dim=1280, out_dim=2048, history='nb8_matched_projection_full_history.csv'),
    'vjepa2': dict(loader=load_dinov2, emb_dir=os.path.join(PATHS['embeddings'], 'vjepa2_dinotxt'),
                   in_dim=1280, out_dim=1024, history='nb8_matched_projection_full_history.csv'),
    'vjepa2_pretrainedpool': dict(loader=load_dinov2,
                                  emb_dir=os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool'),
                                  in_dim=1280, out_dim=1024,
                                  history='nb8_vjepa2_pretrainedpool_full_history.csv'),
}


def candidate_checkpoints(tag, cfg):
    """Every per-seed selected-epoch checkpoint on disk for this backbone, with the
    Top-1 recorded for it during training.

    Which one is "the representative" is deliberately NOT re-derived here. Script 01
    picks the seed whose Top-1 is closest to the mean; with only 2 seeds both seeds
    are exactly equidistant from their own mean, so that rule is decided by
    floating-point rounding inside min() -- and script 02 hardcodes a checkpoint
    filename for the same purpose. For 'vjepa2' the two disagree (script 01's rule
    lands on seed 1; script 02 hardcodes nb8_vjepa2_seed1_bestepoch9.pt, while a
    naive re-derivation of the rule lands on seed 0, a checkpoint whose Top-1 is
    0.15pp higher). Rather than guess, match against the published ranks CSV below.

    This ambiguity disappears at N_SEEDS=5, where no exact tie can arise.
    """
    hist = pd.read_csv(os.path.join(OUT_DIR, cfg['history']))
    hist = hist[hist['tag'] == tag]
    if hist.empty:
        return []
    best = (hist.sort_values(['val_top1', 'val_top5'], ascending=False)
                .groupby('seed').first().reset_index())
    out = []
    for _, r in best.iterrows():
        seed, epoch = int(r['seed']), int(r['epoch'])
        path = os.path.join(CKPT_DIR, f'nb8_{tag}_seed{seed}_bestepoch{epoch}.pt')
        if os.path.exists(path):
            out.append((path, dict(seed=seed, epoch=epoch, val_top1=r['val_top1'])))
    return out


@torch.no_grad()
def rank_val(model, tok, emb_dir):
    verb_embs = model.encode_text(tok.tokenize(verb_vocab).to(DEVICE), normalize=True)
    scale = model.logit_scale.exp()
    records = []
    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Ranking", leave=False):
        emb_path = os.path.join(emb_dir, f"{row['segment_id']}.pt")
        if not os.path.exists(emb_path):
            continue
        v = F.normalize(torch.load(emb_path, map_location=DEVICE).unsqueeze(0), dim=-1)
        ranked = (scale * v @ verb_embs.T).squeeze(0).cpu().argsort(descending=True).tolist()
        gt = label2id.get(row['extracted_verb'], -1)
        rank = ranked.index(gt) + 1 if gt in ranked else len(verb_vocab) + 1
        records.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                        'top1_pred': verb_vocab[ranked[0]], 'rank': rank,
                        'reciprocal_rank': 1.0 / rank,
                        'top1_correct': rank == 1, 'top5_correct': rank <= 5})
    return pd.DataFrame(records).set_index('segment_id')


rows = []
for tag, cfg in BACKBONES.items():
    candidates = candidate_checkpoints(tag, cfg)
    if not candidates:
        print(f"[{tag}] no per-seed checkpoints on disk -- skipping")
        continue

    ranks_csv = os.path.join(OUT_DIR, f'nb8_ranks_{tag}_representative.csv')
    published = pd.read_csv(ranks_csv).set_index('segment_id') if os.path.exists(ranks_csv) else None

    chosen = None
    for ckpt_path, info in candidates:
        print(f"\n[{tag}] trying seed {info['seed']} epoch {info['epoch']} "
              f"(training-time val top-1 {info['val_top1']:.4f}) -- {os.path.basename(ckpt_path)}", flush=True)
        try:
            model, tok = cfg['loader']()
        except Exception as e:
            # DINOv3 needs two local .pth files under /home/ubuntu/weights that are not
            # part of this repo. Skip rather than abort so other backbones still report.
            print(f"  SKIPPED -- could not load the {tag} tower: {type(e).__name__}: {e}")
            break
        model.text_model.head.linear_projection = nn.Linear(cfg['in_dim'], cfg['out_dim'], bias=False).to(DEVICE)
        state = torch.load(ckpt_path, map_location='cpu')
        model.text_model.head.linear_projection.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
        model.eval()

        ranks = rank_val(model, tok, cfg['emb_dir'])
        del model
        torch.cuda.empty_cache()

        if published is None:
            chosen = (ranks, info, 1.0)
            break
        shared = published.index.intersection(ranks.index)
        agree = (published.loc[shared, 'top1_correct'].astype(bool)
                 == ranks.loc[shared, 'top1_correct'].astype(bool)).mean()
        print(f"  reproduces the published ranks CSV on {agree:.2%} of {len(shared):,} segments")
        if agree > 0.999:
            chosen = (ranks, info, agree)
            break

    if chosen is None:
        print(f"[{tag}] NO checkpoint on disk reproduces {os.path.basename(ranks_csv)} -- "
              f"not reporting a macro-F1 for it rather than reporting one from the wrong head")
        continue

    ranks, info, agree = chosen
    print(f"  -> representative confirmed: seed {info['seed']} epoch {info['epoch']}")
    ranks.to_csv(ranks_csv)

    micro = ranks['top1_correct'].mean()
    macro_recall = ranks.groupby('ground_truth')['top1_correct'].mean().mean()
    macro_f1 = f1_score(ranks['ground_truth'], ranks['top1_pred'], average='macro', zero_division=0)
    weighted_f1 = f1_score(ranks['ground_truth'], ranks['top1_pred'], average='weighted', zero_division=0)
    rows.append({'model': tag, 'rep_seed': info['seed'], 'rep_epoch': info['epoch'],
                 'micro_top1': micro, 'macro_top1_recall': macro_recall,
                 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
                 'n_distinct_verbs_predicted': ranks['top1_pred'].nunique(),
                 'n': len(ranks)})
    print(f"  Top-1 {micro*100:.2f}% | macro-recall {macro_recall*100:.2f}% | "
          f"macro-F1 {macro_f1*100:.2f}% | verbs ever predicted {ranks['top1_pred'].nunique()}/{len(verb_vocab)}")

# The blind baseline, scored the same way -- the reason macro-F1 is worth reporting.
blind_pred = df_val['extracted_verb'].value_counts().idxmax()
rows.append({'model': f'BLIND (always "{blind_pred}")', 'rep_seed': np.nan, 'rep_epoch': np.nan,
             'micro_top1': (df_val['extracted_verb'] == blind_pred).mean(),
             'macro_top1_recall': 1.0 / len(verb_vocab),
             'macro_f1': f1_score(df_val['extracted_verb'], [blind_pred] * len(df_val),
                                  average='macro', zero_division=0),
             'weighted_f1': f1_score(df_val['extracted_verb'], [blind_pred] * len(df_val),
                                     average='weighted', zero_division=0),
             'n_distinct_verbs_predicted': 1, 'n': len(df_val)})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT_DIR, 'nb8_macro_f1_recomputed.csv'), index=False)
print("\n=== Macro-F1 next to Top-1 (2-seed run, representative checkpoints) ===")
print(out.to_string(index=False))
print(f"\nSaved -> {os.path.join(OUT_DIR, 'nb8_macro_f1_recomputed.csv')}")
print("DONE")
