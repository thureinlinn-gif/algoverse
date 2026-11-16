import sys
print('--- NB8 script starting ---', flush=True)

# ===== cell 60c02580 =====
import os, sys, json, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# Fixed 2026-08-26: TF32 matmul is off by default in this PyTorch version. Without
# it, DINOv3's text-tower forward (fp32) measured ~1.9s/call for a 282-verb batch —
# with it, ~0.75s/call (2.5x), no change to model dtype/precision otherwise. This
# alone was the difference between Part 3's DINOv3 training taking ~5h vs ~1-1.5h.
# The original NB5b notebook worked around the same issue locally via
# torch.autocast(dtype=torch.float) around its DINOv3 calls; this generalizes that
# fix globally so every backbone benefits, not just DINOv3.
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS.get('processed', '/home/ubuntu/youcook2-videos/processed')
CKPT_DIR = PATHS.get('checkpoints', '/home/ubuntu/youcook2-videos/checkpoints')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===== cell ? =====
# ── Embedding directories — adjust here if your paths.json differs ────────────
EMB_DIR_DINOV2 = PATHS['embeddings']
_dinov3_candidates = [
    os.path.join(PATHS['embeddings'], 'dinov3'),
    os.path.join(PATHS['embeddings'], 'dinov2', 'dinov3'),
]
EMB_DIR_DINOV3 = next((p for p in _dinov3_candidates if os.path.isdir(p) and os.listdir(p)), _dinov3_candidates[0])
EMB_DIR_VJEPA2 = os.path.join(PATHS['embeddings'], 'vjepa2_dinotxt')

for name, path in [('DINOv2', EMB_DIR_DINOV2), ('DINOv3', EMB_DIR_DINOV3), ('V-JEPA2', EMB_DIR_VJEPA2)]:
    n = len([f for f in os.listdir(path) if f.endswith('.pt')]) if os.path.isdir(path) else 0
    print(f"{name:<8} embeddings dir: {path}  ({n:,} files)")


# ===== cell ? =====
df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))

verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id   = {v: i for i, v in enumerate(verb_vocab)}
id2label   = {i: v for v, i in label2id.items()}
n_verbs    = len(verb_vocab)
df['verb_label'] = df['extracted_verb'].map(label2id)

df_train = df[df['subset'] == 'training'].reset_index(drop=True)
df_val   = df[df['subset'] == 'validation'].reset_index(drop=True)

assert n_verbs == 282, f"Expected 282 verbs, got {n_verbs}"
print(f"Total segments : {len(df):,}")
print(f"Verb classes   : {n_verbs}")
print(f"Train / Val    : {len(df_train):,} / {len(df_val):,}")


# ===== cell ? =====
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

def load_fresh_dinov2_dinotxt():
    """Fresh, fully-frozen DINOv2-based DinoTXT. Used both for the DINOv2 branch
    and, with a differently-sized projection, the V-JEPA2 branch (V-JEPA2 reuses
    this same text tower — see NB6)."""
    m = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

tokenizer_v2 = get_tokenizer()

DINOV3_REPO = os.path.join(PATHS['repos'], 'dinov3')
if DINOV3_REPO not in sys.path:
    sys.path.insert(0, DINOV3_REPO)

DINOV3_BACKBONE_WEIGHTS = '/home/ubuntu/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_DINOTXT_WEIGHTS  = '/home/ubuntu/weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'

def load_fresh_dinov3_dinotxt():
    m, tok = torch.hub.load(
        DINOV3_REPO, 'dinov3_vitl16_dinotxt_tet1280d20h24l',
        source='local', weights=DINOV3_DINOTXT_WEIGHTS, backbone_weights=DINOV3_BACKBONE_WEIGHTS,
    )
    m = m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m, tok

print("Model loaders ready.")


# ===== cell da7a3c06 =====
def try_load_csv(*candidates):
    for c in candidates:
        if c and os.path.exists(c):
            d = pd.read_csv(c)
            print(f"Loaded {c}  ({len(d):,} rows)")
            return d
    print(f"Not found: {candidates[0]} (will recompute below)")
    return None

# Fixed 2026-08-26: these used to point at stale/wrong files (a relative-path
# 'results_checkpoint.jsonl' from an interrupted 566-row run, and a
# 'nb5_dinotxt_zeroshot_results.csv' that was never actually written). Now point
# at the full-val corrected runs completed earlier today (both N=3,268, same
# clean CSV, no slicing) — cells 10/11 below will reuse these instead of
# recomputing ~2h of zero-shot eval per backbone.
dinov2_zeroshot_existing = try_load_csv(
    os.path.join(OUT_DIR, 'dinov2_zeroshot_fullval_fixed.csv'),
)
dinov3_zeroshot_existing = try_load_csv(
    os.path.join(OUT_DIR, 'nb5b_dinov3_dinotxt_zeroshot_results.csv'),
)


# ===== cell ? =====
@torch.no_grad()
def get_text_embeddings_caption_context(dinotxt_model, tok, masked_caption, verbs):
    caption_context = masked_caption.replace('[MASK]', '{}')
    verb_texts = [caption_context.format(v) for v in verbs]
    tokens = tok.tokenize(verb_texts).to(DEVICE)
    text_features = dinotxt_model.encode_text(tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)


def run_zeroshot_eval(dinotxt_model, tok, emb_dir, df_subset, verbs, label2id, desc="Zero-shot"):
    records = []
    missing = 0
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=desc):
        emb_path = os.path.join(emb_dir, f"{row['segment_id']}.pt")
        if not os.path.exists(emb_path):
            missing += 1
            continue
        video_emb = torch.load(emb_path, map_location=DEVICE)
        video_emb = video_emb / video_emb.norm()

        text_embs = get_text_embeddings_caption_context(dinotxt_model, tok, row['masked_caption'], verbs)
        similarity = (text_embs @ video_emb).cpu()

        ranked = similarity.argsort(descending=True).tolist()
        gt_idx = label2id.get(row['extracted_verb'], -1)
        rank = ranked.index(gt_idx) + 1 if gt_idx in ranked else len(verbs) + 1

        records.append({
            'segment_id': row['segment_id'], 'video_id': row['video_id'], 'subset': row['subset'],
            'ground_truth': row['extracted_verb'], 'top1_pred': verbs[ranked[0]],
            'top1_correct': ranked[0] == gt_idx, 'top5_correct': gt_idx in ranked[:5],
            'rank': rank, 'reciprocal_rank': 1.0/rank,
        })
    if missing:
        print(f"  ({missing} segments missing embeddings, skipped)")
    return pd.DataFrame(records)


# ===== cell 488f2adb =====
# Fixed 2026-08-26: reuse the completed full-val run from cell 7 instead of
# always recomputing (recomputing takes ~1.5-2h on this GPU for 3,268 segments,
# and we already have a verified, correctly-sourced result).
if dinov2_zeroshot_existing is not None and len(dinov2_zeroshot_existing) == len(df_val):
    print(f"Reusing existing DINOv2 full-val zero-shot results ({len(dinov2_zeroshot_existing):,} rows) — skipping recompute.")
    dinov2_full_results = dinov2_zeroshot_existing
else:
    if dinov2_zeroshot_existing is not None:
        print(f"Existing DINOv2 file has {len(dinov2_zeroshot_existing):,} rows, expected {len(df_val):,} — recomputing.")
    dinotxt_v2_zs = load_fresh_dinov2_dinotxt()
    dinov2_full_results = run_zeroshot_eval(
        dinotxt_v2_zs, tokenizer_v2, EMB_DIR_DINOV2, df_val, verb_vocab, label2id,
        desc="DINOv2 zero-shot (full val)"
    )
    del dinotxt_v2_zs

print(f"DINOv2 zero-shot — N={len(dinov2_full_results):,}  Top-1={dinov2_full_results['top1_correct'].mean()*100:.2f}%  "
      f"Top-5={dinov2_full_results['top5_correct'].mean()*100:.2f}%")
dinov2_full_results.to_csv(os.path.join(OUT_DIR, 'nb8_dinov2_zeroshot_fullval.csv'), index=False)


# ===== cell 6fcedf44 =====
# Fixed 2026-08-26: same reuse logic as the DINOv2 cell above.
if dinov3_zeroshot_existing is not None and len(dinov3_zeroshot_existing) == len(df_val):
    print(f"Reusing existing DINOv3 full-val zero-shot results ({len(dinov3_zeroshot_existing):,} rows) — skipping recompute.")
    dinov3_full_results = dinov3_zeroshot_existing
else:
    if dinov3_zeroshot_existing is not None:
        print(f"Existing DINOv3 file has {len(dinov3_zeroshot_existing):,} rows, expected {len(df_val):,} — recomputing.")
    dinotxt_v3_zs, tokenizer_v3_zs = load_fresh_dinov3_dinotxt()
    dinov3_full_results = run_zeroshot_eval(
        dinotxt_v3_zs, tokenizer_v3_zs, EMB_DIR_DINOV3, df_val, verb_vocab, label2id,
        desc="DINOv3 zero-shot (full val)"
    )
    del dinotxt_v3_zs

print(f"DINOv3 zero-shot — N={len(dinov3_full_results):,}  Top-1={dinov3_full_results['top1_correct'].mean()*100:.2f}%  "
      f"Top-5={dinov3_full_results['top5_correct'].mean()*100:.2f}%")
dinov3_full_results.to_csv(os.path.join(OUT_DIR, 'nb8_dinov3_zeroshot_fullval.csv'), index=False)


# ===== cell 08863392 =====
# ── Config — turn these down if Lambda cost is a concern ──────────────────────
CANDIDATE_LRS = [3e-5, 1e-4, 3e-4]
SWEEP_EPOCHS  = 2
N_SEEDS       = 5         # 2026-09-02: raised 2 -> 5. Two seeds give a mean +/- std
                           # from two draws, which is not enough to claim a ~0.4pp gap
                           # between backbones is real. Five is. Budget: the earlier
                           # 2-seed/10-epoch run took ~5h including the LR sweep;
                           # the sweep is unchanged, so expect roughly 2.5x the
                           # per-seed portion (~11-12h total on one A10).
NUM_EPOCHS    = 10        # matches the original V-JEPA2 total training budget (5+5)
BATCH_SIZE    = 32
GRAD_ACCUM    = 4
SEEDS         = list(range(N_SEEDS))

print(f"LR grid: {CANDIDATE_LRS}  |  sweep epochs: {SWEEP_EPOCHS}  |  "
      f"seeds: {SEEDS}  |  full epochs per run: {NUM_EPOCHS}")


# ===== cell ? =====
class ProjDataset(torch.utils.data.Dataset):
    def __init__(self, df, emb_dir, tag=''):
        self.records = []
        missing = 0
        for _, row in df.iterrows():
            p = os.path.join(emb_dir, f"{row['segment_id']}.pt")
            if not os.path.exists(p):
                missing += 1
                continue
            self.records.append((p, int(row['verb_label'])))
        if missing:
            print(f"  [{tag}] {missing} segments missing embeddings, excluded")
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        p, label = self.records[idx]
        return torch.load(p, map_location='cpu'), label

def collate(batch):
    embs, labels = zip(*batch)
    return torch.stack(embs), torch.tensor(labels, dtype=torch.long)


def train_one_run(model_loader, emb_dir, in_dim, out_dim, lr, num_epochs, seed, tag,
                   batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM, verbose=True):
    """Trains a fresh linear_projection for `num_epochs` at a fixed LR/seed.
    Returns (history_df, dict(epoch -> state_dict)) so the caller can select and
    keep only the best epoch's weights without re-training.
    model_loader() must return either `model` or `(model, tokenizer)`.
    """
    set_seed(seed)
    loaded = model_loader()
    model, tok = loaded if isinstance(loaded, tuple) else (loaded, tokenizer_v2)

    model.text_model.head.linear_projection = nn.Linear(in_dim, out_dim, bias=False).to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = 'text_model.head.linear_projection' in name

    g = torch.Generator().manual_seed(seed)
    train_ds = ProjDataset(df_train, emb_dir, tag=tag)
    val_ds   = ProjDataset(df_val, emb_dir, tag=tag)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                collate_fn=collate, num_workers=0,
                                                pin_memory=True, generator=g)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                                collate_fn=collate, num_workers=0, pin_memory=True)

    verb_tokens_bare = tok.tokenize(verb_vocab).to(DEVICE)

    def evaluate():
        model.eval()
        with torch.no_grad():
            verb_embs = model.encode_text(verb_tokens_bare, normalize=True)
        c1 = c5 = tot = 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for embs, labels in val_loader:
                embs = F.normalize(embs.to(DEVICE), dim=-1)
                labels = labels.to(DEVICE)
                sim = model.logit_scale.exp() * embs @ verb_embs.T
                pred1 = sim.argmax(-1)
                c1 += (pred1 == labels).sum().item()
                top5 = sim.topk(5, dim=-1).indices
                c5 += (top5 == labels.unsqueeze(-1)).any(-1).sum().item()
                tot += labels.size(0)
                all_pred.append(pred1.cpu()); all_true.append(labels.cpu())
        # Added 2026-09-02: 'add' is 35.65% of the validation set, so Top-1 flatters
        # every model here -- including the blind always-'add' baseline, which scores
        # 35.65% Top-1 but 0.19% macro-F1. Macro-F1 is tracked per epoch alongside it
        # so the two can always be reported together.
        y_pred = torch.cat(all_pred).numpy(); y_true = torch.cat(all_true).numpy()
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return c1/tot, c5/tot, macro_f1

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = []
    state_dicts = {}
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        iterator = tqdm(train_loader, desc=f"[{tag} seed{seed} lr{lr:.0e}] epoch {epoch}/{num_epochs}") if verbose else train_loader
        for step, (embs, labels) in enumerate(iterator, start=1):
            embs = F.normalize(embs.to(DEVICE), dim=-1)
            labels = labels.to(DEVICE)
            verb_embs = model.encode_text(verb_tokens_bare, normalize=True)
            logit_scale = model.logit_scale.exp()
            sim = logit_scale * embs @ verb_embs.T
            loss = F.cross_entropy(sim, labels) / grad_accum
            loss.backward()
            running_loss += loss.item() * grad_accum
            if step % grad_accum == 0:
                optimizer.step(); optimizer.zero_grad()
                with torch.no_grad():
                    model.logit_scale.clamp_(0, math.log(100))
        if step % grad_accum != 0:
            optimizer.step(); optimizer.zero_grad()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
        scheduler.step()

        top1, top5, macro_f1 = evaluate()
        avg_loss = running_loss / len(train_loader)
        if verbose:
            print(f"  [{tag} seed{seed}] epoch {epoch}/{num_epochs} | loss {avg_loss:.4f} "
                  f"| Val Top-1 {top1:.4f} | Val Top-5 {top5:.4f} | Val macro-F1 {macro_f1:.4f}")
        history.append({'tag': tag, 'seed': seed, 'lr': lr, 'epoch': epoch,
                         'loss': avg_loss, 'val_top1': top1, 'val_top5': top5,
                         'val_macro_f1': macro_f1})
        state_dicts[epoch] = {k: v.clone().cpu() for k, v in
                               model.text_model.head.linear_projection.state_dict().items()}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(history), state_dicts


def select_best_epoch(history_df):
    """THE single epoch-selection rule used everywhere in this notebook:
    best validation Top-1, ties broken by Top-5."""
    h = history_df.sort_values(['val_top1', 'val_top5'], ascending=False)
    return h.iloc[0]


# ===== cell ? =====
def lr_sweep(model_loader, emb_dir, in_dim, out_dim, tag, candidate_lrs=CANDIDATE_LRS,
             sweep_epochs=SWEEP_EPOCHS, seed=0):
    print(f"\n=== LR sweep: {tag} ===")
    results = []
    for lr in candidate_lrs:
        hist, _ = train_one_run(model_loader, emb_dir, in_dim, out_dim, lr,
                                 num_epochs=sweep_epochs, seed=seed, tag=f"{tag}_sweep", verbose=False)
        best = select_best_epoch(hist)
        print(f"  lr={lr:.0e} -> best Val Top-1 after {sweep_epochs} epochs: {best['val_top1']:.4f}")
        results.append({'lr': lr, 'val_top1': best['val_top1'], 'val_top5': best['val_top5']})
    results_df = pd.DataFrame(results).sort_values('val_top1', ascending=False)
    best_lr = results_df.iloc[0]['lr']
    print(f"  -> chosen LR for {tag}: {best_lr:.0e}")
    return best_lr, results_df

BACKBONE_CONFIGS = {
    'dinov2':  dict(model_loader=load_fresh_dinov2_dinotxt, emb_dir=EMB_DIR_DINOV2, in_dim=1280, out_dim=2048),
    'dinov3':  dict(model_loader=load_fresh_dinov3_dinotxt, emb_dir=EMB_DIR_DINOV3, in_dim=1280, out_dim=2048),
    'vjepa2':  dict(model_loader=load_fresh_dinov2_dinotxt, emb_dir=EMB_DIR_VJEPA2, in_dim=1280, out_dim=1024),
}

sweep_summary = {}
for tag, cfg in BACKBONE_CONFIGS.items():
    best_lr, sweep_df = lr_sweep(cfg['model_loader'], cfg['emb_dir'], cfg['in_dim'], cfg['out_dim'], tag)
    sweep_summary[tag] = best_lr
    sweep_df.to_csv(os.path.join(OUT_DIR, f'nb8_lr_sweep_{tag}.csv'), index=False)

print("\nChosen LRs:", sweep_summary)


# ===== cell ? =====
# ── Full multi-seed training at each backbone's chosen LR ─────────────────────
all_histories = []
best_epoch_per_seed = {}   # tag -> {seed: (best_row, state_dicts)}

for tag, cfg in BACKBONE_CONFIGS.items():
    best_epoch_per_seed[tag] = {}
    for seed in SEEDS:
        hist, state_dicts = train_one_run(
            cfg['model_loader'], cfg['emb_dir'], cfg['in_dim'], cfg['out_dim'],
            lr=sweep_summary[tag], num_epochs=NUM_EPOCHS, seed=seed, tag=tag,
        )
        all_histories.append(hist)
        best_row = select_best_epoch(hist)
        best_epoch_per_seed[tag][seed] = (best_row, state_dicts[int(best_row['epoch'])])
        print(f"[{tag} seed {seed}] selected epoch {int(best_row['epoch'])} "
              f"— Val Top-1 {best_row['val_top1']:.4f}, Val Top-5 {best_row['val_top5']:.4f}")

        # persist the selected checkpoint to disk
        ckpt_path = os.path.join(CKPT_DIR, f'nb8_{tag}_seed{seed}_bestepoch{int(best_row["epoch"])}.pt')
        torch.save(state_dicts[int(best_row['epoch'])], ckpt_path)

full_history_df = pd.concat(all_histories, ignore_index=True)
full_history_df.to_csv(os.path.join(OUT_DIR, 'nb8_matched_projection_full_history.csv'), index=False)


# ===== cell ? =====
# ── Aggregate: mean ± std across seeds, at the selected epoch ──────────────────
agg_rows = []
for tag in BACKBONE_CONFIGS:
    top1s = [best_epoch_per_seed[tag][s][0]['val_top1'] for s in SEEDS]
    top5s = [best_epoch_per_seed[tag][s][0]['val_top5'] for s in SEEDS]
    f1s   = [best_epoch_per_seed[tag][s][0]['val_macro_f1'] for s in SEEDS]
    epochs_selected = [int(best_epoch_per_seed[tag][s][0]['epoch']) for s in SEEDS]
    agg_rows.append({
        'model': tag, 'lr': sweep_summary[tag], 'n_seeds': len(SEEDS),
        'top1_mean': np.mean(top1s), 'top1_std': np.std(top1s),
        'top5_mean': np.mean(top5s), 'top5_std': np.std(top5s),
        'macro_f1_mean': np.mean(f1s), 'macro_f1_std': np.std(f1s),
        'selected_epochs': epochs_selected,
    })

matched_agg_df = pd.DataFrame(agg_rows)
print(matched_agg_df.to_string(index=False))
matched_agg_df.to_csv(os.path.join(OUT_DIR, 'nb8_matched_projection_seed_aggregate.csv'), index=False)

print("\nThese mean +/- std numbers, not any single seed's number, are what should go in the paper.")
print("If std is large relative to the gap between backbones, that gap isn't well-supported yet —")
print("check this before writing anything conclusive about which backbone 'wins'.")


# ===== cell ? =====
# ── Pick one representative checkpoint per backbone for the qualitative sections
# below (blind baselines, rich metrics, motion split). Using the seed whose
# selected-epoch Top-1 is closest to the mean keeps this checkpoint "typical"
# rather than cherry-picking the best-performing seed.
representative_seed = {}
representative_state = {}
for tag in BACKBONE_CONFIGS:
    top1s = {s: best_epoch_per_seed[tag][s][0]['val_top1'] for s in SEEDS}
    mean_top1 = np.mean(list(top1s.values()))
    rep_seed = min(top1s, key=lambda s: abs(top1s[s] - mean_top1))
    representative_seed[tag] = rep_seed
    representative_state[tag] = best_epoch_per_seed[tag][rep_seed][1]
    print(f"{tag}: representative seed = {rep_seed} (Top-1 {top1s[rep_seed]:.4f}, "
          f"mean across seeds {mean_top1:.4f})")


# ===== cell ? =====
def load_model_with_state(model_loader, in_dim, out_dim, state_dict):
    loaded = model_loader()
    model, tok = loaded if isinstance(loaded, tuple) else (loaded, tokenizer_v2)
    model.text_model.head.linear_projection = nn.Linear(in_dim, out_dim, bias=False).to(DEVICE)
    model.text_model.head.linear_projection.load_state_dict({k: v.to(DEVICE) for k, v in state_dict.items()})
    model.eval()
    return model, tok


@torch.no_grad()
def evaluate_blind(model, tok, df_subset, verb_vocab, label2id, mode='zero',
                    emb_dir=None, train_segment_ids=None):
    verb_tokens = tok.tokenize(verb_vocab).to(DEVICE)
    verb_embs = model.encode_text(verb_tokens, normalize=True)

    if mode == 'mean':
        embs = []
        for sid in train_segment_ids:
            p = os.path.join(emb_dir, f"{sid}.pt")
            if os.path.exists(p):
                embs.append(torch.load(p, map_location='cpu'))
        blind_emb = torch.stack(embs).mean(dim=0).to(DEVICE)
        blind_emb = F.normalize(blind_emb.unsqueeze(0), dim=-1).squeeze(0)
    else:
        blind_emb = torch.zeros(verb_embs.shape[-1], device=DEVICE)

    logit_scale = model.logit_scale.exp()
    sim = (logit_scale * blind_emb.unsqueeze(0) @ verb_embs.T).squeeze(0)
    top5_indices = sim.topk(5).indices.tolist()
    top1_idx = top5_indices[0]

    correct_top1 = correct_top5 = total = 0
    for _, row in df_subset.iterrows():
        gt_idx = label2id.get(row['extracted_verb'], -1)
        correct_top1 += (top1_idx == gt_idx)
        correct_top5 += (gt_idx in top5_indices)
        total += 1
    return correct_top1/total, correct_top5/total, [id2label[i] for i in top5_indices]


# ===== cell ? =====
blind_results = []
loaded_models = {}
for tag, cfg in BACKBONE_CONFIGS.items():
    model, tok = load_model_with_state(cfg['model_loader'], cfg['in_dim'], cfg['out_dim'],
                                        representative_state[tag])
    loaded_models[tag] = (model, tok)
    for mode in ['zero', 'mean']:
        t1, t5, preds = evaluate_blind(model, tok, df_val, verb_vocab, label2id, mode=mode,
                                        emb_dir=cfg['emb_dir'], train_segment_ids=df_train['segment_id'].tolist())
        # A blind model predicts one constant label for every segment, so its macro-F1
        # is the single-class F1 divided by n_verbs -- near zero however good Top-1 looks.
        blind_f1 = f1_score(df_val['extracted_verb'], [preds[0]] * len(df_val),
                            average='macro', zero_division=0)
        print(f"{tag:<8} blind[{mode:<4}] — Top-1: {t1*100:.2f}%  Top-5: {t5*100:.2f}%  "
              f"Macro-F1: {blind_f1*100:.2f}%  (always: {preds})")
        blind_results.append({'model': tag, 'mode': mode, 'top1': t1, 'top5': t5,
                              'macro_f1': blind_f1, 'n': len(df_val)})

pd.DataFrame(blind_results).to_csv(os.path.join(OUT_DIR, 'nb8_blind_baselines.csv'), index=False)
print(f"\nMajority-class baseline (always 'add'): {(df_val['extracted_verb']=='add').mean()*100:.2f}%")


# ===== cell ? =====
@torch.no_grad()
def eval_with_ranks(model, tok, emb_dir, df_subset, verb_vocab, label2id):
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
        rank = ranked.index(gt_idx) + 1 if gt_idx in ranked else len(verb_vocab) + 1
        # top1_pred added 2026-09-02: without the predicted label these CSVs support
        # rank-based metrics only, so macro-F1 could not be recomputed afterwards
        # without re-running the whole eval.
        records.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                         'top1_pred': verb_vocab[ranked[0]],
                         'rank': rank, 'reciprocal_rank': 1.0/rank,
                         'top1_correct': rank == 1, 'top5_correct': rank <= 5})
    return pd.DataFrame(records).set_index('segment_id')

ranks = {}
for tag, cfg in BACKBONE_CONFIGS.items():
    model, tok = loaded_models[tag]
    ranks[tag] = eval_with_ranks(model, tok, cfg['emb_dir'], df_val, verb_vocab, label2id)


# ===== cell ? =====
# ── Explicit common-segment-N check — do not skip this ─────────────────────────
ids_dinov2  = set(ranks['dinov2'].index)
ids_dinov3  = set(ranks['dinov3'].index)
ids_vjepa2  = set(ranks['vjepa2'].index)

common_all_three = ids_dinov2 & ids_dinov3 & ids_vjepa2
common_pairs = {
    ('dinov2', 'dinov3'): ids_dinov2 & ids_dinov3,
    ('dinov2', 'vjepa2'): ids_dinov2 & ids_vjepa2,
    ('dinov3', 'vjepa2'): ids_dinov3 & ids_vjepa2,
}

print(f"DINOv2 val coverage : {len(ids_dinov2):,}")
print(f"DINOv3 val coverage : {len(ids_dinov3):,}")
print(f"V-JEPA2 val coverage: {len(ids_vjepa2):,}")
print(f"Common to ALL THREE : {len(common_all_three):,}")
for pair, ids in common_pairs.items():
    print(f"  Common {pair[0]} & {pair[1]}: {len(ids):,}")

if len(common_all_three) < 500:
    print("\n*** WARNING: common-segment N is still small. The paired bootstrap CI below")
    print("*** will be wide and should be reported with that caveat, not as a tight result.")
else:
    print(f"\nCommon-segment N ({len(common_all_three):,}) is well above the old 134-segment bottleneck.")


# ===== cell ? =====
def summarize_metrics(ranks_df, name):
    mrr = ranks_df['reciprocal_rank'].mean()
    median_rank = ranks_df['rank'].median()
    micro_top1 = ranks_df['top1_correct'].mean()
    # macro_top1 is macro-averaged RECALL (mean per-class accuracy). It ignores
    # precision, so a model that over-predicts one class is not penalised for it.
    # macro_f1 is the number to put next to Top-1 in the paper.
    macro_top1 = ranks_df.groupby('ground_truth')['top1_correct'].mean().mean()
    macro_f1 = f1_score(ranks_df['ground_truth'], ranks_df['top1_pred'],
                        average='macro', zero_division=0)
    print(f"{name:<10} | Micro Top-1: {micro_top1*100:5.2f}% | Macro Top-1: {macro_top1*100:5.2f}% "
          f"| Macro-F1: {macro_f1*100:5.2f}% "
          f"| MRR: {mrr:.4f} | Median rank: {median_rank:.0f}/{n_verbs} | N={len(ranks_df):,}")
    return {'model': name, 'micro_top1': micro_top1, 'macro_top1': macro_top1,
            'macro_f1': macro_f1, 'mrr': mrr, 'median_rank': median_rank, 'n': len(ranks_df)}

metrics_summary = [summarize_metrics(ranks[tag], tag) for tag in BACKBONE_CONFIGS]
pd.DataFrame(metrics_summary).to_csv(os.path.join(OUT_DIR, 'nb8_rich_metrics.csv'), index=False)


# ===== cell ? =====
def paired_bootstrap(ranks_a, ranks_b, name_a, name_b, metric='top1_correct', n_boot=10000, seed=0):
    common = ranks_a.index.intersection(ranks_b.index)
    a = ranks_a.loc[common, metric].values.astype(float)
    b = ranks_b.loc[common, metric].values.astype(float)
    n = len(common)
    rng = np.random.default_rng(seed)
    obs_diff = a.mean() - b.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    print(f"{name_a} vs {name_b} on {metric} (N={n:,} common segments)")
    print(f"  Observed diff : {obs_diff*100:+.2f} pts")
    print(f"  95% CI        : [{lo*100:+.2f}, {hi*100:+.2f}] pts")
    print(f"  p-value       : {p_value:.4f}")
    return {'a': name_a, 'b': name_b, 'metric': metric, 'n': n,
            'obs_diff': obs_diff, 'ci_lo': lo, 'ci_hi': hi, 'p_value': p_value}

bootstrap_results = [
    paired_bootstrap(ranks['dinov3'], ranks['dinov2'], 'DINOv3', 'DINOv2'),
    paired_bootstrap(ranks['vjepa2'], ranks['dinov2'], 'V-JEPA2', 'DINOv2'),
    paired_bootstrap(ranks['vjepa2'], ranks['dinov3'], 'V-JEPA2', 'DINOv3'),
]
pd.DataFrame(bootstrap_results).to_csv(os.path.join(OUT_DIR, 'nb8_paired_bootstrap.csv'), index=False)


# ===== cell ? =====
MOTION_VERBS = {
    'stir', 'flip', 'whisk', 'knead', 'shake', 'toss', 'fold', 'mash', 'grind',
    'chop', 'slice', 'dice', 'mince', 'grate', 'peel', 'crush', 'pound', 'beat',
    'mix', 'blend', 'whip', 'roll', 'spread', 'squeeze', 'twist', 'rotate',
    'stretch', 'pull', 'push', 'scrape', 'scoop', 'drain', 'pour', 'drizzle',
    'sprinkle', 'shred', 'julienne', 'skewer', 'wrap', 'tie', 'crack',
    'rub', 'massage', 'flatten', 'press', 'squash', 'churn', 'saute',
}
STATIC_VERBS = {
    'add', 'place', 'put', 'remove', 'season', 'cover', 'serve', 'set', 'rest',
    'wait', 'preheat', 'heat', 'cook', 'bake', 'boil', 'simmer', 'fry', 'steam',
    'combine', 'insert', 'arrange', 'top', 'garnish', 'plate', 'store', 'chill',
    'freeze', 'refrigerate', 'discard', 'measure', 'weigh', 'check', 'taste',
    'turn', 'transfer', 'move', 'lower', 'raise', 'open', 'close', 'get', 'take',
}

def bucket_for(v):
    if v in MOTION_VERBS: return 'motion'
    if v in STATIC_VERBS: return 'static'
    return 'other'

verb_bucket_df = pd.DataFrame({'verb': verb_vocab})
verb_bucket_df['bucket_v1'] = verb_bucket_df['verb'].apply(bucket_for)
verb_bucket_df['rater2_bucket'] = ''   # to be filled in independently
verb_bucket_df['rater2_notes'] = ''

review_path = os.path.join(OUT_DIR, 'nb8_verb_motion_split_for_review.csv')
verb_bucket_df.to_csv(review_path, index=False)

print(f"Motion: {(verb_bucket_df['bucket_v1']=='motion').sum()}  "
      f"Static: {(verb_bucket_df['bucket_v1']=='static').sum()}  "
      f"Other: {(verb_bucket_df['bucket_v1']=='other').sum()}")
print(f"\nExported for independent review -> {review_path}")
print("Have a second person fill in 'rater2_bucket' (motion/static/other) WITHOUT")
print("seeing 'bucket_v1', then rerun the next cell.")


# ===== cell ? =====
# ── Run this once a second rater has filled in 'rater2_bucket' ─────────────────
reviewed = pd.read_csv(review_path)
n_rated = reviewed['rater2_bucket'].notna().sum()

if n_rated == 0:
    print("No second-rater labels yet — this cell is a no-op until 'rater2_bucket' is filled in.")
    KAPPA_OK = False
else:
    from collections import Counter
    def cohens_kappa(a, b, categories):
        n = len(a)
        po = sum(x == y for x, y in zip(a, b)) / n
        pe = sum((sum(1 for x in a if x == c)/n) * (sum(1 for y in b if y == c)/n) for c in categories)
        return (po - pe) / (1 - pe) if pe < 1 else 1.0

    cats = ['motion', 'static', 'other']
    kappa = cohens_kappa(reviewed['bucket_v1'].tolist(), reviewed['rater2_bucket'].tolist(), cats)
    print(f"Cohen's kappa (bucket_v1 vs rater2_bucket): {kappa:.3f}")
    print("  <0.4: poor agreement — do not use this split          "
          "0.4-0.6: moderate — use with caution and note it")
    print("  0.6-0.8: substantial — usable                          "
          ">0.8: near-perfect")
    KAPPA_OK = kappa >= 0.6
    if not KAPPA_OK:
        print("\n*** Agreement is below the 'substantial' threshold. Resolve disagreements")
        print("*** verb-by-verb before using this split as evidence for anything.")


# ===== cell ? =====
def motion_split_accuracy(ranks_df, name):
    df_r = ranks_df.copy()
    df_r['bucket'] = df_r['ground_truth'].apply(bucket_for)
    rows = []
    for bucket, grp in df_r.groupby('bucket'):
        if len(grp) == 0:
            continue
        rows.append({'model': name, 'bucket': bucket, 'n': len(grp),
                     'top1': grp['top1_correct'].mean(), 'top5': grp['top5_correct'].mean(),
                     'mrr': grp['reciprocal_rank'].mean()})
    return pd.DataFrame(rows)

motion_split_df = pd.concat([motion_split_accuracy(ranks[tag], tag) for tag in BACKBONE_CONFIGS], ignore_index=True)

status = "VERIFIED (kappa >= 0.6)" if 'KAPPA_OK' in dir() and KAPPA_OK else "PROVISIONAL — pending independent rater agreement"
print(f"=== Motion vs. static split — {status} ===")
print(motion_split_df.to_string(index=False))
motion_split_df.to_csv(os.path.join(OUT_DIR, 'nb8_motion_vs_static_split.csv'), index=False)

print("\nKey comparison: does V-JEPA2 show a larger motion-bucket lift over DINOv2/DINOv3")
print("relative to its own static-bucket performance, compared to how DINOv2 and DINOv3")
print("compare to each other in the same two buckets? That asymmetry — not the raw")
print("motion-bucket number alone — is what would support the temporal-inductive-bias claim.")


# ===== cell ? =====
summary_df = matched_agg_df.merge(
    pd.DataFrame(metrics_summary), on='model', how='left'
)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUT_DIR, 'nb8_final_summary.csv'), index=False)

print("\n" + "="*70)
print("All nb8_* files written to:", OUT_DIR)
print("="*70)
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('nb8_'):
        print(" ", f)


# ===== cell ? =====
spotcheck_path = os.path.join(PATHS['annotations'], 'spotcheck_sample.csv')
spotcheck = pd.read_csv(spotcheck_path)
n_filled = spotcheck['manual_correct'].notna().sum() if 'manual_correct' in spotcheck else 0
if n_filled == 0:
    print(f"'manual_correct' is empty in {spotcheck_path} — fill it in, then rerun this cell.")
else:
    precision = spotcheck['manual_correct'].mean()
    print(f"Verb extraction precision on {n_filled}-row spot-check: {precision*100:.1f}%")
    print(spotcheck.groupby('extraction_method')['manual_correct'].agg(['mean', 'count']))

print('--- NB8 script finished ---', flush=True)
