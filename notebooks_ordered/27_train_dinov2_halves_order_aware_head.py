"""
Train a DinoTXT projection head on the order-aware DINOv2 pooling (script 26's
halves-pool features) using the EXACT SAME protocol as script 07/13: same
architecture (a single frozen-backbone linear_projection on the text side),
same LR sweep (3 LRs x 2 sweep epochs), same N_SEEDS=2 (matching the
2026-09-08 decision to skip the 5-seed rerun elsewhere), same NUM_EPOCHS=10,
same epoch-selection rule, same batch size/grad accum.

This is the experiment REPRODUCE.md's Phase 6 called for and never ran: does
giving DINOv2 order-aware pooling (rather than the order-blind mean it was
published with) change verb-classification accuracy at all? Script 25 already
answered the narrower question (raw forward/reversed discriminability); this
answers the one that actually matters for the paper's classification headline
and sets up script 28's Claim-B extension.

One asymmetry from script 07, stated plainly: this projection has out_dim=4096
(the halves-pool feature dim) vs. DINOv2 mean-pool's published out_dim=2048 --
double the trainable parameters in the projection matrix, since halves-pool is
twice mean-pool's width by construction (two 2048-dim chunks concatenated).
Not parameter-matched to the mean-pool baseline; report this alongside the
result rather than calling it a matched comparison.

Reads : embeddings/<...>/dinov2_perframe_halves/<segment_id>.pt (script 26)
        annotations/segments_verbs_timed_clean.csv
Writes: checkpoints/nb8_dinov2halves_seed<s>_bestepoch<e>.pt
        processed/nb8_dinov2halves_full_history.csv
        processed/nb8_dinov2halves_seed_aggregate.csv
        processed/nb8_lr_sweep_dinov2halves.csv
"""
import os, sys, json, math, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

OUT_DIR = PATHS['processed']
CKPT_DIR = PATHS['checkpoints']
EMB_DIR = os.path.join(PATHS['embeddings'], 'dinov2_perframe_halves')
IN_DIM, OUT_DIM = 1280, 4096
TAG = 'dinov2halves'

DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

tokenizer_v2 = get_tokenizer()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fresh_dinov2_dinotxt():
    m = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
n_verbs = len(verb_vocab)
assert n_verbs == 282
df['verb_label'] = df['extracted_verb'].map(label2id)
df_train = df[df['subset'] == 'training'].reset_index(drop=True)
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)

n_avail = len([f for f in os.listdir(EMB_DIR) if f.endswith('.pt')])
print(f"Halves-pool embeddings: {n_avail:,} at {EMB_DIR}")

CANDIDATE_LRS = [3e-5, 1e-4, 3e-4]
SWEEP_EPOCHS = 2
N_SEEDS = 2
NUM_EPOCHS = 10
BATCH_SIZE = 32
GRAD_ACCUM = 4
SEEDS = list(range(N_SEEDS))
print(f"LR grid: {CANDIDATE_LRS}  |  sweep epochs: {SWEEP_EPOCHS}  |  seeds: {SEEDS}  |  "
      f"full epochs: {NUM_EPOCHS}")


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


def train_one_run(lr, num_epochs, seed, tag, verbose=True):
    set_seed(seed)
    model = load_fresh_dinov2_dinotxt()
    tok = tokenizer_v2
    model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = 'text_model.head.linear_projection' in name

    g = torch.Generator().manual_seed(seed)
    train_ds = ProjDataset(df_train, EMB_DIR, tag=tag)
    val_ds = ProjDataset(df_val, EMB_DIR, tag=tag)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                                collate_fn=collate, num_workers=0,
                                                pin_memory=True, generator=g)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
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
        y_pred = torch.cat(all_pred).numpy(); y_true = torch.cat(all_true).numpy()
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return c1 / tot, c5 / tot, macro_f1

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
        iterator = tqdm(train_loader, desc=f"[{tag} seed{seed} lr{lr:.0e}] epoch {epoch}/{num_epochs}") \
            if verbose else train_loader
        for step, (embs, labels) in enumerate(iterator, start=1):
            embs = F.normalize(embs.to(DEVICE), dim=-1)
            labels = labels.to(DEVICE)
            verb_embs = model.encode_text(verb_tokens_bare, normalize=True)
            logit_scale = model.logit_scale.exp()
            sim = logit_scale * embs @ verb_embs.T
            loss = F.cross_entropy(sim, labels) / GRAD_ACCUM
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM
            if step % GRAD_ACCUM == 0:
                optimizer.step(); optimizer.zero_grad()
                with torch.no_grad():
                    model.logit_scale.clamp_(0, math.log(100))
        if step % GRAD_ACCUM != 0:
            optimizer.step(); optimizer.zero_grad()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
        scheduler.step()

        top1, top5, macro_f1 = evaluate()
        avg_loss = running_loss / len(train_loader)
        if verbose:
            print(f"  [{tag} seed{seed}] epoch {epoch}/{num_epochs} | loss {avg_loss:.4f} "
                  f"| Val Top-1 {top1:.4f} | Val Top-5 {top5:.4f} | Val macro-F1 {macro_f1:.4f}", flush=True)
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
    h = history_df.sort_values(['val_top1', 'val_top5'], ascending=False)
    return h.iloc[0]


# ── LR sweep ──────────────────────────────────────────────────────────────────
print(f"\n=== LR sweep: {TAG} ===", flush=True)
sweep_results = []
for lr in CANDIDATE_LRS:
    hist, _ = train_one_run(lr, num_epochs=SWEEP_EPOCHS, seed=0, tag=f"{TAG}_sweep", verbose=False)
    best = select_best_epoch(hist)
    print(f"  lr={lr:.0e} -> best Val Top-1 after {SWEEP_EPOCHS} epochs: {best['val_top1']:.4f}", flush=True)
    sweep_results.append({'lr': lr, 'val_top1': best['val_top1'], 'val_top5': best['val_top5']})
sweep_df = pd.DataFrame(sweep_results).sort_values('val_top1', ascending=False)
best_lr = sweep_df.iloc[0]['lr']
print(f"  -> chosen LR: {best_lr:.0e}", flush=True)
sweep_df.to_csv(os.path.join(OUT_DIR, f'nb8_lr_sweep_{TAG}.csv'), index=False)

# ── Full multi-seed training ────────────────────────────────────────────────
all_histories = []
best_per_seed = {}
for seed in SEEDS:
    hist, state_dicts = train_one_run(best_lr, num_epochs=NUM_EPOCHS, seed=seed, tag=TAG)
    all_histories.append(hist)
    best_row = select_best_epoch(hist)
    best_per_seed[seed] = (best_row, state_dicts[int(best_row['epoch'])])
    print(f"[{TAG} seed {seed}] selected epoch {int(best_row['epoch'])} -- "
          f"Val Top-1 {best_row['val_top1']:.4f}, Val Top-5 {best_row['val_top5']:.4f}, "
          f"macro-F1 {best_row['val_macro_f1']:.4f}", flush=True)
    ckpt_path = os.path.join(CKPT_DIR, f'nb8_{TAG}_seed{seed}_bestepoch{int(best_row["epoch"])}.pt')
    torch.save(state_dicts[int(best_row['epoch'])], ckpt_path)

full_history_df = pd.concat(all_histories, ignore_index=True)
full_history_df.to_csv(os.path.join(OUT_DIR, f'nb8_{TAG}_full_history.csv'), index=False)

top1s = [best_per_seed[s][0]['val_top1'] for s in SEEDS]
top5s = [best_per_seed[s][0]['val_top5'] for s in SEEDS]
f1s = [best_per_seed[s][0]['val_macro_f1'] for s in SEEDS]
agg = pd.DataFrame([{
    'model': TAG, 'lr': best_lr, 'n_seeds': len(SEEDS),
    'top1_mean': np.mean(top1s), 'top1_std': np.std(top1s),
    'top5_mean': np.mean(top5s), 'top5_std': np.std(top5s),
    'macro_f1_mean': np.mean(f1s), 'macro_f1_std': np.std(f1s),
    'selected_epochs': [int(best_per_seed[s][0]['epoch']) for s in SEEDS],
}])
print("\n" + agg.to_string(index=False))
agg.to_csv(os.path.join(OUT_DIR, f'nb8_{TAG}_seed_aggregate.csv'), index=False)

print(f"\nFor comparison -- published mean-pool DINOv2: Top-1 42.96%, macro-F1 5.37%")
print(f"For comparison -- V-JEPA2 pretrained-pool: Top-1 43.62%, macro-F1 5.39%")
print("\nDONE")
