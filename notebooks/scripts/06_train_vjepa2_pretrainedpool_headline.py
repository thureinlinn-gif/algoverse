"""
Full matched-projection training + evaluation for V-JEPA2's frozen
PRETRAINED-pooler embeddings (validated by POC: +3.33pp Top-1 over naive
mean-pool, same checkpoint, same trainable-param budget). Same recipe as
every other backbone: LR sweep, N_SEEDS=5, NUM_EPOCHS=10, same epoch-selection
rule -- directly comparable to DINOv2/DINOv3/vjepa2base's existing numbers.
"""
import os, sys, json, math, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

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
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fresh_dinov2_dinotxt():
    m = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


tokenizer_v2 = get_tokenizer()

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
id2label = {i: v for v, i in label2id.items()}
n_verbs = len(verb_vocab)
df['verb_label'] = df['extracted_verb'].map(label2id)
df_train = df[df['subset'] == 'training'].reset_index(drop=True)
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
assert n_verbs == 282

EMB_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool')
n_avail = len([f for f in os.listdir(EMB_DIR) if f.endswith('.pt')])
print(f"Pretrained-pool embeddings: {n_avail:,} at {EMB_DIR}", flush=True)

IN_DIM, OUT_DIM = 1280, 1024


class ProjDataset(torch.utils.data.Dataset):
    def __init__(self, df, emb_dir):
        self.records = []
        for _, row in df.iterrows():
            p = os.path.join(emb_dir, f"{row['segment_id']}.pt")
            if os.path.exists(p):
                self.records.append((p, int(row['verb_label'])))
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        p, label = self.records[idx]
        return torch.load(p, map_location='cpu'), label


def collate(batch):
    embs, labels = zip(*batch)
    return torch.stack(embs), torch.tensor(labels, dtype=torch.long)


def train_one_run(lr, num_epochs, seed, tag, batch_size=32, grad_accum=4, verbose=True):
    set_seed(seed)
    model = load_fresh_dinov2_dinotxt()
    model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = 'text_model.head.linear_projection' in name

    g = torch.Generator().manual_seed(seed)
    train_ds = ProjDataset(df_train, EMB_DIR)
    val_ds = ProjDataset(df_val, EMB_DIR)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                collate_fn=collate, num_workers=0, pin_memory=True, generator=g)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate, num_workers=0, pin_memory=True)
    verb_tokens_bare = tokenizer_v2.tokenize(verb_vocab).to(DEVICE)

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
        # See script 01: 'add' is 35.65% of val, so Top-1 alone flatters everything.
        macro_f1 = f1_score(torch.cat(all_true).numpy(), torch.cat(all_pred).numpy(),
                            average='macro', zero_division=0)
        return c1 / tot, c5 / tot, macro_f1

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
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
        print(f"  [{tag} seed{seed}] epoch {epoch}/{num_epochs} | loss {avg_loss:.4f} "
              f"| Val Top-1 {top1:.4f} | Val Top-5 {top5:.4f}", flush=True)
        history.append({'tag': tag, 'seed': seed, 'lr': lr, 'epoch': epoch,
                         'loss': avg_loss, 'val_top1': top1, 'val_top5': top5,
                         'val_macro_f1': macro_f1})
        state_dicts[epoch] = {k: v.clone().cpu() for k, v in
                               model.text_model.head.linear_projection.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(history), state_dicts


def select_best_epoch(history_df):
    return history_df.sort_values(['val_top1', 'val_top5'], ascending=False).iloc[0]


CANDIDATE_LRS = [3e-5, 1e-4, 3e-4]
SWEEP_EPOCHS = 2
N_SEEDS = 5        # raised 2 -> 5 on 2026-09-02, matching script 01
NUM_EPOCHS = 10
SEEDS = list(range(N_SEEDS))
TAG = 'vjepa2_pretrainedpool'

print(f"\n=== LR sweep: {TAG} ===", flush=True)
sweep_results = []
for lr in CANDIDATE_LRS:
    hist, _ = train_one_run(lr, SWEEP_EPOCHS, seed=0, tag=f"{TAG}_sweep", verbose=False)
    best = select_best_epoch(hist)
    print(f"  lr={lr:.0e} -> best Val Top-1 after {SWEEP_EPOCHS} epochs: {best['val_top1']:.4f}", flush=True)
    sweep_results.append({'lr': lr, 'val_top1': best['val_top1'], 'val_top5': best['val_top5']})
sweep_df = pd.DataFrame(sweep_results).sort_values('val_top1', ascending=False)
best_lr = sweep_df.iloc[0]['lr']
print(f"  -> chosen LR for {TAG}: {best_lr:.0e}", flush=True)
sweep_df.to_csv(os.path.join(OUT_DIR, f'nb8_lr_sweep_{TAG}.csv'), index=False)

all_histories = []
best_epoch_per_seed = {}
for seed in SEEDS:
    hist, state_dicts = train_one_run(best_lr, NUM_EPOCHS, seed=seed, tag=TAG)
    all_histories.append(hist)
    best_row = select_best_epoch(hist)
    best_epoch_per_seed[seed] = (best_row, state_dicts[int(best_row['epoch'])])
    print(f"[{TAG} seed {seed}] selected epoch {int(best_row['epoch'])} "
          f"— Val Top-1 {best_row['val_top1']:.4f}, Val Top-5 {best_row['val_top5']:.4f}", flush=True)
    ckpt_path = os.path.join(CKPT_DIR, f'nb8_{TAG}_seed{seed}_bestepoch{int(best_row["epoch"])}.pt')
    torch.save(state_dicts[int(best_row['epoch'])], ckpt_path)

full_history_df = pd.concat(all_histories, ignore_index=True)
full_history_df.to_csv(os.path.join(OUT_DIR, f'nb8_{TAG}_full_history.csv'), index=False)

top1s = [best_epoch_per_seed[s][0]['val_top1'] for s in SEEDS]
top5s = [best_epoch_per_seed[s][0]['val_top5'] for s in SEEDS]
f1s   = [best_epoch_per_seed[s][0]['val_macro_f1'] for s in SEEDS]
print(f"\n{TAG}: top1_mean={np.mean(top1s):.4f} top1_std={np.std(top1s):.4f} "
      f"top5_mean={np.mean(top5s):.4f} top5_std={np.std(top5s):.4f}", flush=True)
pd.DataFrame([{
    'model': TAG, 'lr': best_lr, 'n_seeds': len(SEEDS),
    'top1_mean': np.mean(top1s), 'top1_std': np.std(top1s),
    'top5_mean': np.mean(top5s), 'top5_std': np.std(top5s),
    'macro_f1_mean': np.mean(f1s), 'macro_f1_std': np.std(f1s),
    'selected_epochs': [int(best_epoch_per_seed[s][0]['epoch']) for s in SEEDS],
}]).to_csv(os.path.join(OUT_DIR, f'nb8_{TAG}_seed_aggregate.csv'), index=False)

mean_top1 = np.mean(top1s)
rep_seed = min(SEEDS, key=lambda s: abs(best_epoch_per_seed[s][0]['val_top1'] - mean_top1))
rep_state = best_epoch_per_seed[rep_seed][1]
print(f"{TAG}: representative seed = {rep_seed}", flush=True)

model = load_fresh_dinov2_dinotxt()
model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
model.text_model.head.linear_projection.load_state_dict({k: v.to(DEVICE) for k, v in rep_state.items()})
model.eval()

verb_tokens = tokenizer_v2.tokenize(verb_vocab).to(DEVICE)
with torch.no_grad():
    verb_embs = model.encode_text(verb_tokens, normalize=True)

blind_results = []
for mode in ['zero', 'mean']:
    if mode == 'mean':
        embs = []
        for sid in df_train['segment_id'].tolist():
            p = os.path.join(EMB_DIR, f"{sid}.pt")
            if os.path.exists(p):
                embs.append(torch.load(p, map_location='cpu'))
        blind_emb = F.normalize(torch.stack(embs).mean(dim=0).unsqueeze(0), dim=-1).squeeze(0).to(DEVICE)
    else:
        blind_emb = torch.zeros(verb_embs.shape[-1], device=DEVICE)
    with torch.no_grad():
        sim = (model.logit_scale.exp() * blind_emb.unsqueeze(0) @ verb_embs.T).squeeze(0)
    top5_idx = sim.topk(5).indices.tolist()
    top1_idx = top5_idx[0]
    c1 = c5 = tot = 0
    for _, row in df_val.iterrows():
        gt = label2id.get(row['extracted_verb'], -1)
        c1 += (top1_idx == gt); c5 += (gt in top5_idx); tot += 1
    blind_f1 = f1_score(df_val['extracted_verb'], [id2label[top1_idx]] * len(df_val),
                        average='macro', zero_division=0)
    print(f"{TAG} blind[{mode}] — Top-1: {c1/tot*100:.2f}%  Top-5: {c5/tot*100:.2f}%  "
          f"Macro-F1: {blind_f1*100:.2f}%  (always: {[id2label[i] for i in top5_idx]})", flush=True)
    blind_results.append({'model': TAG, 'mode': mode, 'top1': c1/tot, 'top5': c5/tot,
                          'macro_f1': blind_f1, 'n': tot})
pd.DataFrame(blind_results).to_csv(os.path.join(OUT_DIR, f'nb8_blind_baselines_{TAG}.csv'), index=False)

records = []
with torch.no_grad():
    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Ranking"):
        emb_path = os.path.join(EMB_DIR, f"{row['segment_id']}.pt")
        if not os.path.exists(emb_path):
            continue
        video_emb = F.normalize(torch.load(emb_path, map_location=DEVICE).unsqueeze(0), dim=-1)
        sim = (model.logit_scale.exp() * video_emb @ verb_embs.T).squeeze(0).cpu()
        ranked = sim.argsort(descending=True).tolist()
        gt_idx = label2id.get(row['extracted_verb'], -1)
        rank = ranked.index(gt_idx) + 1 if gt_idx in ranked else n_verbs + 1
        records.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb'],
                         'top1_pred': verb_vocab[ranked[0]],
                         'rank': rank, 'reciprocal_rank': 1.0/rank,
                         'top1_correct': rank == 1, 'top5_correct': rank <= 5})
ranks_pp = pd.DataFrame(records).set_index('segment_id')
ranks_pp.to_csv(os.path.join(OUT_DIR, f'nb8_ranks_{TAG}_representative.csv'))
print(f"\n{TAG} full-val — N={len(ranks_pp):,} Top-1={ranks_pp['top1_correct'].mean()*100:.2f}% "
      f"Top-5={ranks_pp['top5_correct'].mean()*100:.2f}% "
      f"Macro-F1={f1_score(ranks_pp['ground_truth'], ranks_pp['top1_pred'], average='macro', zero_division=0)*100:.2f}% "
      f"MRR={ranks_pp['reciprocal_rank'].mean():.4f}", flush=True)

del model
torch.cuda.empty_cache()

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
    print(f"{name_a} vs {name_b} on {metric} (N={n:,})  diff={obs_diff*100:+.2f}pp  "
          f"CI=[{lo*100:+.2f},{hi*100:+.2f}]  p={p_value:.4f}", flush=True)
    return {'a': name_a, 'b': name_b, 'metric': metric, 'n': n, 'obs_diff': obs_diff,
            'ci_lo': lo, 'ci_hi': hi, 'p_value': p_value}

ranks_dinov2 = pd.read_csv(os.path.join(OUT_DIR, 'nb8_ranks_dinov2_representative.csv')).set_index('segment_id')
ranks_dinov3 = pd.read_csv(os.path.join(OUT_DIR, 'nb8_ranks_dinov3_representative.csv')).set_index('segment_id')
ranks_vjepa2base = pd.read_csv(os.path.join(OUT_DIR, 'nb8_ranks_vjepa2base_representative.csv')).set_index('segment_id')

bootstrap_results = [
    paired_bootstrap(ranks_pp, ranks_vjepa2base, TAG, 'vjepa2base (mean-pool)'),
    paired_bootstrap(ranks_pp, ranks_dinov2, TAG, 'DINOv2'),
    paired_bootstrap(ranks_pp, ranks_dinov3, TAG, 'DINOv3'),
]
pd.DataFrame(bootstrap_results).to_csv(os.path.join(OUT_DIR, f'nb8_paired_bootstrap_{TAG}.csv'), index=False)

# frequent vs rare
verb_counts = df_train['extracted_verb'].value_counts()
frequent_verbs = set(verb_counts[verb_counts >= 50].index)
rare_verbs = set(verb_counts[verb_counts < 10].index)
def freq_bucket(v):
    if v in frequent_verbs: return 'frequent (>=50 train)'
    if v in rare_verbs: return 'rare (<10 train)'
    return 'mid'
df_r = ranks_pp.copy()
df_r['bucket'] = df_r['ground_truth'].apply(freq_bucket)
rows = []
for bucket, grp in df_r.groupby('bucket'):
    rows.append({'model': TAG, 'bucket': bucket, 'n': len(grp),
                 'micro_top1': grp['top1_correct'].mean(), 'micro_top5': grp['top5_correct'].mean(),
                 'macro_top1': grp.groupby('ground_truth')['top1_correct'].mean().mean(),
                 'mrr': grp['reciprocal_rank'].mean()})
freq_df = pd.DataFrame(rows)
print(f"\n=== {TAG} frequent vs rare ===\n" + freq_df.to_string(index=False), flush=True)
freq_df.to_csv(os.path.join(OUT_DIR, f'nb8_frequent_vs_rare_{TAG}.csv'), index=False)

# optical-flow tercile
flow_df = pd.read_csv(os.path.join(OUT_DIR, 'nb8_optical_flow_scores.csv'))
flow_df['tercile'] = pd.qcut(flow_df['motion_score'], 3, labels=['low_motion', 'mid_motion', 'high_motion'])
flow_df = flow_df.set_index('segment_id')
merged = ranks_pp.join(flow_df[['tercile']], how='inner')
rows = []
for tercile, grp in merged.groupby('tercile', observed=True):
    rows.append({'model': TAG, 'tercile': tercile, 'n': len(grp),
                 'micro_top1': grp['top1_correct'].mean(), 'micro_top5': grp['top5_correct'].mean(),
                 'macro_top1': grp.groupby('ground_truth')['top1_correct'].mean().mean(),
                 'mrr': grp['reciprocal_rank'].mean()})
flow_result_df = pd.DataFrame(rows)
print(f"\n=== {TAG} by optical-flow tercile ===\n" + flow_result_df.to_string(index=False), flush=True)
flow_result_df.to_csv(os.path.join(OUT_DIR, f'nb8_optical_flow_tercile_{TAG}.csv'), index=False)

print("\nDONE", flush=True)
