"""
Phase 7, shared library: script 07's matched-projection recipe and evaluation
helpers, rebuilt on script 32's cached text features so a head trains in seconds
without a text-tower forward. Imported by scripts 34-36; running it directly runs
the reproduction gate below, which must pass before any Phase 7 number is trusted.

The recipe, unchanged from script 07 (train_one_run / lr_sweep / select_best_epoch):
bias-free Linear(1280 -> out_dim) on the text side; CE over
logit_scale * norm(v) @ norm(F W^T)^T with logit_scale frozen at 1/0.07; AdamW
(betas 0.9/0.99, eps 1e-8, wd 0); cosine schedule stepped per epoch; batch 32 x
grad-accum 4; LR sweep {3e-5, 1e-4, 3e-4} for 2 epochs at seed 0; 10 epochs x 2
seeds at the chosen LR; best epoch by val Top-1 (ties: Top-5); representative seed
= closest to the mean. Not bit-identical to script 07 -- 07 builds the full DinoTXT
model between set_seed and nn.Linear, which consumes RNG -- hence the gate
tolerance below rather than an exact match.

Gate (python3 33_linear_head_lib.py):
  1. Scoring path: each existing representative checkpoint, scored through the
     cached features, reproduces its published ranks CSV on >99.9% of segments
     (same rule as script 21).
  2. Training path: retraining DINOv2 and V-JEPA2-pretrainedpool with this module
     lands within 0.3pp of the published 2-seed mean Top-1 (42.96 / 43.62).

Reads : processed/nb8_text_features_*.pt (script 32), embeddings, checkpoints,
        processed/nb8_ranks_*_representative.csv, the *_full_history.csv files
Writes: processed/nb8_phase7_gate.csv (gate only)
"""
import os, sys, json, math, random
from concurrent.futures import ThreadPoolExecutor
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

OUT_DIR, CKPT_DIR, EMB = PATHS['processed'], PATHS['checkpoints'], PATHS['embeddings']

CANDIDATE_LRS = [3e-5, 1e-4, 3e-4]
SWEEP_EPOCHS = 2
N_SEEDS = 2
NUM_EPOCHS = 10
BATCH_SIZE = 32
GRAD_ACCUM = 4
GATE_TOL = 0.003


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data ─────────────────────────────────────────────────────────────────────
def load_segments():
    df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
    verb_vocab = sorted(df['extracted_verb'].unique().tolist())
    assert len(verb_vocab) == 282, f"Expected 282 verbs, got {len(verb_vocab)}"
    label2id = {v: i for i, v in enumerate(verb_vocab)}
    df['verb_label'] = df['extracted_verb'].map(label2id)
    return df, verb_vocab, label2id


def load_text_features(name='dinov2_dinotxt'):
    """(282, 1280) pre-projection verb features + logit scale, or None if script 32
    has not cached this tower (DINOv3 without its weights)."""
    path = os.path.join(OUT_DIR, f'nb8_text_features_{name}.pt')
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location='cpu')
    return d['features'].to(DEVICE), d['logit_scale']


def load_embeddings(emb_dir, segment_ids, transform=None):
    """Stack <emb_dir>/<segment_id>.pt for every id present. Returns (tensor, kept_ids).
    Threaded: per-file torch.load is NFS-latency-bound, not CPU-bound."""
    paths = [(sid, os.path.join(emb_dir, f"{sid}.pt")) for sid in segment_ids]
    paths = [(sid, p) for sid, p in paths if os.path.exists(p)]

    def _load(p):
        x = torch.load(p, map_location='cpu')
        return transform(x) if transform else x
    with ThreadPoolExecutor(16) as ex:
        tensors = list(ex.map(_load, [p for _, p in paths]))
    missing = len(segment_ids) - len(paths)
    if missing:
        print(f"  [{os.path.basename(emb_dir)}] {missing} segments missing embeddings, excluded")
    return torch.stack(tensors), [sid for sid, _ in paths]


# ── Scoring ──────────────────────────────────────────────────────────────────
def verb_text_embs(W, text_feats):
    return F.normalize(text_feats @ W.T, dim=-1)


@torch.no_grad()
def score_matrix(X, W, text_feats, logit_scale, batch=4096):
    """(N, 282) logits for video vectors X under text-side head W -- exactly what
    script 07's eval computes, without the text-tower forward."""
    T = verb_text_embs(W.to(DEVICE), text_feats)
    out = [logit_scale * F.normalize(X[i:i + batch].to(DEVICE), dim=-1) @ T.T
           for i in range(0, len(X), batch)]
    return torch.cat(out).cpu()


def top1_top5_f1(scores, y):
    y = torch.as_tensor(y)
    pred = scores.argmax(-1)
    top5 = scores.topk(5, dim=-1).indices
    return ((pred == y).float().mean().item(),
            (top5 == y.unsqueeze(-1)).any(-1).float().mean().item(),
            f1_score(y.numpy(), pred.numpy(), average='macro', zero_division=0))


# ── Training (script 07's recipe) ────────────────────────────────────────────
def train_one_run(Xtr, ytr, Xva, yva, text_feats, logit_scale, out_dim, lr, num_epochs, seed):
    set_seed(seed)
    proj = nn.Linear(text_feats.shape[1], out_dim, bias=False).to(DEVICE)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    g = torch.Generator().manual_seed(seed)
    Xtr_n = F.normalize(Xtr.to(DEVICE), dim=-1)
    ytr_d = torch.as_tensor(ytr, device=DEVICE)
    n = len(Xtr_n)
    history, state_dicts = [], {}
    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(n, generator=g).to(DEVICE)
        opt.zero_grad()
        running, n_batches = 0.0, 0
        for step, i in enumerate(range(0, n, BATCH_SIZE), start=1):
            idx = perm[i:i + BATCH_SIZE]
            sim = logit_scale * Xtr_n[idx] @ verb_text_embs(proj.weight, text_feats).T
            loss = F.cross_entropy(sim, ytr_d[idx]) / GRAD_ACCUM
            loss.backward()
            running += loss.item() * GRAD_ACCUM
            n_batches += 1
            if step % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
        if step % GRAD_ACCUM != 0:
            opt.step(); opt.zero_grad()
        sched.step()
        top1, top5, mf1 = top1_top5_f1(score_matrix(Xva, proj.weight.detach(), text_feats, logit_scale), yva)
        history.append({'seed': seed, 'lr': lr, 'epoch': epoch, 'loss': running / n_batches,
                        'val_top1': top1, 'val_top5': top5, 'val_macro_f1': mf1})
        state_dicts[epoch] = proj.weight.detach().clone().cpu()
    return pd.DataFrame(history), state_dicts


def select_best_epoch(history_df):
    """Script 07's rule: best validation Top-1, ties broken by Top-5."""
    return history_df.sort_values(['val_top1', 'val_top5'], ascending=False).iloc[0]


def fit_protocol(Xtr, ytr, Xva, yva, text_feats, logit_scale, out_dim, tag='', verbose=True):
    """LR sweep + N_SEEDS full runs + representative seed, as in script 07.
    Returns dict(lr, history, per_seed=[(best_row, W)], rep_W, top1_mean, top1_std)."""
    sweep = []
    for lr in CANDIDATE_LRS:
        h, _ = train_one_run(Xtr, ytr, Xva, yva, text_feats, logit_scale, out_dim, lr, SWEEP_EPOCHS, 0)
        sweep.append((select_best_epoch(h)['val_top1'], lr))
    lr = max(sweep)[1]
    hists, per_seed = [], []
    for seed in range(N_SEEDS):
        h, sds = train_one_run(Xtr, ytr, Xva, yva, text_feats, logit_scale, out_dim, lr, NUM_EPOCHS, seed)
        best = select_best_epoch(h)
        hists.append(h.assign(tag=tag))
        per_seed.append((best, sds[int(best['epoch'])]))
    top1s = [b['val_top1'] for b, _ in per_seed]
    rep = min(range(N_SEEDS), key=lambda s: abs(top1s[s] - np.mean(top1s)))
    if verbose:
        print(f"  [{tag}] lr={lr:.0e}  Top-1 per seed {[round(t, 4) for t in top1s]}  "
              f"mean {np.mean(top1s):.4f}  (rep seed {rep})")
    return {'lr': lr, 'history': pd.concat(hists, ignore_index=True), 'per_seed': per_seed,
            'rep_W': per_seed[rep][1], 'top1_mean': float(np.mean(top1s)), 'top1_std': float(np.std(top1s))}


# ── Evaluation helpers (script 07: eval_with_ranks / summarize_metrics / paired_bootstrap) ──
def eval_with_ranks(scores, segment_ids, y_true, verb_vocab):
    y = torch.as_tensor(y_true)
    true_score = scores.gather(1, y.unsqueeze(1))
    rank = (scores > true_score).sum(1) + 1
    pred = scores.argmax(-1)
    return pd.DataFrame({'segment_id': segment_ids,
                         'ground_truth': [verb_vocab[i] for i in y.tolist()],
                         'top1_pred': [verb_vocab[i] for i in pred.tolist()],
                         'rank': rank.numpy(), 'reciprocal_rank': 1.0 / rank.numpy(),
                         'top1_correct': (rank == 1).numpy(), 'top5_correct': (rank <= 5).numpy(),
                         }).set_index('segment_id')


def summarize_metrics(ranks_df, name):
    micro_top1 = ranks_df['top1_correct'].mean()
    macro_top1 = ranks_df.groupby('ground_truth')['top1_correct'].mean().mean()
    macro_f1 = f1_score(ranks_df['ground_truth'], ranks_df['top1_pred'], average='macro', zero_division=0)
    return {'model': name, 'micro_top1': micro_top1, 'macro_top1': macro_top1, 'macro_f1': macro_f1,
            'mrr': ranks_df['reciprocal_rank'].mean(), 'median_rank': ranks_df['rank'].median(),
            'n': len(ranks_df)}


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Segment-level bootstrap of mean(a) - mean(b) over aligned per-segment values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.array([(a[idx] - b[idx]).mean() for idx in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'obs_diff': (a - b).mean(), 'ci_lo': lo, 'ci_hi': hi,
            'p_value': 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())}


def cluster_bootstrap(values, video_ids, n_boot=10000, seed=0):
    """Video-cluster bootstrap of a per-segment mean (script 24/28's correction):
    resample videos, not segments, so segments of one video stay together."""
    vstats = pd.DataFrame({'v': np.asarray(values, float), 'vid': list(video_ids)}) \
        .groupby('vid')['v'].agg(['count', 'sum'])
    n_a, sum_a = vstats['count'].to_numpy(), vstats['sum'].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(vstats), size=(n_boot, len(vstats)))
    boot = sum_a[idx].sum(axis=1) / n_a[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'mean': sum_a.sum() / n_a.sum(), 'ci_lo': lo, 'ci_hi': hi,
            'p_value': 2 * min((boot <= 0).mean(), (boot >= 0).mean()), 'n_videos': len(vstats)}


# ── Existing representative heads (scripts 07 / 11 / 13 / 27) ───────────────
HEADS = {
    'dinov2': dict(emb_dir=EMB, out_dim=2048, text='dinov2_dinotxt',
                   history='nb8_matched_projection_full_history.csv',
                   published='nb8_ranks_dinov2_representative.csv'),
    'dinov3': dict(emb_dir=os.path.join(EMB, 'dinov3'), out_dim=2048, text='dinov3_dinotxt',
                   history='nb8_matched_projection_full_history.csv',
                   published='nb8_ranks_dinov3_representative.csv'),
    'dinov2halves': dict(emb_dir=os.path.join(EMB, 'dinov2_perframe_halves'), out_dim=4096,
                         text='dinov2_dinotxt', history='nb8_dinov2halves_full_history.csv',
                         published=None,   # no ranks CSV was written; script 28 fixes seed0/epoch10
                         fixed_ckpt='nb8_dinov2halves_seed0_bestepoch10.pt'),
    'vjepa2base': dict(emb_dir=os.path.join(EMB, 'vjepa2_base_dinotxt'), out_dim=1024,
                       text='dinov2_dinotxt', history='nb8_vjepa2base_full_history.csv',
                       published='nb8_ranks_vjepa2base_representative.csv'),
    'vjepa2_pretrainedpool': dict(emb_dir=os.path.join(EMB, 'vjepa2_pretrainedpool'), out_dim=1024,
                                  text='dinov2_dinotxt', history='nb8_vjepa2_pretrainedpool_full_history.csv',
                                  published='nb8_ranks_vjepa2_pretrainedpool_representative.csv'),
}
HEAD_LABELS = {
    'dinov2': 'DINOv2 (mean-pool)', 'dinov3': 'DINOv3 (mean-pool)',
    'dinov2halves': 'DINOv2 (order-aware halves)',
    'vjepa2base': 'V-JEPA2 label-free (mean-pool)',
    'vjepa2_pretrainedpool': 'V-JEPA2 SSv2-pooler (headline)',
}


def load_representative_head(tag, df_val, verb_vocab, verbose=True):
    """Returns (W, X_val, val_ids, info) for the head whose scores reproduce the
    published ranks CSV (script 21's rule), or None if its text tower is not cached."""
    cfg = HEADS[tag]
    tf = load_text_features(cfg['text'])
    if tf is None:
        if verbose:
            print(f"[{tag}] SKIPPED -- text features '{cfg['text']}' not cached (script 32).")
        return None
    text_feats, scale = tf
    X, ids = load_embeddings(cfg['emb_dir'], df_val['segment_id'].tolist())
    y = df_val.set_index('segment_id').loc[ids, 'verb_label'].to_numpy()

    if cfg.get('fixed_ckpt'):
        candidates = [cfg['fixed_ckpt']]
    else:
        hist = pd.read_csv(os.path.join(OUT_DIR, cfg['history']))
        if 'tag' in hist:
            hist = hist[hist['tag'] == tag]
        best = hist.sort_values(['val_top1', 'val_top5'], ascending=False).groupby('seed').first().reset_index()
        candidates = [f"nb8_{tag}_seed{int(r.seed)}_bestepoch{int(r.epoch)}.pt" for r in best.itertuples()]
    published = (pd.read_csv(os.path.join(OUT_DIR, cfg['published'])).set_index('segment_id')
                 if cfg['published'] else None)
    for ck in candidates:
        path = os.path.join(CKPT_DIR, ck)
        if not os.path.exists(path):
            continue
        W = torch.load(path, map_location='cpu')['weight']
        ranks = eval_with_ranks(score_matrix(X, W, text_feats, scale), ids, y, verb_vocab)
        agree = 1.0
        if published is not None:
            shared = published.index.intersection(ranks.index)
            agree = (published.loc[shared, 'top1_correct'].astype(bool)
                     == ranks.loc[shared, 'top1_correct'].astype(bool)).mean()
        if agree > 0.999:
            info = {'tag': tag, 'ckpt': ck, 'agreement': agree, 'top1': ranks['top1_correct'].mean(),
                    'n': len(ids), 'text_feats': text_feats, 'logit_scale': scale, 'ranks': ranks}
            if verbose:
                print(f"[{tag}] {ck}: Top-1 {info['top1']*100:.2f}% on N={len(ids):,} "
                      f"(reproduces published ranks on {agree:.2%})")
            return W, X, ids, info
        if verbose:
            print(f"[{tag}] {ck} reproduces published ranks on only {agree:.2%} -- trying next")
    raise RuntimeError(f"[{tag}] no checkpoint on disk reproduces {cfg['published']}")


# ── Gate ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df, verb_vocab, label2id = load_segments()
    df_train = df[df['subset'] == 'training'].reset_index(drop=True)
    df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
    rows = []

    print("=== Gate 1: existing representative heads through cached text features ===")
    for tag in HEADS:
        r = load_representative_head(tag, df_val, verb_vocab)
        if r is None:
            rows.append({'check': 'scoring', 'model': tag, 'status': 'SKIPPED (no text features)'})
            continue
        info = r[3]
        rows.append({'check': 'scoring', 'model': tag, 'ckpt': info['ckpt'], 'n': info['n'],
                     'top1': info['top1'], 'agreement_with_published': info['agreement'], 'status': 'PASS'})

    print("\n=== Gate 2: retrain with this module's recipe, compare to published 2-seed mean ===")
    published_mean = {
        'dinov2': pd.read_csv(os.path.join(OUT_DIR, 'nb8_matched_projection_seed_aggregate.csv'))
                    .set_index('model').loc['dinov2', 'top1_mean'],
        'vjepa2_pretrainedpool': pd.read_csv(os.path.join(OUT_DIR, 'nb8_vjepa2_pretrainedpool_seed_aggregate.csv'))
                    .set_index('model').loc['vjepa2_pretrainedpool', 'top1_mean'],
    }
    text_feats, scale = load_text_features('dinov2_dinotxt')
    all_ok = True
    for tag, target in published_mean.items():
        cfg = HEADS[tag]
        Xtr, tr_ids = load_embeddings(cfg['emb_dir'], df_train['segment_id'].tolist())
        Xva, va_ids = load_embeddings(cfg['emb_dir'], df_val['segment_id'].tolist())
        ytr = df_train.set_index('segment_id').loc[tr_ids, 'verb_label'].to_numpy()
        yva = df_val.set_index('segment_id').loc[va_ids, 'verb_label'].to_numpy()
        res = fit_protocol(Xtr, ytr, Xva, yva, text_feats, scale, cfg['out_dim'], tag=tag)
        gap = res['top1_mean'] - target
        ok = abs(gap) <= GATE_TOL
        all_ok &= ok
        print(f"  [{tag}] retrained {res['top1_mean']*100:.2f}% vs published {target*100:.2f}% "
              f"-> gap {gap*100:+.2f}pp  {'PASS' if ok else 'FAIL'} (tol {GATE_TOL*100:.1f}pp)")
        rows.append({'check': 'training', 'model': tag, 'n': len(va_ids), 'lr': res['lr'],
                     'top1': res['top1_mean'], 'published_top1': target, 'gap': gap,
                     'status': 'PASS' if ok else 'FAIL'})

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'nb8_phase7_gate.csv'), index=False)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'nb8_phase7_gate.csv')}")
    print("GATE PASSED" if all_ok else "GATE FAILED -- do not trust Phase 7 numbers until resolved")
    print("DONE")
