"""
Phase 7, Experiment B: the papers' evaluation protocol -- a trained attentive probe
over per-timestep tokens, not a pooled vector into a linear head -- on the 1,500
POC segments that already have per-timestep features. Zero extraction.

Arms (identical treatment; each run with an attentive probe AND a mean-pool control):
  vjepa2_lf   embeddings/.../vjepa2_poc_temporal          (32, 1024)  facebook/vjepa2-vitl-fpc64-256,
              label-free, space-only mean per timestep (nb8_attnpool_poc.py)
  dinov2      embeddings/.../dinov2_perframe              (<=50, 2048) -> 32 steps by linspace index
  vjepa2_ssv2 embeddings/.../vjepa2_poc_ssv2_pertimestep  (8, 1024)   SSv2-finetuned; reference only
Both main arms sample frames evenly across the whole segment (64 and <=50 frames).

Probe (sized for ~1,200 training examples -- the 2-block/1024-d version would be
~25M params, and the pooler-finetune POC already lost 5.3pp to overfitting at that
scale): LayerNorm -> Linear(d_in, 256) -> + learned temporal position embedding ->
1 pre-norm transformer block (4 heads, FF 1024, dropout 0.1) -> cross-attention
with one learnable query -> LayerNorm. Position embeddings are required: Phase 6
showed attention pooling without them is permutation-invariant over time. The
text side is a bias-free Linear(1280 -> 256) over script 32's cached DinoTXT
features, scored with the frozen logit_scale -- the project's text-aligned protocol.
The mean-pool control is the SAME module fed the time-mean as a length-1 sequence,
so (attn - mean) isolates what attention + order adds.

Asymmetry stated up front: dinov2_perframe is dino.txt's encode_image output, already
aligned to this text space; V-JEPA2 tokens are raw encoder features. At 1,200
training examples that favours DINOv2, which is why the interaction
(V-JEPA2 attn - mean) - (DINOv2 attn - mean) is reported beside the raw difference.

Evaluation: 5-fold video-grouped CV over all 1,500 segments (every segment is
scored exactly once, by a model that never saw its video). Inside each outer fold,
LR and epoch count are chosen on a 20% video-grouped inner split (min inner CE),
then the probe is retrained on the full outer-train fold. 2 seeds; per-segment
metrics are averaged over seeds. Hard negatives come from script 34 (held-out for
val segments, out-of-fold for train segments).

Metrics: Top-1 (282-way), hard-pair and frequency-random-pair accuracy (script 34),
macro-F1 (reported, but 85 of the 126 POC verbs have <5 examples -- not interpretable).

Controls / verification (asserted or printed):
  - outer folds share zero videos;
  - time-mean of vjepa2_poc_temporal reproduces the published vjepa2_base_dinotxt
    mean-pool vectors (same label-free fpc64 checkpoint -- confirms the main arm has
    no SSv2 training);
  - label-shuffle: attentive probe trained on permuted labels falls to the prior;
  - time-permutation at test: shuffling the time axis lowers the attentive probes
    if (and only if) they use order.

Power: n=1,500, so a 3pp true difference on the pair metric is detected only
~half the time. A null here is inconclusive, not evidence of absence.

Reads : embeddings (3 POC dirs + dinov2_perframe + vjepa2_base_dinotxt),
        processed/nb8_text_features_dinov2_dinotxt.pt (32), processed/nb8_text_baseline_masked_caption.csv (34)
Writes: processed/nb8_attnprobe_poc_segments.csv, nb8_attnprobe_poc_summary.csv, nb8_attnprobe_poc_diffs.csv
~10 min on one A10.
"""
import os, sys, importlib, math
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
lib = importlib.import_module('33_linear_head_lib')
DEVICE, OUT_DIR, EMB = lib.DEVICE, lib.OUT_DIR, lib.EMB

WIDTH, HEADS, FF, DROPOUT = 256, 4, 1024, 0.1
T_STEPS = 32
LRS = [3e-4, 1e-3]
WEIGHT_DECAY = 0.05
MAX_EPOCHS = 40
BATCH = 64
SEEDS = [0, 1]
N_FOLDS = 5
INNER_FRAC = 0.2
DECISION_PP = 0.03
# Learned temporal position embedding init scale. 0.02 (ViT default) is ~3% of the
# projected token scale; in the 2026-10-24 run the time-permutation control showed
# zero effect at that scale, so POS_INIT_STD=0.5 was run as the diagnostic variant
# (outputs suffixed _pos0.5). See REPRODUCE.md Phase 7.
POS_INIT_STD = float(os.environ.get('POS_INIT_STD', 0.02))
OUT_SUFFIX = '' if POS_INIT_STD == 0.02 else f'_pos{POS_INIT_STD:g}'

ARMS = {
    'vjepa2_lf': os.path.join(EMB, 'vjepa2_poc_temporal'),
    'dinov2': os.path.join(EMB, 'dinov2_perframe'),
    'vjepa2_ssv2': os.path.join(EMB, 'vjepa2_poc_ssv2_pertimestep'),
}


def to_32_steps(x):
    """Resample a (T, d) stack to 32 steps by linspace index (DINOv2 has <=50 frames)."""
    if x.shape[0] == T_STEPS:
        return x
    return x[np.linspace(0, x.shape[0] - 1, T_STEPS).round().astype(int)]


# ── Data ─────────────────────────────────────────────────────────────────────
df, verb_vocab, label2id = lib.load_segments()
poc_ids = sorted(f[:-3] for f in os.listdir(ARMS['vjepa2_lf']) if f.endswith('.pt'))
poc = df.set_index('segment_id').loc[poc_ids].reset_index()
print(f"POC segments: {len(poc):,}  ({poc['subset'].value_counts().to_dict()}), videos: {poc['video_id'].nunique():,}")
vc = poc['extracted_verb'].value_counts()
print(f"Verbs present: {len(vc)} of 282 ({(vc < 5).sum()} with <5 examples); 'add' = {vc['add']/len(poc)*100:.1f}%")

feats = {}
for arm, d in ARMS.items():
    X, ids = lib.load_embeddings(d, poc_ids, transform=to_32_steps if arm == 'dinov2' else None)
    assert ids == poc_ids, f"{arm}: missing POC segments"
    feats[arm] = X.float()
    print(f"  {arm:<12} {tuple(X.shape)}")

# Verification: the label-free arm's time-mean is the published label-free mean-pool vector.
pub, pub_ids = lib.load_embeddings(os.path.join(EMB, 'vjepa2_base_dinotxt'), poc_ids)
cos = F.cosine_similarity(feats['vjepa2_lf'][[poc_ids.index(s) for s in pub_ids]].mean(1), pub, dim=-1)
print(f"Check: mean(vjepa2_poc_temporal) vs vjepa2_base_dinotxt -- cosine min {cos.min():.4f}, "
      f"median {cos.median():.4f} over {len(pub_ids)} segments")
assert cos.median() > 0.99, "POC tokens do not match the label-free mean-pool features"

y_all = torch.tensor(poc['verb_label'].to_numpy())
neg = pd.read_csv(os.path.join(OUT_DIR, 'nb8_text_baseline_masked_caption.csv')).set_index('segment_id').loc[poc_ids]
hard_idx = torch.tensor(neg['hard_negative'].map(label2id).to_numpy())
freq_idx = torch.tensor(neg['freq_negative'].map(label2id).to_numpy())
assert not (hard_idx == y_all).any()

text_feats, logit_scale = lib.load_text_features('dinov2_dinotxt')
groups = poc['video_id'].to_numpy()
outer = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(poc)), groups=groups))
for tr, te in outer:
    assert not set(groups[tr]) & set(groups[te]), "outer folds share videos"
print(f"Outer folds: {[len(te) for _, te in outer]} test segments, zero video overlap (asserted)")


# ── Probe ────────────────────────────────────────────────────────────────────
class TemporalProbe(nn.Module):
    def __init__(self, d_in, t_steps):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, WIDTH))
        self.pos = nn.Parameter(torch.zeros(1, t_steps, WIDTH))
        nn.init.trunc_normal_(self.pos, std=POS_INIT_STD, a=-2 * POS_INIT_STD, b=2 * POS_INIT_STD)
        self.block = nn.TransformerEncoderLayer(WIDTH, HEADS, FF, DROPOUT, batch_first=True, norm_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, WIDTH) * 0.02)
        self.xattn = nn.MultiheadAttention(WIDTH, HEADS, dropout=DROPOUT, batch_first=True)
        self.norm = nn.LayerNorm(WIDTH)
        self.text_proj = nn.Linear(text_feats.shape[1], WIDTH, bias=False)

    def forward(self, x):
        h = self.block(self.inp(x) + self.pos)
        v, _ = self.xattn(self.query.expand(len(h), -1, -1), h, h)
        v = F.normalize(self.norm(v.squeeze(1)), dim=-1)
        t = F.normalize(self.text_proj(text_feats), dim=-1)
        return logit_scale * v @ t.T


def make_input(X, pool):
    return X.mean(1, keepdim=True) if pool == 'mean' else X


def train(X, y, lr, epochs, seed, X_eval=None, y_eval=None):
    """Returns (model, per-epoch eval CE list)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = TemporalProbe(X.shape[-1], X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(X) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.1, anneal_strategy='cos')
    Xd, yd = X.to(DEVICE), y.to(DEVICE)
    g = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g).to(DEVICE)
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(Xd[idx]), yd[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if X_eval is not None:
            curve.append(F.cross_entropy(predict(model, X_eval), y_eval).item())
    return model, curve


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.cat([model(X[i:i + 512].to(DEVICE)).cpu() for i in range(0, len(X), 512)])


def choose_lr_epochs(X, y, grp, seed):
    """Inner 20% video-grouped split: (lr, epochs) with the lowest inner CE."""
    itr, iva = next(GroupShuffleSplit(n_splits=1, test_size=INNER_FRAC, random_state=seed)
                    .split(np.zeros(len(X)), groups=grp))
    best = (float('inf'), None, None)
    for lr in LRS:
        _, curve = train(X[itr], y[itr], lr, MAX_EPOCHS, seed, X[iva], y[iva])
        ep = int(np.argmin(curve))
        if curve[ep] < best[0]:
            best = (curve[ep], lr, ep + 1)
    return best[1], best[2]


def cv_scores(arm, pool, shuffle_labels=False, seeds=SEEDS, permute_time_test=False):
    """Out-of-fold logits (N, 282) per seed, plus logits on time-permuted test inputs."""
    X = make_input(feats[arm], pool)
    out = {s: torch.zeros(len(X), len(verb_vocab)) for s in seeds}
    out_perm = {s: torch.zeros(len(X), len(verb_vocab)) for s in seeds}
    chosen = []
    for k, (tr, te) in enumerate(outer):
        tr_t, te_t = torch.tensor(tr), torch.tensor(te)
        y_tr = y_all[tr_t]
        if shuffle_labels:
            y_tr = y_tr[torch.randperm(len(y_tr), generator=torch.Generator().manual_seed(k))]
        for s in seeds:
            lr, ep = choose_lr_epochs(X[tr_t], y_tr, groups[tr], s)
            model, _ = train(X[tr_t], y_tr, lr, ep, s)
            out[s][te_t] = predict(model, X[te_t])
            if permute_time_test and X.shape[1] > 1:
                g = torch.Generator().manual_seed(1000 + k)
                Xp = torch.stack([x[torch.randperm(x.shape[0], generator=g)] for x in X[te_t]])
                out_perm[s][te_t] = predict(model, Xp)
            chosen.append((k, s, lr, ep))
    return out, (out_perm if permute_time_test else None), chosen


def per_segment(logits_by_seed):
    """Seed-averaged per-segment correctness for each metric (values in {0, .5, 1})."""
    cols = {'top1': [], 'hard_pair': [], 'freq_pair': []}
    preds = []
    for L in logits_by_seed.values():
        s_true = L.gather(1, y_all[:, None]).squeeze(1)
        cols['top1'].append((L.argmax(1) == y_all).float())
        cols['hard_pair'].append((s_true > L.gather(1, hard_idx[:, None]).squeeze(1)).float())
        cols['freq_pair'].append((s_true > L.gather(1, freq_idx[:, None]).squeeze(1)).float())
        preds.append(L.argmax(1))
    res = {k: torch.stack(v).mean(0).numpy() for k, v in cols.items()}
    res['macro_f1'] = np.mean([f1_score(y_all.numpy(), p.numpy(), average='macro', zero_division=0) for p in preds])
    return res


# ── Run ──────────────────────────────────────────────────────────────────────
seg = poc[['segment_id', 'video_id', 'subset', 'extracted_verb']].copy()
seg['hard_negative'] = neg['hard_negative'].to_numpy()
summary_rows = []


def record(name, arm, pool, logits, control=''):
    m = per_segment(logits)
    row = {'run': name, 'arm': arm, 'pool': pool, 'control': control, 'n': len(poc),
           'n_videos': poc['video_id'].nunique(), 'macro_f1_uninterpretable': m['macro_f1']}
    for k in ('top1', 'hard_pair', 'freq_pair'):
        seg[f'{name}__{k}'] = m[k]
        cb = lib.cluster_bootstrap(m[k], groups)
        row.update({k: cb['mean'], f'{k}_ci_lo': cb['ci_lo'], f'{k}_ci_hi': cb['ci_hi']})
    summary_rows.append(row)
    print(f"  {name:<28} Top-1 {row['top1']*100:5.1f}  hard-pair {row['hard_pair']*100:5.1f} "
          f"[{row['hard_pair_ci_lo']*100:.1f},{row['hard_pair_ci_hi']*100:.1f}]  "
          f"freq-pair {row['freq_pair']*100:5.1f}  (macro-F1 {row['macro_f1_uninterpretable']*100:.1f})", flush=True)


print(f"\n=== Training probes: 5 outer folds x {len(SEEDS)} seeds, LR/epochs chosen on inner splits ===")
for arm in ARMS:
    for pool in ('attn', 'mean'):
        logits, perm_logits, chosen = cv_scores(arm, pool, permute_time_test=(pool == 'attn'))
        record(f'{arm}_{pool}', arm, pool, logits)
        if perm_logits is not None:
            record(f'{arm}_{pool}_timeperm', arm, pool, perm_logits, control='time axis permuted at test')
        print(f"    chosen (fold, seed, lr, epochs): {chosen}", flush=True)

print("\n=== Label-shuffle control (attentive probe, seed 0) ===")
for arm in ('vjepa2_lf', 'dinov2'):
    logits, _, _ = cv_scores(arm, 'attn', shuffle_labels=True, seeds=[0])
    record(f'{arm}_attn_labelshuffle', arm, 'attn', logits, control='labels shuffled in training')

# Prior-only reference: always predict the training-fold majority ranking (i.e. by frequency).
prior = torch.zeros(len(poc), len(verb_vocab))
for tr, te in outer:
    prior[torch.tensor(te)] = torch.log1p(torch.bincount(y_all[torch.tensor(tr)], minlength=len(verb_vocab)).float())
record('prior_only', 'none', 'none', {0: prior}, control='training-fold verb frequency, no video')

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUT_DIR, f'nb8_attnprobe_poc_summary{OUT_SUFFIX}.csv'), index=False)
seg.to_csv(os.path.join(OUT_DIR, f'nb8_attnprobe_poc_segments{OUT_SUFFIX}.csv'), index=False)


# ── Differences ──────────────────────────────────────────────────────────────
def diff(a, b, metric):
    d = seg[f'{a}__{metric}'] - seg[f'{b}__{metric}']
    cb = lib.cluster_bootstrap(d, groups)
    return {'comparison': f'{a} - {b}', 'metric': metric, 'diff': cb['mean'],
            'ci_lo': cb['ci_lo'], 'ci_hi': cb['ci_hi'], 'p_value': cb['p_value'], 'n': len(seg)}


def interaction(metric):
    d = ((seg[f'vjepa2_lf_attn__{metric}'] - seg[f'vjepa2_lf_mean__{metric}'])
         - (seg[f'dinov2_attn__{metric}'] - seg[f'dinov2_mean__{metric}']))
    cb = lib.cluster_bootstrap(d, groups)
    return {'comparison': '(vjepa2_lf attn - mean) - (dinov2 attn - mean)', 'metric': metric,
            'diff': cb['mean'], 'ci_lo': cb['ci_lo'], 'ci_hi': cb['ci_hi'], 'p_value': cb['p_value'], 'n': len(seg)}


diff_rows = []
for metric in ('hard_pair', 'top1', 'freq_pair'):
    diff_rows += [diff('vjepa2_lf_attn', 'dinov2_attn', metric),
                  diff('vjepa2_lf_mean', 'dinov2_mean', metric),
                  diff('vjepa2_lf_attn', 'vjepa2_lf_mean', metric),
                  diff('dinov2_attn', 'dinov2_mean', metric),
                  interaction(metric),
                  diff('vjepa2_lf_attn', 'vjepa2_lf_attn_timeperm', metric),
                  diff('dinov2_attn', 'dinov2_attn_timeperm', metric)]
diffs = pd.DataFrame(diff_rows)
diffs.to_csv(os.path.join(OUT_DIR, f'nb8_attnprobe_poc_diffs{OUT_SUFFIX}.csv'), index=False)

print("\n=== Differences (pp, video-cluster 95% CI, N=1,500) ===")
for r in diffs.itertuples():
    print(f"  [{r.metric:<9}] {r.comparison:<48} {r.diff*100:+6.2f}  [{r.ci_lo*100:+.2f}, {r.ci_hi*100:+.2f}]  p={r.p_value:.3f}")

r = diffs[(diffs['comparison'] == 'vjepa2_lf_attn - dinov2_attn') & (diffs['metric'] == 'hard_pair')].iloc[0]
hit = r['diff'] >= DECISION_PP and r['ci_lo'] > 0
half = (r['ci_hi'] - r['ci_lo']) / 2
print(f"\n=== Pre-registered check: V-JEPA2 (label-free, attentive) - DINOv2 (attentive), hard pair ===")
print(f"  {r['diff']*100:+.2f}pp [{r['ci_lo']*100:+.2f}, {r['ci_hi']*100:+.2f}] -> {'POSITIVE' if hit else 'null'}  "
      f"(CI half-width {half*100:.1f}pp; 80%-power MDE ~{half/1.96*2.8*100:.1f}pp -- a null below that is inconclusive)")
print("DONE")
