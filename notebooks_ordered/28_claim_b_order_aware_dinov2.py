"""
Claim B, replayed for the order-aware DINOv2 head: does the trained
DINOv2-halves+DinoTXT alignment head score a clip's true verb higher forward
than reversed, the way V-JEPA2's trained head does (script 15: 64.5%
forward-win rate, mean diff +0.0181)? Everything else this session tested raw
discriminability (a generic probe); this is the first test of the actual
trained alignment behavior the paper's Claim B depends on.

Uses script 27's trained checkpoint (no retraining) and script 26's cached
forward halves-pool embeddings; reversed embeddings are computed on the fly
from script 22's per-frame stacks via torch.flip (exact, no new extraction) --
the same trick used throughout this investigation. CPU/GPU-light: no video
decode, no training, just loading cached tensors through a frozen projection.

Reports both the segment-level bootstrap CI (script 15/24's original method)
and the video-cluster bootstrap CI (script 24's correction), side by side,
so this isn't reintroducing the pseudoreplication issue script 24 fixed for
V-JEPA2's version of this same test.

Reads : embeddings/<...>/dinov2_perframe_halves/<segment_id>.pt (script 26, forward)
        embeddings/<...>/dinov2_perframe/<segment_id>.pt (script 22, for reversed via flip)
        checkpoints/nb8_dinov2halves_seed0_bestepoch10.pt
        annotations/segments_verbs_timed_clean.csv (for video_id, cluster bootstrap)
Writes: processed/nb8_claim_b_dinov2halves.csv
        processed/nb8_claim_b_dinov2halves_summary.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from scipy.stats import binomtest

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
CKPT_DIR = PATHS['checkpoints']
HALVES_DIR = os.path.join(PATHS['embeddings'], 'dinov2_perframe_halves')
PERFRAME_DIR = os.path.join(PATHS['embeddings'], 'dinov2_perframe')
IN_DIM, OUT_DIM = 1280, 4096
# Representative: both seeds tied exactly on val_top1 (0.4318); seed0/epoch10 has the
# higher val_top5 (0.7509 vs seed1/epoch7's 0.7445) -- same tie-break rule select_best_epoch
# already uses (top1 first, top5 second). Documented quirk, same as the N_SEEDS=2
# ambiguity noted in script 02/21 for the other backbones.
CKPT_PATH = os.path.join(CKPT_DIR, 'nb8_dinov2halves_seed0_bestepoch10.pt')

DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
df_val = df[df['subset'] == 'validation'].set_index('segment_id')

available = {f[:-3] for f in os.listdir(HALVES_DIR) if f.endswith('.pt')}
val_ids = [sid for sid in df_val.index if sid in available]
print(f"Validation segments with halves-pool features: {len(val_ids):,}")

model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.text_model.head.linear_projection.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
model.eval()
tok = get_tokenizer()
print(f"Loaded trained checkpoint: {os.path.basename(CKPT_PATH)}")

verb_tokens = tok.tokenize(verb_vocab).to(DEVICE)
with torch.no_grad():
    verb_text_embs = model.encode_text(verb_tokens, normalize=True)  # (282, 4096)

results = []
for seg_id in tqdm(val_ids, desc="Scoring forward vs reversed against true verb text"):
    verb = df_val.loc[seg_id, 'extracted_verb']
    if verb not in label2id:
        continue
    fwd = torch.load(os.path.join(HALVES_DIR, f"{seg_id}.pt"), map_location='cpu')
    stack = torch.load(os.path.join(PERFRAME_DIR, f"{seg_id}.pt"), map_location='cpu')
    T = stack.shape[0]
    rstack = torch.flip(stack, [0])
    half = max(T // 2, 1)
    rev = torch.cat([rstack[:half].mean(dim=0), rstack[half:].mean(dim=0)]) if T > half \
        else torch.cat([rstack.mean(dim=0), rstack.mean(dim=0)])

    fwd = F.normalize(fwd.to(DEVICE), dim=-1)
    rev = F.normalize(rev.to(DEVICE), dim=-1)
    text_emb = verb_text_embs[label2id[verb]]
    sim_fwd = (fwd @ text_emb).item()
    sim_rev = (rev @ text_emb).item()
    results.append({'segment_id': seg_id, 'ground_truth': verb, 'video_id': df_val.loc[seg_id, 'video_id'],
                     'sim_forward': sim_fwd, 'sim_reversed': sim_rev,
                     'diff_fwd_minus_rev': sim_fwd - sim_rev})

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, 'nb8_claim_b_dinov2halves.csv'), index=False)

n = len(res_df)
diffs = res_df['diff_fwd_minus_rev'].to_numpy()
mean_diff = diffs.mean()
frac_fwd_wins = (diffs > 0).mean()
n_decisive = int((diffs != 0).sum())
n_fwd = int((diffs > 0).sum())
sign_test = binomtest(n_fwd, n_decisive, 0.5)

print(f"\n=== DINOv2-halves+DinoTXT (order-aware): forward vs. reversed sim to TRUE verb text ===")
print(f"N segments: {n}")
print(f"Mean diff(fwd - rev): {mean_diff:+.4f}")
print(f"Forward wins: {frac_fwd_wins*100:.1f}%  ({n_fwd}/{n_decisive} decisive)  sign-test p={sign_test.pvalue:.4g}")

# ── Segment-level bootstrap (script 15's original method) ──────────────────
rng = np.random.default_rng(0)
boot_seg = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(10000)])
seg_ci_lo, seg_ci_hi = np.percentile(boot_seg, [2.5, 97.5])
seg_p = 2 * min((boot_seg <= 0).mean(), (boot_seg >= 0).mean())
print(f"\nSegment-level bootstrap: 95% CI=[{seg_ci_lo:+.4f},{seg_ci_hi:+.4f}]  p={seg_p:.4g}")

# ── Video-cluster bootstrap (script 24's correction) ────────────────────────
vstats = res_df.groupby('video_id')['diff_fwd_minus_rev'].agg(['count', 'sum']).reset_index()
n_videos = len(vstats)
n_a = vstats['count'].to_numpy(); sum_a = vstats['sum'].to_numpy()
rng2 = np.random.default_rng(0)
idx = rng2.integers(0, n_videos, size=(10000, n_videos))
boot_n = n_a[idx].sum(axis=1); boot_sum = sum_a[idx].sum(axis=1)
boot_cluster = boot_sum / boot_n
clu_ci_lo, clu_ci_hi = np.percentile(boot_cluster, [2.5, 97.5])
clu_p = 2 * min((boot_cluster <= 0).mean(), (boot_cluster >= 0).mean())
print(f"Video-cluster bootstrap (n_videos={n_videos}): 95% CI=[{clu_ci_lo:+.4f},{clu_ci_hi:+.4f}]  p={clu_p:.4g}")

print(f"\n=== For comparison: V-JEPA2 pretrained-pool (script 15) ===")
print(f"Mean diff +0.0181  Forward wins 64.5% (2,107/3,268)  segment-CI [0.0162,0.0199]  "
      f"video-cluster-CI [0.0162,0.0198]  both p<0.0001")

summary = pd.DataFrame([{
    'model': 'dinov2halves', 'n': n, 'mean_diff': mean_diff, 'frac_fwd_wins': frac_fwd_wins,
    'sign_test_p': sign_test.pvalue,
    'seg_ci_lo': seg_ci_lo, 'seg_ci_hi': seg_ci_hi, 'seg_boot_p': seg_p,
    'cluster_ci_lo': clu_ci_lo, 'cluster_ci_hi': clu_ci_hi, 'cluster_boot_p': clu_p,
    'n_videos': n_videos,
}])
summary.to_csv(os.path.join(OUT_DIR, 'nb8_claim_b_dinov2halves_summary.csv'), index=False)
print("\nSaved -> nb8_claim_b_dinov2halves.csv, nb8_claim_b_dinov2halves_summary.csv")
print("DONE")
