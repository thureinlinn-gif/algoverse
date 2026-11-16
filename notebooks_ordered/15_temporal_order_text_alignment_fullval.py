"""
Does the trained V-JEPA2 + DinoTXT system -- the actual multimodal combination
this paper is about, not just V-JEPA2's raw video encoder -- assign higher
video-text similarity to a clip played FORWARD against its true verb's text
embedding than to the same clip played BACKWARD?

This is a stronger, more on-target test than 07_temporal_order_raw_embedding_probe.py
(which only probed the raw pooled V-JEPA2 embedding with an arbitrary freshly
trained linear classifier -- it never touched DinoTXT's text space, so it didn't
actually demonstrate anything about the video+text combination). Here we reuse
the ALREADY-TRAINED matched-projection checkpoint (nb8_vjepa2_pretrainedpool_seed0)
-- the real text-to-video alignment head -- and ask whether cosine similarity to
the ground-truth verb's text embedding is systematically higher for the forward
clip than the reversed one.

For DINOv2/DINOv3, sim(forward, text) and sim(reversed, text) are IDENTICAL by
construction (mean-pooling is order-invariant, so the video embedding itself
doesn't change when reversed) -- so any measured gap here is a capability
DINOv2/DINOv3 cannot have at all, not just one they score lower on.

Extracts forward+reversed pooled V-JEPA2 embeddings for a fresh sample (saving
the raw embeddings this time, unlike the prior script, so this run's output is
reusable), encodes each segment's ground-truth verb via the trained DinoTXT
projection, and compares similarity(forward, verb_text) vs similarity(reversed,
verb_text) per segment with a paired sign test + bootstrap CI.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
from tqdm.auto import tqdm

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUT_DIR = PATHS['processed']
CKPT_DIR = PATHS['checkpoints']
VIDEO_DIR = PATHS['videos']
N_SAMPLE = 10_000  # >= full val set size, so .sample() below just returns everyone (shuffled)
SEED = 1  # different sample from 07_temporal_order_raw_embedding_probe.py's SEED=0, for an independent check

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
df_sample = df_val.sample(n=min(N_SAMPLE, len(df_val)), random_state=SEED).reset_index(drop=True)
print(f"Sampled {len(df_sample)} val segments", flush=True)

# ── Extract forward + reversed pooled V-JEPA2 embeddings (save raw this time) ──
MODEL_NAME = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
processor = VJEPA2VideoProcessor.from_pretrained(MODEL_NAME)
full_model = VJEPA2ForVideoClassification.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, ignore_mismatched_sizes=True,
).to(DEVICE)
encoder = full_model.vjepa2
pooler = full_model.pooler
for p in encoder.parameters():
    p.requires_grad = False
for p in pooler.parameters():
    p.requires_grad = False
encoder.eval(); pooler.eval()
FRAMES_PER_CLIP = encoder.config.frames_per_clip
print(f"FRAMES_PER_CLIP={FRAMES_PER_CLIP}", flush=True)

EMB_DIR_FWD = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_order_fwd')
EMB_DIR_REV = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_order_rev')
os.makedirs(EMB_DIR_FWD, exist_ok=True)
os.makedirs(EMB_DIR_REV, exist_ok=True)


@torch.no_grad()
def embed_clip(vr, fps, start_sec, end_sec, reversed_order):
    start_idx = max(0, int(start_sec * fps))
    end_idx = min(len(vr) - 1, int(end_sec * fps))
    if start_idx >= end_idx:
        return None
    indices = np.linspace(start_idx, end_idx, FRAMES_PER_CLIP, dtype=int)
    if reversed_order:
        indices = indices[::-1].copy()
    frames = vr.get_batch(indices).asnumpy()
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2)
    inputs = processor(clip, return_tensors='pt').to(DEVICE)
    enc_out = encoder(**inputs, skip_predictor=True)
    pooled = pooler(enc_out.last_hidden_state).squeeze(0).cpu()
    return pooled


seg_records = []
by_video = df_sample.groupby('video_id')
errors = 0
for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="Forward+reversed extraction"):
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        errors += len(grp)
        continue
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        errors += len(grp)
        continue
    fps = vr.get_avg_fps()
    for _, row in grp.iterrows():
        fwd_path = os.path.join(EMB_DIR_FWD, f"{row['segment_id']}.pt")
        rev_path = os.path.join(EMB_DIR_REV, f"{row['segment_id']}.pt")
        try:
            if not (os.path.exists(fwd_path) and os.path.exists(rev_path)):
                fwd = embed_clip(vr, fps, row['start_sec'], row['end_sec'], reversed_order=False)
                rev = embed_clip(vr, fps, row['start_sec'], row['end_sec'], reversed_order=True)
                if fwd is None or rev is None:
                    errors += 1
                    continue
                torch.save(fwd, fwd_path)
                torch.save(rev, rev_path)
            seg_records.append({'segment_id': row['segment_id'], 'ground_truth': row['extracted_verb']})
        except Exception as e:
            print(f"  ERROR {row['segment_id']}: {e}", flush=True)
            errors += 1

print(f"\nExtracted {len(seg_records)} segments ({errors} errors/skipped)", flush=True)
del full_model, encoder, pooler
torch.cuda.empty_cache()

# ── Load the ALREADY-TRAINED text-to-video projection (matched-projection recipe) ──
DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

tokenizer_v2 = get_tokenizer()
IN_DIM, OUT_DIM = 1280, 1024
model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
model.text_model.head.linear_projection = nn.Linear(IN_DIM, OUT_DIM, bias=False).to(DEVICE)

CKPT_PATH = os.path.join(CKPT_DIR, 'nb8_vjepa2_pretrainedpool_seed0_bestepoch10.pt')
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.text_model.head.linear_projection.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
model.eval()
print(f"\nLoaded trained text-to-video projection from {CKPT_PATH}", flush=True)

# Bare-verb text, matching this project's V-JEPA2 protocol (no caption context)
verb_tokens = tokenizer_v2.tokenize(verb_vocab).to(DEVICE)
with torch.no_grad():
    verb_text_embs = model.encode_text(verb_tokens, normalize=True)  # (n_verbs, 1024)
label2id = {v: i for i, v in enumerate(verb_vocab)}

# ── Compare sim(forward, true_verb_text) vs sim(reversed, true_verb_text) ──
results = []
for r in tqdm(seg_records, desc="Scoring forward vs reversed against true verb text"):
    seg_id, verb = r['segment_id'], r['ground_truth']
    if verb not in label2id:
        continue
    fwd = torch.load(os.path.join(EMB_DIR_FWD, f"{seg_id}.pt")).to(DEVICE)
    rev = torch.load(os.path.join(EMB_DIR_REV, f"{seg_id}.pt")).to(DEVICE)
    fwd = F.normalize(fwd, dim=-1)
    rev = F.normalize(rev, dim=-1)
    text_emb = verb_text_embs[label2id[verb]]
    sim_fwd = (fwd @ text_emb).item()
    sim_rev = (rev @ text_emb).item()
    results.append({'segment_id': seg_id, 'ground_truth': verb,
                     'sim_forward': sim_fwd, 'sim_reversed': sim_rev,
                     'diff_fwd_minus_rev': sim_fwd - sim_rev})

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_text_alignment.csv'), index=False)

n = len(res_df)
mean_diff = res_df['diff_fwd_minus_rev'].mean()
frac_fwd_wins = (res_df['diff_fwd_minus_rev'] > 0).mean()
frac_rev_wins = (res_df['diff_fwd_minus_rev'] < 0).mean()
frac_tie = (res_df['diff_fwd_minus_rev'] == 0).mean()

# Paired bootstrap CI on the mean difference
rng = np.random.default_rng(0)
diffs = res_df['diff_fwd_minus_rev'].values
boot_means = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(10000)])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
p_value = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())

# Sign test (binomial) on forward-wins vs reversed-wins, excluding exact ties
from scipy.stats import binomtest
n_decisive = int((res_df['diff_fwd_minus_rev'] != 0).sum())
n_fwd = int((res_df['diff_fwd_minus_rev'] > 0).sum())
sign_test = binomtest(n_fwd, n_decisive, 0.5)

print(f"\n=== Forward vs. reversed: similarity to TRUE verb text, via trained V-JEPA2+DinoTXT projection ===", flush=True)
print(f"N segments: {n}", flush=True)
print(f"Mean sim(forward, text) - sim(reversed, text): {mean_diff:+.4f}  "
      f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  bootstrap p={p_value:.4f}", flush=True)
print(f"Forward wins: {frac_fwd_wins*100:.1f}%  Reversed wins: {frac_rev_wins*100:.1f}%  "
      f"Ties: {frac_tie*100:.1f}%", flush=True)
print(f"Sign test (forward-win rate vs. 50%): {n_fwd}/{n_decisive} = {n_fwd/n_decisive*100:.1f}%  "
      f"p={sign_test.pvalue:.4f}", flush=True)
print(f"\nFor DINOv2/DINOv3: sim(forward, text) == sim(reversed, text) EXACTLY, for every "
      f"segment -- not measured, proven (mean-pooled video embedding cannot change under "
      f"any frame reordering, so this entire comparison is a hard 50/50 tie by construction).", flush=True)
print("\nDONE", flush=True)
