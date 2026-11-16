"""
Sharper version of the forward-vs-reversed finding: does reversing a clip whose
true verb is "add" make the trained V-JEPA2+DinoTXT system's similarity shift
specifically TOWARD "remove" -- its natural temporal-inverse action in this
vocabulary (add(4,721 val+train occurrences) / remove(231)) -- not just away
from "add" in general?

This is a stronger, more legible claim than a bare similarity-gap number: it
says the confusion caused by reversing time is semantically coherent (reversed
"add" looks like "remove"), not just noise. DINOv2/DINOv3 + DinoTXT cannot
produce ANY shift here, in either direction, for any segment -- proven, not
measured (mean-pooled video embedding is identical whether reversed or not).

Method: sample ADD and REMOVE segments from train+val (safe to use train here
since we're only evaluating a fixed, already-trained checkpoint, not fitting
anything new). Extract forward+reversed pooled V-JEPA2 embeddings (reusing the
cache from the other two order-related scripts where segment_ids overlap).
For each segment, compute similarity to both "add" and "remove" text
embeddings, forward and reversed. The key statistic is the DIFFERENTIAL shift:
(sim(reversed, opposite_verb) - sim(reversed, true_verb)) minus
(sim(forward, opposite_verb) - sim(forward, true_verb)) -- positive means
reversing pushed the representation toward the opposite verb specifically.
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
SEED = 2
N_PER_VERB = 200  # caps 'add' sample; 'remove' has ~231 total so mostly all of it is used

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())

df_add = df[df['extracted_verb'] == 'add'].sample(n=min(N_PER_VERB, (df['extracted_verb'] == 'add').sum()), random_state=SEED)
df_remove = df[df['extracted_verb'] == 'remove'].sample(n=min(N_PER_VERB, (df['extracted_verb'] == 'remove').sum()), random_state=SEED)
df_sample = pd.concat([df_add, df_remove], ignore_index=True)
print(f"Sampled {len(df_add)} 'add' + {len(df_remove)} 'remove' segments (train+val combined)", flush=True)

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
for video_id, grp in tqdm(by_video, total=by_video.ngroups, desc="Forward+reversed extraction (add/remove)"):
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

# ── Load trained text-to-video projection (raw linear-layer state dict) ────
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

verb_tokens = tokenizer_v2.tokenize(verb_vocab).to(DEVICE)
with torch.no_grad():
    verb_text_embs = model.encode_text(verb_tokens, normalize=True)
label2id = {v: i for i, v in enumerate(verb_vocab)}
add_idx, remove_idx = label2id['add'], label2id['remove']
add_text, remove_text = verb_text_embs[add_idx], verb_text_embs[remove_idx]

# ── Score forward/reversed against BOTH 'add' and 'remove' text embeddings ──
rows = []
for r in tqdm(seg_records, desc="Scoring add/remove antonym shift"):
    seg_id, verb = r['segment_id'], r['ground_truth']
    fwd = F.normalize(torch.load(os.path.join(EMB_DIR_FWD, f"{seg_id}.pt")).to(DEVICE), dim=-1)
    rev = F.normalize(torch.load(os.path.join(EMB_DIR_REV, f"{seg_id}.pt")).to(DEVICE), dim=-1)
    rows.append({
        'segment_id': seg_id, 'ground_truth': verb,
        'sim_fwd_add': (fwd @ add_text).item(), 'sim_fwd_remove': (fwd @ remove_text).item(),
        'sim_rev_add': (rev @ add_text).item(), 'sim_rev_remove': (rev @ remove_text).item(),
    })
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_antonym_add_remove.csv'), index=False)


def analyze(sub, true_verb, opp_verb, true_col_fwd, opp_col_fwd, true_col_rev, opp_col_rev):
    n = len(sub)
    fwd_margin = sub[opp_col_fwd] - sub[true_col_fwd]     # negative = correctly favors true verb when forward
    rev_margin = sub[opp_col_rev] - sub[true_col_rev]     # how much it favors opposite verb when reversed
    shift = rev_margin - fwd_margin                        # the differential shift toward the opposite verb
    rng = np.random.default_rng(0)
    boot = np.array([shift.values[rng.integers(0, n, n)].mean() for _ in range(10000)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p_value = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    frac_shift_toward_opp = (shift > 0).mean()
    print(f"\n=== True verb = '{true_verb}', opposite = '{opp_verb}'  (N={n}) ===")
    print(f"Forward margin (sim_opp - sim_true), mean: {fwd_margin.mean():+.4f}  "
          f"(negative = correctly favors '{true_verb}')")
    print(f"Reversed margin (sim_opp - sim_true), mean: {rev_margin.mean():+.4f}")
    print(f"Differential shift toward '{opp_verb}' when reversed: {shift.mean():+.4f}  "
          f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p={p_value:.4f}")
    print(f"Fraction of segments where reversal shifted margin toward '{opp_verb}': {frac_shift_toward_opp*100:.1f}%")
    return {'true_verb': true_verb, 'opposite_verb': opp_verb, 'n': n,
            'fwd_margin_mean': fwd_margin.mean(), 'rev_margin_mean': rev_margin.mean(),
            'shift_mean': shift.mean(), 'shift_ci_lo': ci_lo, 'shift_ci_hi': ci_hi,
            'shift_p_value': p_value, 'frac_shift_toward_opposite': frac_shift_toward_opp}


summary = []
sub_add = res[res['ground_truth'] == 'add']
summary.append(analyze(sub_add, 'add', 'remove', 'sim_fwd_add', 'sim_fwd_remove', 'sim_rev_add', 'sim_rev_remove'))
sub_remove = res[res['ground_truth'] == 'remove']
summary.append(analyze(sub_remove, 'remove', 'add', 'sim_fwd_remove', 'sim_fwd_add', 'sim_rev_remove', 'sim_rev_add'))

pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_antonym_summary.csv'), index=False)
print(f"\nFor DINOv2/DINOv3: this entire shift is EXACTLY 0.0000 for every segment, in both "
      f"directions -- proven, not measured (video embedding cannot change under reversal).", flush=True)
print("\nDONE", flush=True)
