"""
Corrected, full-validation DINOv2+DinoTXT zero-shot verb prediction.

Fixes Confound C (NB5's original loop iterated over ALL 13,003 segments,
train+val combined, and was interrupted at 566/13,003 -> only 134 landed in
val) by restricting the loop to the validation split only, so it actually
completes, and by sourcing from the same canonical clean CSV NB6/V-JEPA2
uses (segments_verbs_timed_clean.csv), matching the CSV-standardization fix.

Logic is unchanged from the reviewed NB5 / NB8-Part-2 zero-shot loop:
caption-context text-conditioning, frozen DinoTXT, cosine similarity.
"""
import os, sys, json, time
import torch
import torch.nn.functional as F
import pandas as pd

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)

from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
label2id = {v: i for i, v in enumerate(verb_vocab)}
n_verbs = len(verb_vocab)
assert n_verbs == 282, f"Expected 282 verbs, got {n_verbs}"

df_val = df[df['subset'] == 'validation'].reset_index(drop=True)
print(f"Total segments: {len(df):,} | Val segments: {len(df_val):,} | Verbs: {n_verbs}", flush=True)

EMB_DIR = PATHS['embeddings']  # dinov2 embeddings, precomputed, 13003 files
n_avail = len([f for f in os.listdir(EMB_DIR) if f.endswith('.pt')])
print(f"DINOv2 embeddings available: {n_avail:,} at {EMB_DIR}", flush=True)

print("Loading frozen DINOv2 + DinoTXT...", flush=True)
model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
tokenizer = get_tokenizer()
for p in model.parameters():
    p.requires_grad = False
print("Model loaded and frozen.", flush=True)


@torch.no_grad()
def get_text_embeddings_caption_context(masked_caption, verbs):
    caption_context = masked_caption.replace('[MASK]', '{}')
    verb_texts = [caption_context.format(v) for v in verbs]
    tokens = tokenizer.tokenize(verb_texts).to(DEVICE)
    text_features = model.encode_text(tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)


records = []
missing = 0
start = time.time()

with torch.no_grad():
    for i, row in df_val.iterrows():
        emb_path = os.path.join(EMB_DIR, f"{row['segment_id']}.pt")
        if not os.path.exists(emb_path):
            missing += 1
            continue
        video_emb = torch.load(emb_path, map_location=DEVICE)
        video_emb = video_emb / video_emb.norm()

        text_embs = get_text_embeddings_caption_context(row['masked_caption'], verb_vocab)
        similarity = (text_embs @ video_emb).cpu()

        ranked = similarity.argsort(descending=True).tolist()
        gt_idx = label2id.get(row['extracted_verb'], -1)
        rank = ranked.index(gt_idx) + 1 if gt_idx in ranked else n_verbs + 1

        records.append({
            'segment_id': row['segment_id'], 'video_id': row['video_id'], 'subset': row['subset'],
            'ground_truth': row['extracted_verb'], 'top1_pred': verb_vocab[ranked[0]],
            'top1_correct': ranked[0] == gt_idx, 'top5_correct': gt_idx in ranked[:5],
            'rank': rank, 'reciprocal_rank': 1.0 / rank,
        })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = elapsed / (i + 1)
            eta_min = rate * (len(df_val) - (i + 1)) / 60
            running_top1 = sum(r['top1_correct'] for r in records) / len(records)
            print(f"  [{i+1}/{len(df_val)}] elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min "
                  f"running_top1={running_top1*100:.2f}%", flush=True)

results_df = pd.DataFrame(records)
top1 = results_df['top1_correct'].mean()
top5 = results_df['top5_correct'].mean()
print(f"\nDONE. DINOv2 zero-shot full val -- N={len(results_df):,} (missing={missing}) "
      f"Top-1={top1*100:.2f}% Top-5={top5*100:.2f}%", flush=True)

out_path = os.path.join(PATHS['processed'], 'dinov2_zeroshot_fullval_fixed.csv')
results_df.to_csv(out_path, index=False)
print(f"Saved -> {out_path}", flush=True)
