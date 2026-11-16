"""
Follow-up to 10_temporal_order_antonym_margin_shift.py: does reversing an 'add'/'remove'
clip actually flip the model's TOP-1 decision over the full 282-verb vocabulary,
or does the pairwise similarity-margin shift (already established: +0.056pp
add->remove, +0.024pp remove->add, both p<0.01) fail to survive contact with
'add's massive class-imbalance prior (4,721 'add' vs 231 'remove' training
examples)?

This matters for the paper: a raw top-1 flip-rate number is NOT a controlled
statistic here, because 'add' can win the full 282-way argmax almost
regardless of what's on screen, given how dominant it is in training. This
script isolates the REVERSAL-ATTRIBUTABLE component from that pre-existing
bias by comparing forward-clip misclassification rate against reversed-clip
classification rate (both at 'is top-1 == add'), and runs a McNemar-style
paired sign test on the discordant pairs -- i.e. it only counts segments
whose top-1 guess actually changed between forward and reversed, rather than
crediting the bias itself as if it were a reversal effect.

Persists analysis that was originally run as one-off interactive Python
during the 2026-08-31 session (produced nb8_temporal_order_antonym_top1_flips.csv
at the time, but the analysis code itself was never saved -- a reproducibility
gap for a paper submission). No new GPU extraction: reuses the cached
embeddings written by 10_temporal_order_antonym_margin_shift.py.
"""
import os, sys, json
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import binomtest

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = PATHS['processed']
CKPT_DIR = PATHS['checkpoints']

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())

DINOV2_REPO = os.path.join(PATHS['repos'], 'dinov2')
if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

tok = get_tokenizer()
model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
model.text_model.head.linear_projection = nn.Linear(1280, 1024, bias=False).to(DEVICE)
CKPT_PATH = os.path.join(CKPT_DIR, 'nb8_vjepa2_pretrainedpool_seed0_bestepoch10.pt')
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.text_model.head.linear_projection.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
verb_tokens = tok.tokenize(verb_vocab).to(DEVICE)
with torch.no_grad():
    verb_text = model.encode_text(verb_tokens, normalize=True)
print(f"Loaded trained text-to-video projection from {CKPT_PATH}", flush=True)

res_path = os.path.join(OUT_DIR, 'nb8_temporal_order_antonym_add_remove.csv')
res = pd.read_csv(res_path)
FWD_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_order_fwd')
REV_DIR = os.path.join(PATHS['embeddings'], 'vjepa2_pretrainedpool_order_rev')

rows = []
for _, r in res.iterrows():
    seg, verb = r['segment_id'], r['ground_truth']
    fwd = F.normalize(torch.load(os.path.join(FWD_DIR, f'{seg}.pt')).to(DEVICE), dim=-1)
    rev = F.normalize(torch.load(os.path.join(REV_DIR, f'{seg}.pt')).to(DEVICE), dim=-1)
    top1_fwd = verb_vocab[(verb_text @ fwd).argmax().item()]
    top1_rev = verb_vocab[(verb_text @ rev).argmax().item()]
    rows.append({'segment_id': seg, 'ground_truth': verb, 'top1_fwd': top1_fwd, 'top1_rev': top1_rev})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_antonym_top1_flips.csv'), index=False)

summary_rows = []
for true_v, opp_v in [('add', 'remove'), ('remove', 'add')]:
    sub = out[out['ground_truth'] == true_v]
    n = len(sub)
    fwd_correct = (sub['top1_fwd'] == true_v).mean()
    rev_still_correct = (sub['top1_rev'] == true_v).mean()
    rev_flipped_to_opp = (sub['top1_rev'] == opp_v).mean()
    fwd_predicted_opp = (sub['top1_fwd'] == opp_v).mean()  # baseline confusion, no reversal
    flip_to_opp = ((sub['top1_fwd'] != opp_v) & (sub['top1_rev'] == opp_v)).sum()
    flip_from_opp = ((sub['top1_fwd'] == opp_v) & (sub['top1_rev'] != opp_v)).sum()
    n_discordant = flip_to_opp + flip_from_opp
    mcnemar_p = binomtest(flip_to_opp, n_discordant, 0.5).pvalue if n_discordant > 0 else float('nan')

    print(f"\n=== true={true_v} (n={n}) ===")
    print(f"forward top1=={true_v}: {fwd_correct*100:.1f}%  |  reversed top1 still =={true_v}: {rev_still_correct*100:.1f}%")
    print(f"BASELINE (forward) rate misclassified as '{opp_v}': {fwd_predicted_opp*100:.1f}%")
    print(f"REVERSED rate classified as '{opp_v}':               {rev_flipped_to_opp*100:.1f}%")
    print(f"reversal-attributable delta:                         {(rev_flipped_to_opp-fwd_predicted_opp)*100:+.1f}pp")
    print(f"McNemar discordant pairs: {flip_to_opp} flipped TO '{opp_v}', {flip_from_opp} flipped AWAY -- p={mcnemar_p:.4f}")

    summary_rows.append({
        'true_verb': true_v, 'opposite_verb': opp_v, 'n': n,
        'fwd_top1_correct_rate': fwd_correct, 'rev_top1_still_correct_rate': rev_still_correct,
        'baseline_fwd_misclassified_as_opp': fwd_predicted_opp,
        'reversed_classified_as_opp': rev_flipped_to_opp,
        'reversal_attributable_delta_pp': (rev_flipped_to_opp - fwd_predicted_opp) * 100,
        'n_flip_to_opp': flip_to_opp, 'n_flip_from_opp': flip_from_opp, 'mcnemar_p': mcnemar_p,
    })

pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_antonym_bias_corrected.csv'), index=False)
print("\nInterpretation: the pairwise similarity-MARGIN shift (nb8_temporal_order_antonym_summary.csv) "
      "is trustworthy because it's differenced per-segment, canceling any static class-imbalance bias. "
      "The raw top-1 argmax over all 282 verbs is NOT similarly protected -- 'add' can win the full "
      "vocabulary argmax almost regardless of the clip, given severe class imbalance (4,721:231 train "
      "ratio) -- so only the reversal-attributable DELTA above baseline should be reported as evidence, "
      "not the raw reversed-classification rate.")
print("\nDONE")
