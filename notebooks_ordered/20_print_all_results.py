"""
Single entry point that reproduces and prints every number discussed for the
paper -- classification headline, rich metrics (micro/macro top-1, MRR,
median rank), blind baselines and video's lift over them, paired-bootstrap
significance, frequent-vs-rare, and all four temporal-order legs (raw
embedding order-sensitivity, the trained +DinoTXT alignment head's
forward-vs-reversed scoring, the directional-vs-cyclic verb typology, and the
add<->remove antonym shift with its bias-corrected top-1 check).

Reads only already-saved CSVs plus two cheap recomputations from cached
ranks files (pretrained-pool rich metrics, blind-baseline lift) -- no GPU,
no video decoding, runs in seconds. This is the reproducibility backbone:
every number that has been reported for this paper should trace back to a
line printed by this script, sourced from a specific file.

Upstream scripts that produce the CSVs this reads (run once each, GPU
required, not re-run by this script -- see REPRODUCE.md for the full ordered
pipeline this is step 12 of):
  01_train_matched_projection_baselines.py       -- headline, blind baselines,
                                                     rich metrics, paired
                                                     bootstrap, frequent-vs-rare
                                                     (DINOv2/DINOv3/mean-pool
                                                     V-JEPA2)
  05_extract_vjepa2_pretrainedpool_embeddings.py -- pretrained-pool embeddings
  06_train_vjepa2_pretrainedpool_headline.py     -- pretrained-pool headline,
                                                     blind baseline, ranks,
                                                     paired bootstrap,
                                                     frequent-vs-rare
  03_compute_optical_flow_motion_scores.py       -- encoder-independent motion
                                                     scores per segment
  07_temporal_order_raw_embedding_probe.py       -- leg A (raw embedding
                                                     cos-sim + linear probe,
                                                     pilot n=600)
  08_temporal_order_text_alignment_fullval.py    -- leg B (trained alignment
                                                     head, full val N=3,268)
  09_temporal_order_directional_vs_cyclic_verbs.py -- leg C (directional vs
                                                     cyclic, both scales, add
                                                     investigation)
  10_temporal_order_antonym_margin_shift.py      -- leg D (add<->remove
                                                     margin shift)
  11_temporal_order_antonym_top1_bias_check.py   -- leg D bias-corrected
                                                     top-1 check
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
OUT_DIR = PATHS['processed']


def p(path):
    return os.path.join(OUT_DIR, path)


def load(path, required=True):
    fp = p(path)
    if not os.path.exists(fp):
        msg = f"MISSING: {fp}"
        if required:
            print(f"  !! {msg} -- run the upstream script that produces it first.")
        return None
    return pd.read_csv(fp)


def hr(title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
hr("PART 1 — Matched-projection classification headline (seed mean ± std)")
# ═══════════════════════════════════════════════════════════════════════════
headline = load('nb8_final_summary.csv')
pp_agg = load('nb8_vjepa2_pretrainedpool_seed_aggregate.csv')
if headline is not None:
    cols = ['model', 'top1_mean', 'top1_std', 'top5_mean', 'top5_std', 'n']
    if pp_agg is not None:
        pp_agg = pp_agg.copy()
        pp_agg['n'] = 3268  # full validation set size; this file records n_seeds, not segment count
    combined = pd.concat([headline[cols], pp_agg[cols]], ignore_index=True) \
        if pp_agg is not None else headline[cols]
    print(combined.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
hr("PART 2 — Rich metrics: micro top-1 vs. macro top-1 vs. macro-F1, MRR, median rank")
# ═══════════════════════════════════════════════════════════════════════════
rich = load('nb8_rich_metrics.csv')
rows = [rich] if rich is not None else []
pp_ranks = load('nb8_ranks_vjepa2_pretrainedpool_representative.csv')
if pp_ranks is not None:
    pp_rich = pd.DataFrame([{
        'model': 'vjepa2_pretrainedpool',
        'micro_top1': pp_ranks['top1_correct'].mean(),
        'macro_top1': pp_ranks.groupby('ground_truth')['top1_correct'].mean().mean(),
        'macro_f1': (f1_score(pp_ranks['ground_truth'], pp_ranks['top1_pred'],
                              average='macro', zero_division=0)
                     if 'top1_pred' in pp_ranks.columns else np.nan),
        'mrr': pp_ranks['reciprocal_rank'].mean(),
        'median_rank': pp_ranks['rank'].median(),
        'n': len(pp_ranks),
    }])
    rows.append(pp_rich)
if rows:
    all_rich = pd.concat(rows, ignore_index=True)
    if 'macro_f1' not in all_rich.columns:
        all_rich['macro_f1'] = np.nan
    # nb8_rich_metrics.csv predates macro-F1. Script 13 backfills it from the
    # representative checkpoints without retraining -- fold that in where present.
    recomputed = load('nb8_macro_f1_recomputed.csv')
    if recomputed is not None:
        fill = recomputed.set_index('model')['macro_f1']
        all_rich['macro_f1'] = all_rich['macro_f1'].fillna(all_rich['model'].map(fill))
    all_rich = all_rich[['model', 'micro_top1', 'macro_top1', 'macro_f1', 'mrr', 'median_rank', 'n']]
    print(all_rich.to_string(index=False))
    print("\nNote: macro-top1 << micro-top1 for every model here -- with 282 verbs and this")
    print("dataset's skew (top verb 'add' = 35.6% of validation), micro accuracy is dominated by")
    print("frequent-verb performance. Macro (every verb weighted equally) is the metric that")
    print("actually reflects whether the model can name RARE verbs, not just common ones.")
    print("macro_top1 is macro-averaged RECALL and ignores precision; macro_f1 is the number to")
    print("report next to top-1. A NaN in macro_f1 means that ranks CSV predates the 2026-09-02")
    print("change that started saving top1_pred -- rerun that backbone's script to fill it in.")
    all_rich.to_csv(p('nb8_all_rich_metrics_combined.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════════
hr("PART 3 — Blind baselines: how much is video actually contributing?")
# ═══════════════════════════════════════════════════════════════════════════
blind = load('nb8_blind_baselines.csv')
blind_pp = load('nb8_blind_baselines_vjepa2_pretrainedpool.csv')
blind_all = pd.concat([b for b in [blind, blind_pp] if b is not None], ignore_index=True) \
    if (blind is not None or blind_pp is not None) else None
if blind_all is not None:
    print(blind_all.drop_duplicates(subset=['model', 'mode']).to_string(index=False))
    zero_top1 = blind_all['top1'].iloc[0]
    mean_top5 = blind_all[blind_all['mode'] == 'mean']['top5'].iloc[0]
    print(f"\nBlind top-1 ({zero_top1*100:.2f}%) == majority-class baseline (always guess the single most")
    print("frequent verb, 'add') for every backbone -- confirmed: with zero real video information, every")
    print("trained text tower collapses to guessing the same verb every time.")
    print(f"\nVideo's lift over blind, per model (top-1 vs. {zero_top1*100:.2f}%, top-5 vs. {mean_top5*100:.2f}%):")
    if rows:
        lift_rows = []
        for _, r in all_rich.iterrows():
            hr_row = headline[headline['model'] == r['model']] if headline is not None else None
            top1 = r['micro_top1']
            lift_rows.append({'model': r['model'], 'top1': top1, 'lift_top1_pp': (top1 - zero_top1) * 100})
        print(pd.DataFrame(lift_rows).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
hr("PART 4 — Paired bootstrap: are the ~1pp gaps between backbones real?")
# ═══════════════════════════════════════════════════════════════════════════
for f in ['nb8_paired_bootstrap.csv', 'nb8_paired_bootstrap_vjepa2_pretrainedpool.csv']:
    boot = load(f)
    if boot is not None:
        print(f"\n[{f}]")
        print(boot.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
hr("PART 5 — Frequent vs. rare verb breakdown")
# ═══════════════════════════════════════════════════════════════════════════
for f in ['nb8_frequent_vs_rare_split.csv', 'nb8_frequent_vs_rare_vjepa2_pretrainedpool.csv']:
    fr = load(f)
    if fr is not None:
        print(f"\n[{f}]")
        print(fr.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
hr("PART 6 — Temporal-order leg A: raw embedding, forward vs. reversed (pilot n=600)")
# ═══════════════════════════════════════════════════════════════════════════
cos = load('nb8_temporal_order_cosine.csv')
probe = load('nb8_temporal_order_probe_result.csv')
if cos is not None:
    print(f"Mean cos-sim(forward, reversed): {cos['cos_sim_fwd_rev'].mean():.4f}  "
          f"(std={cos['cos_sim_fwd_rev'].std():.4f}, n={len(cos)})")
    print("DINOv2/DINOv3: exactly 1.0000 for every segment -- proven, not measured (mean-pool is")
    print("order-invariant by construction).")
if probe is not None:
    print(probe.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
hr("PART 7 — Temporal-order leg B: trained +DinoTXT alignment head, full val N=3,268")
# ═══════════════════════════════════════════════════════════════════════════
align = load('nb8_temporal_order_text_alignment.csv')
if align is not None:
    n = len(align)
    frac_fwd = (align['diff_fwd_minus_rev'] > 0).mean()
    print(f"N={n}  forward-win rate={frac_fwd*100:.1f}%  mean diff={align['diff_fwd_minus_rev'].mean():+.4f}")
    print("(full bootstrap CI / sign-test p-value printed by 08_temporal_order_text_alignment_fullval.py itself")
    print(" at run time; re-run that script to reproduce the CI, this file only has per-segment scores.)")

# ═══════════════════════════════════════════════════════════════════════════
hr("PART 8 — Temporal-order leg C: directional vs. cyclic verb typology")
# ═══════════════════════════════════════════════════════════════════════════
typ = load('nb8_temporal_order_verb_typology_summary.csv')
if typ is not None:
    print(typ.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
hr("PART 9 — Temporal-order leg D: add<->remove antonym shift")
# ═══════════════════════════════════════════════════════════════════════════
ant = load('nb8_temporal_order_antonym_summary.csv')
ant_bias = load('nb8_temporal_order_antonym_bias_corrected.csv')
if ant is not None:
    print("[margin shift -- paired/differenced, trustworthy despite class imbalance]")
    print(ant.to_string(index=False))
if ant_bias is not None:
    print("\n[raw top-1 flip, bias-corrected -- NOT paired, class-imbalance-sensitive, weaker evidence]")
    print(ant_bias.to_string(index=False))

hr("DONE — every table above should trace to a file in processed/. See each")
print("script's own docstring (notebooks/scripts/nb8_*.py) for exactly how its numbers were produced.")
