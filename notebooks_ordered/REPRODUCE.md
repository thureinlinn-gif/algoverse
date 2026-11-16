# Reproducing these results

This reproduces the paper's core comparison: DINOv2, DINOv3, and V-JEPA2, each
combined with DinoTXT, evaluated fairly on YouCook2 cooking-verb classification
(matched-projection protocol — see each script's docstring), and the temporal-order
findings that became the paper's actual claim (V-JEPA2+DinoTXT encodes clip
direction that *frame-averaged* DINOv2/DINOv3+DinoTXT provably cannot).

Take "frame-averaged" literally. The impossibility is a property of the mean this
pipeline pools with, not of DINOv2's features — see Phase 6, which builds the
order-aware DINOv2 baseline that makes the comparison fair, and note that it has
not been run yet.

The phases below are the story: get the data, get the verbs, set up the
frame-based backbones, set up V-JEPA2 properly, the results, then the fair
temporal baseline. Follow them in order — several steps have hard-coded file
reads with no fallback, so running out of order crashes partway through rather
than silently producing wrong numbers.

Before quoting any number out of here, read the two sections directly below: they
fix which N every table is over, and why Top-1 alone overstates every model on
this dataset.

`08_mentor_requested_analyses.ipynb` is **stale** — it predates the pooling-confound
fix and all temporal-order work below. Don't treat it as the source of truth.

## Environment

```
source venv/bin/activate   # python 3.12, see venv/pyvenv.cfg
```

`paths.json` at the repo root is read by every step (`PATHS['embeddings']`,
`PATHS['processed']`, `PATHS['checkpoints']`, `PATHS['annotations']`,
`PATHS['videos']`, `PATHS['repos']`). Scripts (as opposed to notebooks) are run
from the repo root: `python3 notebooks/scripts/<script>.py`.

---

## The canonical evaluation set — read before quoting any N

Every number reported for this paper is computed on **the 3,268 validation
segments** of `annotations/segments_verbs_timed_clean.csv`. That file holds
13,003 segments in total: 9,735 training + 3,268 validation. Any other N in a
table is a subset, and the table has to say so.

| N | what it is |
|---|---|
| 13,003 | the whole annotated set (train + val). Never an evaluation number. |
| 9,735 | training split. Only ever used to fit the projection head. |
| **3,268** | **the canonical evaluation set.** Every headline number uses this. |
| 600 | step 7's deliberate pilot subsample of the 3,268 (`N_SAMPLE`, seed 0), because it needs a fresh GPU extraction per segment. Superseded at full scale by step 7b. |

Verb counts follow from that: `add` is **1,165 of 3,268 = 35.65%** of the
evaluation set (and 4,721 of 13,003 overall). 35.65% is therefore both the
majority-class rate and the blind-baseline Top-1 — that agreement is the point of
the blind baseline, not a coincidence.

**The split is by video, not by segment**, and always was: `subset` comes from
YouCook2's own official train/val split, which is defined over videos. Verified
directly — 1,321 training videos and 450 validation videos, **zero video ids in
common**, and 13,003 distinct segment ids over 13,003 rows. No clip from a
training video appears in validation.

Two things were wrong here before 2026-09-02, both now fixed:

* **Step 9 reported n=3,567.** It built its table by globbing the cached
  forward/reversed embedding directories rather than by selecting the evaluation
  set, and that cache had picked up 299 *training* segments from an earlier
  exploratory run. Nothing in step 9 is trained, so nothing leaked — but it made
  that one table disagree with every other table in the paper (`add` n=1,313
  rather than 1,165; `remove` n=206 rather than 55). Now restricted to the 3,268.
  **This changed a reported result** — see the note under step 9.
* **Step 7's probe split by segment_id, not video_id.** Now splits by video.
  Step 7b measures what that was worth: **+0.15pp**, against 0.79pp seed-to-seed
  noise. The probe number was not inflated. Details under step 7b.

## Reporting metrics: macro-F1 belongs next to Top-1

One verb is 35.65% of the evaluation set, so Top-1 flatters everything, including
a model with no video input at all. Scored side by side on the 3,268:

| model | Top-1 | macro-F1 | distinct verbs ever predicted |
|---|---|---|---|
| blind (always `add`) | 35.65% | 0.32% | 1 / 282 |
| DINOv2 + DinoTXT | 42.87% | 5.37% | 42 / 282 |
| V-JEPA2 (mean-pool) + DinoTXT | 38.71% | 1.20% | 17 / 282 |
| V-JEPA2 (pretrained pooler) + DinoTXT | 43.64% | 5.39% | 46 / 282 |

Read that second column before quoting the first. The headline
V-JEPA2-vs-DINOv2 classification gap is **+0.77pp on Top-1 but +0.02pp on
macro-F1** — on the metric that does not reward guessing `add`, those two are
indistinguishable. The paper's claim should rest on the temporal-order results
(Phase 5), which is where the two genuinely differ, not on the classification
headline.

Note `macro_top1` in the older CSVs is macro-averaged *recall* — it ignores
precision, so it does not penalise over-predicting one class. `macro_f1` is the
number to report.

## Phase 0 — Environment

**`01_environment_setup (1).ipynb`** — one-time dependency setup. No data yet.

---

## Phase 1 — Get YouCook2

**`02_youcook2_download_parsing.ipynb`**
Needs: nothing (fresh start).
Produces: `videos/*.mp4`, `annotations/segments.csv`, `annotations/segments_available.csv`.

---

## Phase 2 — Get the verbs

**`03_verb_extraction.ipynb`**
Needs: `annotations/segments_available.csv` (Phase 1).
Produces: `annotations/segments_verbs.csv`, `annotations/spotcheck_sample.csv`.

This file isn't final yet — no timing, not filtered for segments that fail
embedding later. It gets completed in Phase 3's first step; don't be surprised that
"get the verbs" spans two phases.

---

## Phase 3 — Set up DINOv2 + DINOv3 with DinoTXT

**`04_frame_extraction_dinov2_embeddings.ipynb`**
Needs: `segments.csv` (Phase 1) + `segments_verbs.csv` (Phase 2).
Produces: `embeddings/dinov2/*.pt`, **and**
**`annotations/segments_verbs_timed_clean.csv`** — the single most-depended-on file
in the whole pipeline; every step from here on reads it. This notebook does double
duty: DINOv2 embeddings *and* finalizing the verb annotation. Don't skip it thinking
it's "just" the embeddings step.

**`04b_frame_extraction_dinov3_embeddings.ipynb`**
Needs: same inputs as above (independent of its output — can run in parallel).
Produces: `embeddings/dinov2/dinov3/*.pt`.

**`06_vjepa2_dinotxt_integration.ipynb`** — yes, V-JEPA2, but it belongs here, not
Phase 4. The next script trains all three backbones together in one run, so V-JEPA2
needs its (naive, unfixed) embeddings ready at the same time as DINOv2/DINOv3's.
This is V-JEPA2's simplest setup (mean-pooling) — not the version the paper actually
uses; that's Phase 4.
Needs: `segments_verbs_timed_clean.csv`.
Produces: `embeddings/dinov2/vjepa2_dinotxt/*.pt`.

**`01_train_matched_projection_baselines.py`** — trains a DinoTXT linear-projection
head for DINOv2, DINOv3, and V-JEPA2(mean-pool) together, evaluates all three
identically (matched trainable-parameter budget, per-backbone LR sweep, **2 seeds**,
standardized epoch selection). Also logs val macro-F1 per epoch and writes the
predicted verb (`top1_pred`) into the ranks CSVs.

*Changed 2026-09-02: `N_SEEDS` 2 -> 5*, then *reverted 2026-09-08: `N_SEEDS` 5 -> 2*
per mentor review. The original concern was real — two seeds cannot support a claim
about a ~0.4pp gap between backbones, and they make the "representative checkpoint"
rule (the seed whose Top-1 is closest to the mean) an exact tie broken by
floating-point rounding, which is how script 01 and script 02 ended up disagreeing
about which V-JEPA2 checkpoint was representative. But the ~12h rerun this would
have cost isn't buying anything here: the comparison it exists to support
(DINOv2 vs. V-JEPA2-pretrainedpool macro-F1, 5.37 vs. 5.39) is already
indistinguishable at 2 seeds, so a tighter mean/std estimate can't change the
conclusion. Decision: skip the rerun, report 2 seeds, keep the checkpoint-tie
caveat on record (see script 02's `REPRESENTATIVE` note) rather than pay for a
run that wouldn't resolve it anyway.
Needs: all three embedding directories above.
Produces: `nb8_final_summary.csv`, `nb8_matched_projection_*`, `nb8_blind_baselines.csv`,
`nb8_rich_metrics.csv`, `nb8_paired_bootstrap.csv`, `nb8_motion_vs_static_split.csv`.
~11-12h (LR sweep + 5 seeds × 10 epochs × 3 backbones) — the longest single step.

**`02_extract_ranks_and_frequency_split.py`**
Needs: checkpoints from the script above (reloads, does not retrain).
Note its `REPRESENTATIVE` dict hardcodes checkpoint filenames. Re-point them after
any retraining — with 5 seeds the selected epochs will differ from the current
2-seed ones, and a stale filename here silently reports the wrong head.
Produces: `nb8_ranks_{dinov2,dinov3,vjepa2}_representative.csv`, `nb8_frequent_vs_rare_split.csv`.
~30-45 min.

**`03_compute_optical_flow_motion_scores.py`** — encoder-independent motion
measurement (Farneback flow on raw pixels, no DINO/V-JEPA2 involved), reused again
in Phase 5.
Needs: ranks above.
Produces: `nb8_optical_flow_scores.csv`, `nb8_optical_flow_by_verb.csv`.
~45 min.

---

## Phase 4 — Set up V-JEPA2 *properly* with DinoTXT

Phase 3's V-JEPA2 line was naive mean-pooling — worse than DINOv2/DINOv3, badly so
on motion verbs. This phase proves that's a pooling artifact, not a real deficit,
and fixes it. **This is the config the paper's classification numbers actually use.**

**`04a_extract_vjepa2_base_meanpool_embeddings.py` → `04b_train_vjepa2_base_meanpool_baseline.py`**
Needs: raw video (independent extraction, a different checkpoint than Phase 3 used).
Produces: `nb8_ranks_vjepa2base_representative.csv` — rules out "wrong checkpoint"
as the explanation. Not a headline result on its own, but the next step hard-reads
this file with no fallback, so it's a real prerequisite despite being background
context. ~10h (64-frame checkpoint extraction dominates) — the most expensive step
in the whole pipeline; see the cost note at the bottom.

**`05_extract_vjepa2_pretrainedpool_embeddings.py`** — the actual fix: V-JEPA2's
own frozen, pretrained SSv2 attentive pooler, instead of naive mean-pooling.
Produces: `embeddings/dinov2/vjepa2_pretrainedpool/*.pt`. ~2.4h.

**`06_train_vjepa2_pretrainedpool_headline.py`**
Needs: ranks from Phase 3's step 2 and this phase's base-checkpoint step, flow
scores from Phase 3's step 3, embeddings from the step above.
Produces: `nb8_vjepa2_pretrainedpool_seed_aggregate.csv`,
`nb8_blind_baselines_vjepa2_pretrainedpool.csv`,
`nb8_ranks_vjepa2_pretrainedpool_representative.csv`,
`nb8_paired_bootstrap_vjepa2_pretrainedpool.csv`,
`nb8_frequent_vs_rare_vjepa2_pretrainedpool.csv`,
`nb8_optical_flow_tercile_vjepa2_pretrainedpool.csv`. ~2.4h at 2 seeds, ~6h at 5
(raised to 5 on 2026-09-02 to match script 01; also logs macro-F1 and `top1_pred`).

---

## Phase 5 — Results (the paper's actual claim)

Everything below needs the trained checkpoint from Phase 4's last step.

**`07_temporal_order_raw_embedding_probe.py`** — leg A: does the raw pooled
embedding even change when a clip is reversed? (DINOv2/DINOv3 provably cannot;
V-JEPA2 does — cos-sim 0.80 not 1.0, linear probe 69% vs. a proven 50% ceiling.)
Needs: prerequisite embeddings only. Produces: `nb8_temporal_order_cosine.csv`,
`nb8_temporal_order_probe_result.csv`. ~20 min.
The probe now splits **by video id** (changed 2026-09-02 from segment id).

**`08_temporal_order_text_alignment_fullval.py`** — leg B: does the *trained*
alignment head score a clip's true verb higher forward than reversed, at full
validation scale? Also caches the forward/reversed embeddings steps 9-10 reuse.
Needs: checkpoint from Phase 4. Produces: `nb8_temporal_order_text_alignment.csv`. ~1h.

**`07b_temporal_order_probe_split_level_check.py`** — runs *after* step 8, not
after step 7: it reuses the forward/reversed embeddings step 8 caches, so on a
fresh run there is nothing for it to read until step 8 has finished. Numbered 07b
because it belongs to step 7's claim. It is the evidence that the split
level was not doing the work. Reuses step 8's cached forward/reversed embeddings,
so no GPU extraction and no video decoding: full canonical evaluation set
(n=3,268, not step 7's 600), 5 random splits at each split level.

    split by segment   70.34% +/- 1.57
    split by video     70.19% +/- 0.79      (chance = 50.00%)

**+0.15pp**, against 0.79pp seed-to-seed noise at the video level — the
segment-level split was not inflating anything, and 70.2% at full scale
corroborates step 7's 69.2% pilot. Needs step 8's cached embeddings. Produces:
`nb8_temporal_order_probe_split_level.csv`. Seconds — no GPU extraction.

**`09_temporal_order_directional_vs_cyclic_verbs.py`** — leg C: does that
sensitivity concentrate on verbs where reversal changes real meaning (directional)
vs. verbs that are inherently back-and-forth (cyclic)? Includes the "add" exception
investigation (ruled out noise, ruled out low motion).
Needs: step 7's output, step 3's flow scores, step 8's cached embeddings.

**Result changed 2026-09-02.** This step used to analyse every cached
forward/reversed pair (n=3,567, including 299 training segments); it now uses the
canonical 3,268. The typology claim survives but one number moved across the
significance line:

| | before (n=3,567) | after (n=3,268) |
|---|---|---|
| directional **incl.** `add` < cyclic | p = 0.022 | **p = 0.053** |
| directional **excl.** `add` < cyclic | p = 1.9e-09 | p = 1.0e-07 |

Report the excl.-`add` comparison as the finding — it is what the directional
/cyclic claim actually rests on, it is unambiguous either way, and `add` was
always documented as the exception to the pattern. The incl.-`add` test is not
significant at 0.05 on the canonical set and should not be quoted as if it were.

Produces: `nb8_temporal_order_cosine_fullscale.csv`,
`nb8_temporal_order_directional_cyclic_fullscale.csv`,
`nb8_temporal_order_verb_typology_summary.csv`. Seconds — no GPU.

**`10_temporal_order_antonym_margin_shift.py`** — leg D: does reversing "add"
shift the similarity margin toward its natural opposite, "remove" (and vice versa)?
Needs: checkpoint from Phase 4. Produces: `nb8_temporal_order_antonym_add_remove.csv`,
`nb8_temporal_order_antonym_summary.csv`. ~20 min.

**`11_temporal_order_antonym_top1_bias_check.py`** — leg D's bias-corrected
top-1 check, controlling for add:remove's 4,721:231 class imbalance.
Needs: checkpoint from Phase 4, cached embeddings from step 10.
Produces: `nb8_temporal_order_antonym_top1_flips.csv`,
`nb8_temporal_order_antonym_bias_corrected.csv`. ~1 min.

**`12_print_all_results.py`** — prints and exports every table above, consolidated.
Needs: everything above. Produces: printed output + `nb8_all_rich_metrics_combined.csv`.
Now reports macro-F1 next to Top-1, backfilling it from step 13 where a ranks CSV
predates the `top1_pred` change. Seconds.

**`13_recompute_macro_f1_from_checkpoints.py`** — macro-F1 for the *already
trained* representative checkpoints, no retraining. Originally written so the
metric could be reported before committing to a planned 5-seed rerun; that rerun
was subsequently skipped (2026-09-08, see script 01's note above) since it
wasn't going to change an already-indistinguishable comparison, so this script's
2-seed numbers are now the source, not a cross-check. It identifies the
representative checkpoint by finding the one that
reproduces the published ranks CSV per-segment, rather than re-deriving the
tie-broken "closest to the mean" rule — and refuses to report a number for any
backbone where no checkpoint on disk reproduces the published file.
Needs: `checkpoints/nb8_*_seed*_bestepoch*.pt` + the full-history CSVs.
Produces: `nb8_macro_f1_recomputed.csv`, and adds `top1_pred` to the ranks CSVs.
~5 min. **DINOv3 is skipped**: it needs
`/home/ubuntu/weights/dinov3_vitl16_{pretrain,dinotxt_...}.pth`, which are not in
this repo and are absent from the current box — restore them to get its row.

---

---

## Phase 6 — The fair temporal baseline (not yet run)

The temporal-order claim currently compares V-JEPA2's temporal pooling against a
DINOv2 arm that *cannot* represent order, because notebook 04 saved
`frame_embeddings.mean(dim=0)` and discarded the time axis. That mean is
order-invariant by construction, which makes the comparison unanswerable as
stated: the baseline was built to lose. The claim that survives is the narrow
one — *frame-averaged* DINOv2 cannot encode order — not that DINOv2 features
cannot.

**The per-frame features are not on disk.** `embeddings/dinov2/*.pt` are pooled
(2048,) vectors, one per segment; `frames/` is empty. So this needs a
re-extraction, which is what step 14 is.

**`14_extract_dinov2_perframe_embeddings.py`** — same frame sampling, same eval
transform, same frozen `encode_image` as notebook 04, saving the (T, 2048) stack
instead of its mean. The point is that the published mean-pool baseline is
recoverable exactly from these files (`per_frame.mean(dim=0) == pooled`, verified
to ~5e-5 on this box), so an order-aware head trained on them is compared against
the published number over *identical* features, with pooling as the only
difference. ~2-3h on one A10, ~5.3 GB, resumable.
Produces: `embeddings/dinov2/dinov2_perframe/*.pt`.

Then the head itself, which is not yet written: same trainable-parameter budget
and same LR sweep / 5 seeds / epoch-selection protocol as script 01, over the
per-frame features, with a pooling that can see order — a small temporal
transformer or GRU with positional encoding, or, as the cheapest honest version,
concatenating the first-half and second-half means (which is order-sensitive by
construction and adds almost no parameters). Report it beside mean-pool DINOv2 and
V-JEPA2 on both the classification numbers and the step 7b order probe.

**The head must inject temporal position explicitly. Attention alone is not
enough.** `not_required/nb8_attnpool_poc.py`'s `AttentionPool` — one learned query
attending over the timestep stack, no positional encoding — is permutation
invariant over time, exactly like the mean it was meant to improve on. Checked
directly: feeding it a (4, 32, 1024) stack and the same stack with
`.flip(1)` gives outputs agreeing to 9.5e-07, i.e. identical to float precision.

That head was harmless where it was used, because V-JEPA2's encoder writes
temporal position into the token *values* before pooling ever happens, so order
survives regardless of what the pooler does. Per-frame DINO features have no such
property: each frame is encoded independently, so order lives only in the sequence
itself and an order-blind pooler destroys it. Reusing that head over
`dinov2_perframe` would reproduce the exact flaw Phase 6 exists to fix, while
looking like a fix.

Expected outcome worth stating in advance: an order-aware DINOv2 head will
probably score above 50% on the order probe, and the honest claim becomes a
quantitative gap rather than an in-principle impossibility.

## Phase 7 — Where would V-JEPA2's paper-documented advantage show up? (run 2026-10-24)

Near-zero-GPU sandbox before any compute request. Papers: V-JEPA2 beats image
models on motion benchmarks (SSv2, Diving-48) under a trained attentive probe over
spatiotemporal tokens, and ties them on appearance benchmarks (COIN, the closest to
YouCook2: 90.7 vs 90.7). Two gaps between this project and the papers were tested:
the metric (Top-1 rewards appearance) and the protocol (pooled vector + linear head).

Environment on the 2026-10 box: the repo's `venv/` targets Python 3.12, which the
box lacks. Rebuilt at `/home/ubuntu/venv-yc2` (uv-managed CPython 3.12) with
`uv pip install --no-deps -r requirements.txt` -- the freeze pins numpy 1.26.4
while opencv-python-headless 4.13 declares numpy>=2, so normal resolution fails.

**`32_cache_text_features.py`** — DinoTXT verb features before
`linear_projection` (the last op before first-token pooling, so exact). Verified
to reproduce `encode_text(normalize=True)` to 6.6e-7 through the original and two
trained projections. Seconds. DINOv3 is added automatically once its weights are in
`/home/ubuntu/weights/`.

**`33_linear_head_lib.py`** — script 07's recipe and helpers on cached features;
`python3 33_linear_head_lib.py` runs the gate. Passed: every representative head
reproduces its published ranks (99.97-100%); retraining lands at 43.15 / 43.59 vs
published 42.96 / 43.62 (tol 0.3pp). Representatives: dinov2 seed0/ep9,
dinov2halves seed0/ep10, vjepa2base **seed1/ep9**, vjepa2_pretrainedpool seed1/ep9.
<1 min. Writes `nb8_phase7_gate.csv`.

**`34_hard_negative_verb_test.py`** (Experiment A, N=3,268) — each segment scored
against the verb its masked caption most suggests (TF-IDF+LR, C=10 by grouped CV;
val Top-1 58.02%; train segments get out-of-fold negatives for script 35). Within-pair
accuracy, chance 50%:

| head | all | excl. add | synonym (606) | same object (2,007) | blind (prior only) |
|---|---|---|---|---|---|
| DINOv2 mean-pool | 63.6 | 57.3 | 64.9 | 61.5 | 54.3 |
| DINOv2 order-aware halves | 63.8 | 57.3 | 64.7 | 62.0 | 54.3 |
| V-JEPA2 label-free mean-pool | 59.5 | 50.1 | 65.5 | 56.8 | 54.3 |
| V-JEPA2 SSv2-pooler (headline) | 64.9 | 58.9 | 66.2 | 63.0 | 54.2 |

Headline V-JEPA2 − best DINO (halves): **+1.10pp [−0.22, +2.44]**, video-cluster CI —
null under the pre-registered ≥3pp rule, with an 80%-power MDE of ~1.9pp, so a 3pp
advantage is ruled out, not missed. Label-free V-JEPA2: −4.35pp [−5.52, −3.14].
Controls: uniform-random negatives 96-97% for all heads. ~4 min CPU.
`SYNONYM_GROUPS` is a judgment call, exported to `nb8_hard_negative_synonym_groups.csv`.

**`35_attentive_probe_poc.py`** (Experiment B, N=1,500 POC segments, 5-fold
video-grouped CV, zero extraction) — **inconclusive: failed its own verification.**
Probe: 1 block, width 256, learned temporal position embedding, one-query
cross-attention, text side over cached features. Mean-pool control = same module on
the time mean. Checks that passed: zero video overlap across folds; time-mean of
`vjepa2_poc_temporal` matches `vjepa2_base_dinotxt` (median cosine 1.0000, so the
main arm is the label-free checkpoint); label shuffle falls to the prior. Check that
**failed**: permuting the time axis at test changes nothing (≤0.1pp) for any arm,
at position-embedding init std 0.02 and again at 0.5 (`POS_INIT_STD=0.5`, outputs
suffixed `_pos0.5`). The probes also barely clear the prior (Top-1 36.8-38.8 vs
35.9; early-stopped at 3-5 epochs). With ~1,200 training examples and a text
alignment learned from scratch, the probe does not learn order, so B says nothing
about whether the token-level protocol helps V-JEPA2. For the record (std 0.02):
V-JEPA2 attn − DINOv2 attn on hard pairs −2.60pp [−4.32, −0.90]; interaction
(attn−mean gain, V-JEPA2 minus DINOv2) −0.57pp [−1.95, +0.82]. ~10 min on one A10.

**`36_youcook2_appearance_diagnostic.py`** (Experiment C, N=3,268) — the null explained:
masked-caption text alone 58.0% Top-1 vs every video head 39-44%; one DINOv2 frame
41.9% vs full clip 42.9%; where the text model is wrong (n=1,372 — matches the
earlier unsaved n=1,377) every head falls to 9.5-16.7%. DINOv3's single-frame
number (34.8%, below always-"add") comes from `not_required/nb8_singleframe_motion_test.py`
and looks like a preprocessing artifact of that script — don't cite its +8.2pp
clip-over-frame gap without rechecking. ~1 min.

**Status against the pre-registered decision rule:** A is a well-powered null; B is
not a valid null (verification failed), so neither the "B positive" nor the "both
null" row strictly applies. DINOv3 rows are missing until its weights are restored
(then rerun 32, 33, 34, 36).

## Cost note: the ~10h step you might want to skip

**Total GPU time for a full from-scratch run: roughly 22-24 hours**, of which
Phase 4's base-checkpoint extraction alone is ~10h — and its only purpose is
satisfying a hard-coded comparison read inside the headline step
(`nb8_ranks_vjepa2base_representative.csv` is never a headline claim, just
background context showing the SSv2 finetuning wasn't the source of the mean-pool
deficit). **If reproduction cost matters, the highest-value fix is patching
`06_train_vjepa2_pretrainedpool_headline.py`** to make that one comparison optional
(wrap the `pd.read_csv('nb8_ranks_vjepa2base_representative.csv')` call in a
try/except) rather than actually running the 10h extraction — this would cut total
time to ~13-14h with no loss to any claim in the paper. Not done here since it
changes a script's behavior rather than just documenting it.

## Not needed — exploratory dead ends

Moved to `notebooks/scripts/not_required/` (own README explaining each file's
status) so they're out of the way of anyone working through the phases above. None
of their outputs are read by anything above — their conclusions are already baked
into choices the required pipeline makes (e.g. "use the pretrained pooler, not
naive mean-pool" was decided by a POC before Phase 4 was ever run at full scale).
Also not needed among the top-level notebooks, confirmed rather than assumed:
`05_vjepa2_verb_classification (1).ipynb` writes to a *different*, unused embedding
directory; `05_dinotxt_zeroshot_verb_prediction.ipynb` and
`05_dinov3_dinotxt_zeroshot_verb_prediction.ipynb` are optional cache warmers Phase
3's first script regenerates itself if absent;
`02_youcook2_mentor_presentation_cells.ipynb` duplicates Phase 1 cells for a mentor
demo; `DEPRECATED_03_verb_extraction_ground_truth.ipynb` is explicitly deprecated.

## Outstanding caveats (not yet resolved — flag if a reviewer asks)

- The motion/static verb bucket labels used in Phase 3's `nb8_motion_vs_static_split.csv`
  are one person's categorization, exported to `nb8_verb_motion_split_for_review.csv`
  for a second independent rater; that rating was never collected. Superseded by
  Phase 5's directional/cyclic typology for the paper's actual claims, so this
  caveat applies only if the older motion/static split is cited anywhere.
- Verb-extraction spot-check precision (`annotations/spotcheck_sample.csv`,
  `manual_correct` column) has not been manually reviewed.
- DINOv3's macro-F1 is missing (weights absent from the box, see step 13). Every
  other backbone has one.
- The 5-seed rerun of scripts 01 and 06 was **deliberately skipped** (2026-09-08,
  mentor call) — the numbers in `processed/` are from the 2-seed run, and the
  scripts are now reverted to `N_SEEDS=2` to match. Rationale: the comparison the
  extra seeds existed to sharpen (DINOv2 vs. V-JEPA2-pretrainedpool macro-F1,
  5.37 vs. 5.39) is already indistinguishable at 2 seeds, so the ~12h rerun
  wouldn't change the conclusion. The macro-F1 column in the table above comes
  from step 13 over those same 2-seed checkpoints. The checkpoint-selection tie
  noted in script 02 (only breakable by floating-point rounding at N_SEEDS=2)
  remains an open caveat, not resolved by this decision.
- Phase 6 is written but not run: step 14 extracts the per-frame features, and the
  order-aware DINOv2 head that uses them does not exist yet.
- The add↔remove antonym finding (Phase 5, step 10) is solid at the
  similarity-margin level (paired/differenced, controls for class imbalance by
  construction) but the raw top-1 classification-flip framing (step 11) is weaker —
  only a borderline bias-corrected +6pp effect (p=0.05) for remove→add, and no
  significant effect for add→remove. Report the margin-shift number as the primary
  evidence, the top-1 number only with its caveat attached.
