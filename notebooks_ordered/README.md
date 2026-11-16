# Run order

Files are numbered 01-22 in the order they run — same order as `REPRODUCE.md`
(also included here), which has the full detail on what each one reads and
writes. Notebooks (01-06) are opened and run top to bottom; scripts (07-22) are
run from wherever `paths.json` lives, as `python3 07_train_matched_projection_baselines.py`
etc.

Several steps hard-read a file an earlier one writes, so **run them in order** —
skipping ahead crashes partway rather than quietly producing wrong numbers.

**Before running anything: open `paths.json` and point every path at wherever
this project's data lives on your machine** (videos, embeddings, checkpoints,
annotations, processed, repos). It's read by every numbered file here.

`requirements.txt` is the Python environment (3.12) the results were produced in.

01  environment_setup
02  youcook2_download_parsing
03  verb_extraction
04  frame_extraction_dinov2_embeddings        — also writes segments_verbs_timed_clean.csv,
                                                 which every later step reads
05  frame_extraction_dinov3_embeddings
06  vjepa2_dinotxt_integration
07  train_matched_projection_baselines         — ~11-12h, the longest step
08  extract_ranks_and_frequency_split
09  compute_optical_flow_motion_scores
10  extract_vjepa2_base_meanpool_embeddings    — ~10h, the single most expensive step
11  train_vjepa2_base_meanpool_baseline
12  extract_vjepa2_pretrainedpool_embeddings
13  train_vjepa2_pretrainedpool_headline       — the headline model
14  temporal_order_raw_embedding_probe
15  temporal_order_text_alignment_fullval
16  temporal_order_probe_split_level_check     — runs after 15, reads the cache 15 writes
17  temporal_order_directional_vs_cyclic_verbs
18  temporal_order_antonym_margin_shift
19  temporal_order_antonym_top1_bias_check
20  print_all_results                          — start here to see current results:
                                                 reads only saved CSVs, no GPU, seconds
21  recompute_macro_f1_from_checkpoints
22  extract_dinov2_perframe_embeddings         — the fair-baseline re-extraction; the
                                                 order-aware head that uses it isn't written yet

## Time estimate

The full pipeline is roughly 22-24h of GPU time end to end. Two steps dominate
that: **07** (train_matched_projection_baselines, ~11-12h — an LR sweep plus 5
seeds x 10 epochs across three backbones) and **10**
(extract_vjepa2_base_meanpool_embeddings, ~10h, almost entirely video decoding).
Everything else in the list is under 2.5h, and most are well under 1h.
